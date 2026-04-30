#include "offload/runtime.cuh"
#include "task/config.cuh"
#include "task/gemv.cuh"
#include "task/gemv_multicast.cuh"
#include "task/utils.cuh"
// #include "dae/context.cuh"

#include <torch/extension.h>

#include <cuda.h> // Driver API
#include <cuda_runtime.h>

#include <vector>
#include <cstdint>
#include <stdio.h>
#include <iostream>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

enum KERNEL_OPCODES {
    GEMV = 0,
    GEMV_HORIZONTAL = 1,
    GEMV_MULTICAST = 2,
    GEMV_HORIZONTAL_MULTICAST = 3,
    GEMV_MULTICAST_PREFILL = 4,
    GEMV_HORIZONTAL_MULTICAST_PREFILL = 5,
};


namespace cde = cuda::device::experimental;
using half_t = cutlass::half_t;



template <typename T>
static inline T* check_tensor_ptr(torch::Tensor t, const char* name) {
  TORCH_CHECK(t.defined(), name, " must be defined");
//   TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(t.scalar_type() == torch::kUInt8, name, " must be uint8");
  TORCH_CHECK(t.dim() == 2, name, " must be rank-2");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");

  const int64_t rows = t.size(0);
  const int64_t cols = t.size(1);

  TORCH_CHECK(cols == (int64_t)sizeof(T),
              name, " second dimension must equal sizeof(T) = ",
              sizeof(T), " but got ", cols);

  // Now memory layout is guaranteed to be:
  // rows contiguous records of sizeof(T) bytes each.
  auto* p = reinterpret_cast<T*>(t.data_ptr<uint8_t>());

  // Alignment safety (important for 16-byte aligned structs)
  uintptr_t addr = reinterpret_cast<uintptr_t>(p);
  TORCH_CHECK(addr % alignof(T) == 0,
              name, " misaligned pointer: address mod alignof(T) = ",
              (addr % alignof(T)));

  return p;
}

// definition of instruction formats
struct CInst {
  uint16_t opcode;
  uint16_t args[3];
};

// function 3: build TMA descriptors
static inline CUtensorMapSwizzle to_swizzle(int64_t swizzle_bytes)
{
    switch (swizzle_bytes)
    {
    case 0:
        return CU_TENSOR_MAP_SWIZZLE_NONE;
    case 32:
        return CU_TENSOR_MAP_SWIZZLE_32B;
    case 64:
        return CU_TENSOR_MAP_SWIZZLE_64B;
    case 128:
        return CU_TENSOR_MAP_SWIZZLE_128B;
    default:
        TORCH_CHECK(false, "Unsupported swizzle_bytes=", swizzle_bytes, " (expected 0/32/64/128)");
    }
}

static inline CUtensorMapDataType to_dtype(torch::ScalarType st)
{
    // Extend as you need
    switch (st)
    {
    case torch::kFloat16:
        return CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    case torch::kBFloat16:
        return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    case torch::kFloat32:
        return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    case torch::kUInt8:
        return CU_TENSOR_MAP_DATA_TYPE_UINT8;
    case torch::kInt32:
        return CU_TENSOR_MAP_DATA_TYPE_INT32;
    case torch::kUInt32:
        return CU_TENSOR_MAP_DATA_TYPE_UINT32;
    default:
        TORCH_CHECK(false, "Unsupported tensor dtype for TMA: ", c10::toString(st));
    }
}

// Build a CUtensorMap descriptor for a tensor.
// Arguments that must be consistent with your kernel's expected layout.
//
// shape:          sizes in elements, rank R
// strides_bytes:  strides in BYTES, rank R  (yes, bytes; not elements)
// box_dim:        tile dimensions in elements, rank R
// elem_strides:   element strides inside the tile, rank R (often all-ones)
// swizzle_bytes:  0/32/64/128
// interleave:     0 for NONE, 1 for 16B, 2 for 32B (optional; use NONE if unsure)
// l2_promo:       0 NONE, 1 64B, 2 128B, 3 256B (varies; use 256B commonly)
// oob_fill:       0 NONE, 1 NAN (float) etc (usually NONE)
torch::Tensor py_build_tma_desc(
    torch::Tensor base,                 // CUDA tensor providing base_ptr + device
    std::vector<int64_t> shape,         // length R
    std::vector<int64_t> strides_bytes, // length R
    std::vector<int64_t> box_dim,       // length R
    std::vector<int64_t> elem_strides,  // length R
    int64_t swizzle_bytes)
{
    TORCH_CHECK(base.defined(), "base must be defined");
    // TORCH_CHECK(base.is_cuda(), "base must be a CUDA tensor");
    TORCH_CHECK(base.numel() > 0, "base must have storage");
    TORCH_CHECK(shape.size() == strides_bytes.size() + 1, "shape and strides_bytes must have same length");
    TORCH_CHECK(shape.size() == box_dim.size(), "shape and box_dim must have same length");
    TORCH_CHECK(shape.size() == elem_strides.size(), "shape and elem_strides must have same length");

    const int R = (int)shape.size();
    TORCH_CHECK(R >= 1 && R <= 5, "tensorRank=", R, " not supported here (adjust if needed)");

    // Allocate descriptor storage on device as opaque bytes
    auto desc = torch::empty({(int64_t)sizeof(CUtensorMap)},
                             torch::TensorOptions().dtype(torch::kUInt8));

    // Prepare arrays
    std::vector<cuuint64_t> gdim(R);
    std::vector<cuuint64_t> gstride(R);
    std::vector<cuuint32_t> bdim(R);
    std::vector<cuuint32_t> estride(R);

    for (int i = 0; i < R; i++)
    {
        TORCH_CHECK(shape[i] > 0, "shape[", i, "] must be > 0");
        TORCH_CHECK(box_dim[i] > 0, "box_dim[", i, "] must be > 0");
        TORCH_CHECK(elem_strides[i] > 0, "elem_strides[", i, "] must be > 0");
        gdim[i] = (cuuint64_t)shape[i];
        bdim[i] = (cuuint32_t)box_dim[i];
        estride[i] = (cuuint32_t)elem_strides[i];

        if (i < R - 1)
        {
            TORCH_CHECK(strides_bytes[i] > 0, "strides_bytes[", i, "] must be > 0");
            gstride[i] = (cuuint64_t)strides_bytes[i];
        }
    }

    CUtensorMapDataType dtype = to_dtype(base.scalar_type());
    CUtensorMapSwizzle swz = to_swizzle(swizzle_bytes);

    CUtensorMapInterleave interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
    CUtensorMapL2promotion l2p = CU_TENSOR_MAP_L2_PROMOTION_L2_256B;
    CUtensorMapFloatOOBfill oob = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

    // Fill descriptor in device memory
    CUtensorMap *tma = reinterpret_cast<CUtensorMap *>(desc.data_ptr<uint8_t>());

    CUresult r = cuTensorMapEncodeTiled(
        tma,
        dtype,
        (cuuint32_t)R,
        (void *)base.data_ptr(),
        gdim.data(),
        gstride.data(),
        bdim.data(),
        estride.data(),
        interleave,
        swz,
        l2p,
        oob);

    TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed with error code ", r);

    return desc;
}

void py_cuTensorMapReplaceAddress(torch::Tensor desc, torch::Tensor data) {
    auto desc_ptr = reinterpret_cast<CUtensorMap*>(desc.data_ptr<uint8_t>());
    void* new_addr = (void *)data.data_ptr();
    CUresult r = cuTensorMapReplaceAddress(desc_ptr, new_addr);
    TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapReplaceAddress failed with error code ", r);
}


void py_gemv(torch::Tensor a_desc, torch::Tensor b_desc, torch::Tensor c_desc,
          int M, int N, int K, int num_copy_blocks, int smem_size, int sms_per_row
        //   , torch::Tensor d_start, torch::Tensor d_end
        )
{
    // Check tensors and extract pointers
    auto a_desc_ptr = reinterpret_cast<CUtensorMap*>(a_desc.data_ptr<uint8_t>());
    auto b_desc_ptr = reinterpret_cast<CUtensorMap*>(b_desc.data_ptr<uint8_t>());
    auto c_desc_ptr = reinterpret_cast<CUtensorMap*>(c_desc.data_ptr<uint8_t>());

    // auto d_start_ptr = reinterpret_cast<unsigned long long*>(d_start.data_ptr());
    // auto d_end_ptr = reinterpret_cast<unsigned long long*>(d_end.data_ptr());

    // Launch kernel
    dim3 grid(NUM_SMS);
    dim3 block(NUM_PRODUCER_THREADS + NUM_CONSUMER_THREADS);

    // Use PyTorch current stream so graph capture/replay can see this launch.
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    gemv<<<grid, block, smem_size, stream>>>(*a_desc_ptr, *b_desc_ptr, *c_desc_ptr, M, N, K, num_copy_blocks, sms_per_row);
}

void py_gemv_horizontal(torch::Tensor a_desc_h, torch::Tensor a_desc_d, torch::Tensor b_desc, torch::Tensor c_desc,
                        int h_M, int d_M, int N, int K, int num_copy_blocks, int num_copy_host_blocks, int smem_size, int h_sms_per_row, int d_sms_per_row
                        // ,torch::Tensor d_start, torch::Tensor d_end
          )
{
    // Check tensors and extract pointers
    auto a_desc_h_ptr = reinterpret_cast<CUtensorMap*>(a_desc_h.data_ptr<uint8_t>());
    auto a_desc_d_ptr = reinterpret_cast<CUtensorMap*>(a_desc_d.data_ptr<uint8_t>());
    auto b_desc_ptr = reinterpret_cast<CUtensorMap*>(b_desc.data_ptr<uint8_t>());
    auto c_desc_ptr = reinterpret_cast<CUtensorMap*>(c_desc.data_ptr<uint8_t>());

        // auto d_start_ptr = reinterpret_cast<unsigned long long*>(d_start.data_ptr());
        // auto d_end_ptr = reinterpret_cast<unsigned long long*>(d_end.data_ptr());
    

    // Launch kernel
    dim3 grid(NUM_SMS);
    dim3 block(NUM_PRODUCER_THREADS + NUM_CONSUMER_THREADS);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    gemv_horizontal<<<grid, block, smem_size, stream>>>(*a_desc_h_ptr, *a_desc_d_ptr, *b_desc_ptr, *c_desc_ptr, h_M, d_M, N, K, num_copy_blocks, num_copy_host_blocks, h_sms_per_row, d_sms_per_row);
}


void py_gemv_horizontal_multicast(
    torch::Tensor h_a_desc,
    torch::Tensor d_a_desc,
    torch::Tensor b_desc,
    torch::Tensor c_desc,
    int h_M,
    int d_M,
    int N,
    int K,
    int num_copy_blocks,
    int num_copy_host_blocks,
    int smem_size,
    int h_sms_per_row,
    int d_sms_per_row
) {
    // Check tensors and extract pointers
    auto h_a_ptr = reinterpret_cast<CUtensorMap*>(h_a_desc.data_ptr<uint8_t>());
    auto d_a_ptr = reinterpret_cast<CUtensorMap*>(d_a_desc.data_ptr<uint8_t>());
    auto b_ptr = reinterpret_cast<CUtensorMap*>(b_desc.data_ptr<uint8_t>());
    auto c_ptr = reinterpret_cast<CUtensorMap*>(c_desc.data_ptr<uint8_t>());


    // Launch exactly the requested copy blocks. This matches the standalone
    // debug benchmark and avoids creating extra CTAs for a clustered kernel.
    dim3 grid(num_copy_blocks);
    dim3 block(NUM_PRODUCER_THREADS * 2 + 2 * NUM_CONSUMER_THREADS);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    gemv_horizontal_multicast<TILE_M_MULTICAST, TILE_N_MULTICAST, TILE_K_MULTICAST, BUFFER_SLOTS_MULTICAST, BUFFER_SLOTS_B_MULTICAST, CHUNK_SIZE_A, CHUNK_SIZE_B, CHUNK_SIZE_C>
        <<<grid, block, smem_size, stream>>>(*h_a_ptr, *d_a_ptr, *b_ptr, *c_ptr, h_M, d_M, N, K, num_copy_blocks, num_copy_host_blocks, h_sms_per_row, d_sms_per_row);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void py_gemv_horizontal_multicast_prefill(
    torch::Tensor h_a_desc,
    torch::Tensor d_a_desc,
    torch::Tensor b_desc,
    torch::Tensor c_desc,
    int h_M,
    int d_M,
    int N,
    int K,
    int num_copy_blocks,
    int num_copy_host_blocks,
    int smem_size,
    int h_sms_per_row,
    int d_sms_per_row
) {
    // Check tensors and extract pointers
    auto h_a_ptr = reinterpret_cast<CUtensorMap*>(h_a_desc.data_ptr<uint8_t>());
    auto d_a_ptr = reinterpret_cast<CUtensorMap*>(d_a_desc.data_ptr<uint8_t>());
    auto b_ptr = reinterpret_cast<CUtensorMap*>(b_desc.data_ptr<uint8_t>());
    auto c_ptr = reinterpret_cast<CUtensorMap*>(c_desc.data_ptr<uint8_t>());


    // Launch exactly the requested copy blocks. This matches the standalone
    // debug benchmark and avoids creating extra CTAs for a clustered kernel.
    dim3 grid(num_copy_blocks);
    dim3 block(NUM_PRODUCER_THREADS * 2 + 2 * NUM_CONSUMER_THREADS);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    gemv_horizontal_multicast_prefill<TILE_M_MULTICAST, TILE_N_MULTICAST, TILE_K_MULTICAST, BUFFER_SLOTS_MULTICAST, BUFFER_SLOTS_B_MULTICAST, CHUNK_SIZE_A, CHUNK_SIZE_B, CHUNK_SIZE_C>
        <<<grid, block, smem_size, stream>>>(*h_a_ptr, *d_a_ptr, *b_ptr, *c_ptr, h_M, d_M, N, K, num_copy_blocks, num_copy_host_blocks, h_sms_per_row, d_sms_per_row);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void py_gemv_multicast(
    torch::Tensor a_desc,
    torch::Tensor b_desc,
    torch::Tensor c_desc,
    int M,
    int N,
    int K,
    int num_copy_blocks,
    int smem_size,
    int sms_per_row
) {

    auto a_desc_ptr = reinterpret_cast<CUtensorMap*>(a_desc.data_ptr<uint8_t>());
    auto b_desc_ptr = reinterpret_cast<CUtensorMap*>(b_desc.data_ptr<uint8_t>());
    auto c_desc_ptr = reinterpret_cast<CUtensorMap*>(c_desc.data_ptr<uint8_t>());

    // Launch kernel
    dim3 grid(NUM_SMS);
    dim3 block(NUM_PRODUCER_THREADS * 2 + 2 * NUM_CONSUMER_THREADS);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    gemv_multicast<TILE_M_MULTICAST, TILE_N_MULTICAST, TILE_K_MULTICAST, BUFFER_SLOTS_MULTICAST, BUFFER_SLOTS_B_MULTICAST, CHUNK_SIZE_A, CHUNK_SIZE_B, CHUNK_SIZE_C>
        <<<grid, block, smem_size, stream>>>(*a_desc_ptr, *b_desc_ptr, *c_desc_ptr, M, N, K, num_copy_blocks, sms_per_row);
}

void py_gemv_multicast_prefill(
    torch::Tensor a_desc,
    torch::Tensor b_desc,
    torch::Tensor c_desc,
    int M,
    int N,
    int K,
    int num_copy_blocks,
    int smem_size,
    int sms_per_row
) {

    auto a_desc_ptr = reinterpret_cast<CUtensorMap*>(a_desc.data_ptr<uint8_t>());
    auto b_desc_ptr = reinterpret_cast<CUtensorMap*>(b_desc.data_ptr<uint8_t>());
    auto c_desc_ptr = reinterpret_cast<CUtensorMap*>(c_desc.data_ptr<uint8_t>());

    // Launch kernel
    dim3 grid(NUM_SMS);
    dim3 block(NUM_PRODUCER_THREADS * 2 + 2 * NUM_CONSUMER_THREADS);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    gemv_multicast_prefill<TILE_M_MULTICAST, TILE_N_MULTICAST, TILE_K_MULTICAST, BUFFER_SLOTS_MULTICAST, BUFFER_SLOTS_B_MULTICAST, CHUNK_SIZE_A, CHUNK_SIZE_B, CHUNK_SIZE_C>
        <<<grid, block, smem_size, stream>>>(*a_desc_ptr, *b_desc_ptr, *c_desc_ptr, M, N, K, num_copy_blocks, sms_per_row);
}

void py_flush_l2_cache(torch::Tensor dummy, torch::Tensor sink) {
    dim3 grid(NUM_SMS);
    dim3 block(NUM_PRODUCER_THREADS + NUM_CONSUMER_THREADS);
    auto dummy_ptr = reinterpret_cast<float*>(dummy.data_ptr<float>());
    auto sink_ptr = reinterpret_cast<float*>(sink.data_ptr<float>());
    size_t size = dummy.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    flush_l2_cache<<<grid, block, 0, stream>>>(dummy_ptr, size, sink_ptr);
}

size_t py_set_smem_size(size_t smem_size, int opcode=GEMV) {
    void* gemv_ptr;
    if (opcode == GEMV) {
        gemv_ptr = reinterpret_cast<void*>(gemv);
    } else if (opcode == GEMV_HORIZONTAL) {
        gemv_ptr = reinterpret_cast<void*>(gemv_horizontal);
    } else if (opcode == GEMV_MULTICAST) {
        gemv_ptr = reinterpret_cast<void*>(
            gemv_multicast<TILE_M_MULTICAST, TILE_N_MULTICAST, TILE_K_MULTICAST, BUFFER_SLOTS_MULTICAST, BUFFER_SLOTS_B_MULTICAST,
                    CHUNK_SIZE_A, CHUNK_SIZE_B, CHUNK_SIZE_C>);
    } else if (opcode == GEMV_HORIZONTAL_MULTICAST) {
        gemv_ptr = reinterpret_cast<void*>(
            gemv_horizontal_multicast<TILE_M_MULTICAST, TILE_N_MULTICAST, TILE_K_MULTICAST, BUFFER_SLOTS_MULTICAST, BUFFER_SLOTS_B_MULTICAST,
                    CHUNK_SIZE_A, CHUNK_SIZE_B, CHUNK_SIZE_C>);
    } else if (opcode == GEMV_MULTICAST_PREFILL) {
        gemv_ptr = reinterpret_cast<void*>(
            gemv_multicast_prefill<TILE_M_MULTICAST, TILE_N_MULTICAST, TILE_K_MULTICAST, BUFFER_SLOTS_MULTICAST, BUFFER_SLOTS_B_MULTICAST,
                    CHUNK_SIZE_A, CHUNK_SIZE_B, CHUNK_SIZE_C>);
    } else if (opcode == GEMV_HORIZONTAL_MULTICAST_PREFILL) {
        gemv_ptr = reinterpret_cast<void*>(
            gemv_horizontal_multicast_prefill<TILE_M_MULTICAST, TILE_N_MULTICAST, TILE_K_MULTICAST, BUFFER_SLOTS_MULTICAST, BUFFER_SLOTS_B_MULTICAST,
                    CHUNK_SIZE_A, CHUNK_SIZE_B, CHUNK_SIZE_C>);
    } else {
        std::cerr << "Unsupported opcode: " << opcode << std::endl;
        return 0;
    }

    cudaError_t err = cudaFuncSetAttribute(
        gemv_ptr,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    );

    if (err != cudaSuccess) {
        std::cerr << "Kernel set shared memory failed: " << cudaGetErrorString(err) << std::endl;
    }

    if (opcode==GEMV_MULTICAST || opcode==GEMV_HORIZONTAL_MULTICAST) {
        err = cudaFuncSetAttribute(gemv_ptr, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
        if (err != cudaSuccess) {
            std::cerr << "Kernel set cluster size failed: " << cudaGetErrorString(err) << std::endl;
        }
    }
    return smem_size;
}

void py_print_device_name(int device_id)
{
    print_device_name(device_id);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    //   auto op = m.def_submodule("opcode", "DAE2 OpCodes");
    //   #define DAE_OP(name, value) op.attr(#name) = (int)name;
    //   #include "dae/opcode.cuh.inc"
    //   #undef DAE_OP

    //   auto config = m.def_submodule("config", "DAE2 Configuration Constants");
    //   config.attr("slot_size") = slotSizeKb * 1024;
    //   config.attr("num_slots") = numSlots;
    //   config.attr("max_insts") = numInsts;
    //   config.attr("num_profile_events") = numProfileEvents;
    //   config.attr("max_tmas") = numTmas;
    //   config.attr("max_bars") = numBars;

    m.def("set_smem_size", &py_set_smem_size,
          "Set dynamic shared memory size for DAE2 kernel");
    m.def("build_tma_desc", &py_build_tma_desc,
          "Build CUtensorMap descriptor for given tensor and layout");
    m.def("tensormap_replace_address", &py_cuTensorMapReplaceAddress,
          "Replace the address of a CUtensorMap descriptor");
    m.def("gemv", &py_gemv,
          "Launch GEMV kernel with given TMA descriptors and parameters");
    m.def("gemv_multicast", &py_gemv_multicast,
          "Launch GEMV_MULTICAST kernel with given TMA descriptors and parameters");
    m.def("gemv_horizontal_multicast", &py_gemv_horizontal_multicast,
          "Launch GEMV_HORIZONTAL_MULTICAST kernel with given TMA descriptors and parameters");
    m.def("gemv_horizontal", &py_gemv_horizontal,
          "Launch GEMV kernel with given TMA descriptors and parameters");
    m.def("gemv_multicast_prefill", &py_gemv_multicast_prefill,
          "Launch GEMV_MULTICAST_PREFILL kernel with given TMA descriptors and parameters");
    m.def("gemv_horizontal_multicast_prefill", &py_gemv_horizontal_multicast_prefill,
          "Launch GEMV_HORIZONTAL_MULTICAST_PREFILL kernel with given TMA descriptors and parameters");
    m.def("flush_l2_cache", &py_flush_l2_cache,
          "Flush L2 cache by launching a dummy kernel that touches a large buffer");

    m.def("print_device_name", &py_print_device_name,
          "Print the name of the CUDA device with the given ID");
}
