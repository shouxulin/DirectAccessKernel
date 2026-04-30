#include <cuda_runtime.h>
#include <cuda/barrier>
#include <iostream>
#include <thread>
#include <chrono>
#include <string>
#include <fstream>
#include <cstring>
#include <iomanip>
#include <sys/mman.h>
#include <cassert>
#include <cudaTypedefs.h>
#include <cuda.h>
#include <assert.h>
#include <random>
#include <cstdint>
#include <cutlass/half.h>
#include <cute/tensor.hpp>
#include <cute/arch/mma_sm90.hpp>      // SM80_16x8x16_F16F16F16F16_TN
#include <cute/atom/mma_atom.hpp>      // MMA_Atom / make_tiled_mma
#include <cute/algorithm/gemm.hpp>     // cute::gemm
#include <cuda.h>
#include <curand_kernel.h>
#include "config.cuh"


using half_t = cutlass::half_t;
namespace cde = cuda::device::experimental;


__device__ uint get_smid(void) {
     uint ret;
     asm("mov.u32 %0, %smid;" : "=r"(ret) );
     return ret;
}

__device__ __forceinline__ unsigned long long get_clock(bool global_timer=true) {
    if (global_timer) {
        unsigned long long t;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); // read the 64-bit global nanosecond timer
        return t;
    } else {
        return clock64();
    }
}

template<int BarrierID, int Count>
__device__ __forceinline__ void __sync_barrier() {
    asm volatile(
        "bar.sync %0, %1;"
        :
        : "n"(BarrierID), "n"(Count)
        : "memory"
    );
}

__device__ __forceinline__ void cp_reduce_async_bulk_tensor_2d_shared_to_gloabl (
    const void* tensorMap, int c0, int c1, void* src
){
    uint32_t src_shared = static_cast<uint32_t>(__cvta_generic_to_shared(src));
    asm volatile(
        "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group"
        "[%0, {%1, %2}], [%3];\n"
        :
        : "l"(tensorMap), "r"(c0), "r"(c1), "r"(src_shared)
        : "memory"
    );
}


__device__ __forceinline__ void task_wgmma_m64n256k16(half_t *a_buffer, half_t *b_buffer, half_t *sC,
                                                      const int TILE_ELEMS_A, const int TILE_ELEMS_B, const int NUM_TILES_K,
                                                      int tile_ik_base, int tile_ik_stride, 
                                                      int &nbuf, 
                                                      cuda::barrier<cuda::thread_scope_block> filled[], cuda::barrier<cuda::thread_scope_block> drained[]
// #if RECORD_NUM_CHUNKS > 0
//                                                       ,unsigned long long chunk_times_end[], int &num_chunks_recorded
// #endif
                                                     ){
    using namespace cute;

    constexpr int MMA_M = 64, MMA_N = 8, MMA_K = 16;
    // constexpr int numThreads = 128;

    // Make a bigger "kernel" by repeating the atom
    auto tiled_mma = make_tiled_mma(
        MMA_Atom<
            // SM90_64x8x16_F16F16F16_SS<GMMA::Major::K, GMMA::Major::K>
            SM90_64x8x16_F16F16F16_SS<GMMA::Major::MN, GMMA::Major::K>
        >{},
        make_layout(make_shape(Int<1>{}, Int<1>{}, Int<1>{})), // number of thread-parallel atoms
        make_tile(Int<TILE_M/MMA_M>{}, Int<TILE_N/MMA_N>{}, Int<TILE_K/MMA_K>{}) // number of wgmma instructions inside one warp group, MAY increase the useage of registers.
    );
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);

    // this layout should match the TMA load layout
    // auto layout_sA = tile_to_shape(
    //     GMMA::Layout_K_SW128_Atom<half_t>{},
    //     make_shape(Int<TILE_M>{},Int<TILE_K>{}));
    auto layout_sA = tile_to_shape(
        GMMA::Layout_MN_SW128_Atom<half_t>{},
        make_shape(Int<TILE_M>{},Int<TILE_K>{}));

    auto layout_sB = tile_to_shape(
        GMMA::Layout_K_SW128_Atom<half_t>{},
        make_shape(Int<TILE_N>{}, Int<TILE_K>{}));

    // C must be laid out as MxN to match the GMMA accumulator fragments
    auto layout_sC = tile_to_shape(
        GMMA::Layout_MN_SW128_Atom<half_t>{},
        make_shape(Int<TILE_M>{}, Int<TILE_N>{}));

    Tensor t_sC = make_tensor(
        make_smem_ptr((half_t*)sC),
        layout_sC
    );

    auto frag_C = thr_mma.partition_fragment_C(t_sC);
    clear(frag_C);

    for (int tile_ik = tile_ik_base; tile_ik < NUM_TILES_K; tile_ik+=tile_ik_stride) {
        int next_slot = nbuf % BUFFER_SLOTS;
        auto token = cuda::device::barrier_arrive_tx(filled[next_slot], 1, 0);
        filled[next_slot].wait(cuda::std::move(token));

// #if RECORD_NUM_CHUNKS > 0
//         if (threadIdx.x==0 && num_chunks_recorded < RECORD_NUM_CHUNKS) {
//             chunk_times_end[num_chunks_recorded++] = get_clock(true);
//         }
// #endif

        half_t *sA = (half_t*)a_buffer + next_slot * TILE_ELEMS_A;
        half_t *sB = (half_t*)b_buffer + next_slot * TILE_ELEMS_B;

        auto t_sA = make_tensor(make_smem_ptr(sA), layout_sA);
        auto t_sB = make_tensor(make_smem_ptr(sB), layout_sB);

        auto frag_A = thr_mma.partition_fragment_A(t_sA);
        auto frag_B = thr_mma.partition_fragment_B(t_sB);


        warpgroup_arrive();
        gemm(tiled_mma, frag_A, frag_B, frag_C);   // emit multiple wgmma instructions by cute, see make_tiled_mma        
        warpgroup_commit_batch();
        warpgroup_wait<0>();

        (void) drained[next_slot].arrive();
        nbuf++;
    }

    copy(frag_C, thr_mma.partition_C(t_sC));
    cuda::ptx::fence_proxy_async();
}



// __device__ __forceinline__ void task_wgmma_m64n256k16(half_t *a_buffer, half_t *b_buffer, half_t *sC,
//                                                       const int TILE_ELEMS_A, const int TILE_ELEMS_B, const int NUM_TILES_K,
//                                                       int tile_ik_base, int tile_ik_stride, /* used to reduce: multiple sms per row*/
//                                                       int &nbuf, 
//                                                       cuda::barrier<cuda::thread_scope_block> filled[], cuda::barrier<cuda::thread_scope_block> drained[]
// #if RECORD_NUM_CHUNKS > 0
//                                                       ,unsigned long long chunk_times_end[], int &num_chunks_recorded
// #endif
//                                                      ){
//     using namespace cute;

//     // static_assert(TILE_N == 8, "Only support N=8 for now");

//     using Atom = SM80_16x8x16_F32F16F16F32_TN;
//     using AtomTrait = MMA_Traits<Atom>;
//     using data_t = typename AtomTrait::ValTypeA;
//     // using data_t = half_t;
//     using accum_t = typename AtomTrait::ValTypeC;

//     constexpr int MMA_M = shape<0>(typename AtomTrait::Shape_MNK{});
//     constexpr int MMA_N = shape<1>(typename AtomTrait::Shape_MNK{});
//     constexpr int MMA_K = shape<2>(typename AtomTrait::Shape_MNK{});
//     constexpr int numThreads = 32 * (TILE_M / MMA_M);

//     static_assert(TILE_M % MMA_M == 0, "TILE_M must be multiple of MMA_M");
//     static_assert(TILE_K % MMA_K == 0, "TILE_K must be multiple of MMA_K");
//     static_assert(numThreads == 128, "Only support a 128-thread compute group for now");

//     auto tiled_mma = make_tiled_mma(
//         MMA_Atom<Atom>{},
//         make_layout(make_shape(Int<TILE_M / MMA_M>{}, Int<1>{}, Int<1>{})), // atom replication
//         make_tile(Int<TILE_M>{}, Int<MMA_N>{}, Int<TILE_K>{}) // final target MNK
//     );

//     int tid = threadIdx.x;
//     auto thr_mma  = tiled_mma.get_slice(tid);


//     auto layout_sA = tile_to_shape(
//         GMMA::Layout_MN_SW128_Atom<data_t>{},
//         make_shape(Int<TILE_M>{}, Int<TILE_K>{}));
//     auto layout_sB = tile_to_shape(
//         GMMA::Layout_K_SW128_Atom<data_t>{},
//         make_shape(Int<MMA_N>{}, Int<TILE_K>{}));
//     auto layout_sC = tile_to_shape(
//         GMMA::Layout_MN_SW128_Atom<data_t>{},
//         make_shape(Int<TILE_M>{}, Int<MMA_N>{}));


//     auto t_dummyA = make_tensor(make_smem_ptr(static_cast<data_t*>(nullptr)), layout_sA);
//     auto t_dummyB = make_tensor(make_smem_ptr(static_cast<data_t*>(nullptr)), layout_sB);
//     auto t_dummyC = make_tensor(make_smem_ptr(static_cast<accum_t*>(nullptr)),
//         Layout<Shape<Int<TILE_M>, Int<MMA_N>>, Stride<Int<1>, Int<TILE_M>>>());

//     auto frag_A = thr_mma.partition_fragment_A(t_dummyA);
//     auto frag_B = thr_mma.partition_fragment_B(t_dummyB);
//     auto frag_C = thr_mma.partition_fragment_C(t_dummyC);

//     clear(frag_C);


//     for (int tile_ik = tile_ik_base; tile_ik < NUM_TILES_K ; tile_ik+= tile_ik_stride) {
//         int next_slot = nbuf % BUFFER_SLOTS;
//         auto token = cuda::device::barrier_arrive_tx(filled[next_slot], 1, 0);
//         filled[next_slot].wait(cuda::std::move(token));
// // #if RECORD_NUM_CHUNKS > 0
// //         if (threadIdx.x==0 && num_chunks_recorded < RECORD_NUM_CHUNKS) {
// //             chunk_times_end[num_chunks_recorded++] = get_clock(true);
// //         }
// // #endif
        
//         data_t *sA = (data_t*)a_buffer + next_slot * TILE_ELEMS_A;
//         data_t *sB = (data_t*)b_buffer + next_slot * TILE_ELEMS_B;

//         auto t_sA = make_tensor(make_smem_ptr(sA), layout_sA);
//         copy(thr_mma.partition_A(t_sA), frag_A);

//         auto t_sB = make_tensor(make_smem_ptr(sB), layout_sB);
//         copy(thr_mma.partition_B(t_sB), frag_B);

//         // warpgroup_arrive();
//         gemm(tiled_mma, frag_C, frag_A, frag_B, frag_C);       
//         // warpgroup_commit_batch();
//         // warpgroup_wait<0>();
//         __sync_barrier<8, numThreads>();

//         (void) drained[next_slot].arrive();
//         nbuf++;
//     }

//     auto t_sC = make_tensor(make_smem_ptr(sC), layout_sC);
//     copy(frag_C, thr_mma.partition_C(t_sC));
//     cuda::ptx::fence_proxy_async();
// }

__device__ __forceinline__ void loop (const CUtensorMap *a_tensor_map, const CUtensorMap *b_tensor_map, const CUtensorMap *c_tensor_map, 
                                      int M, int N, int K,
                                      int M_offset, /* used in horizontal split to ensure store the result to the right row in C*/
                                      int blockId, int num_copy_blocks,
                                      int tile_ik_base, int tile_ik_stride /* used to reduce: multiple sms per row*/
#if RECORD_NUM_CHUNKS > 0
                                      ,unsigned long long chunk_times_start[], unsigned long long chunk_times_end[]
#endif
                                    )
{
    const int TILE_ELEMS_A = TILE_M * TILE_K;
    const int TILE_ELEMS_B = TILE_N * TILE_K;

    const int TILE_BYTES_A = TILE_M * TILE_K * sizeof(half_t);
    const int TILE_BYTES_B = TILE_N * TILE_K * sizeof(half_t);

    const int NUM_TILES_M = (M + TILE_M - 1) / TILE_M; // number of tiles in M direction
    const int NUM_TILES_N = (N + TILE_N - 1) / TILE_N; // number of tiles in N direction
    const int NUM_TILES_K = (K + TILE_K - 1) / TILE_K; // number of tiles in K direction

    extern __shared__ __align__(128) half_t shared_mem[];

    half_t* a_buffer = shared_mem;
    half_t* b_buffer = a_buffer + TILE_ELEMS_A * BUFFER_SLOTS;
    half_t* c_buffer = b_buffer + TILE_ELEMS_B * BUFFER_SLOTS;

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ cuda::barrier<cuda::thread_scope_block> filled[BUFFER_SLOTS], drained[BUFFER_SLOTS];

#if RECORD_NUM_CHUNKS > 0
    int num_chunks_recorded = 0;
#endif

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < BUFFER_SLOTS; i++) {
            init(&filled[i], NUM_CONSUMER_THREADS + 1); // +1 for producer
            init(&drained[i], NUM_CONSUMER_THREADS + 1); // +1 for producer
        }
    }
    __syncthreads();

    if (threadIdx.x == NUM_CONSUMER_THREADS) { // producer
        int nbuf = 0;
        // for (int tile_im = sm_id / NUM_SMS_PER_ROW; tile_im < NUM_TILES_M; tile_im += num_copy_blocks) { // partition N dimention by SM
        for (int tile_im = blockId; tile_im < NUM_TILES_M; tile_im += num_copy_blocks) { // partition N dimention by SM
            // slide over d_M dimension
            for (int tile_in = 0; tile_in < NUM_TILES_N; tile_in++) {
                // slide over K dimension
                for (int tile_ik = tile_ik_base; tile_ik < NUM_TILES_K ; tile_ik+= tile_ik_stride) {
                    int next_slot = nbuf % BUFFER_SLOTS;
                    // if (blockId == 0) printf("[Producer] waiting to drain slot %d for tile_im=%d, tile_ik=%d\n", next_slot, tile_im, tile_ik);
                    drained[next_slot].arrive_and_wait();

#if RECORD_NUM_CHUNKS > 0
                    if (num_chunks_recorded < RECORD_NUM_CHUNKS) {
                        chunk_times_start[num_chunks_recorded++] = get_clock(true);
                    }
#endif
                    half_t *dst_ptr;
                    // copy a tile from A
                    dst_ptr = a_buffer + next_slot * TILE_ELEMS_A;
                    // cde::cp_async_bulk_tensor_3d_global_to_shared(dst_ptr, a_tensor_map, 0, tile_im * TILE_M, tile_ik * (TILE_K / 64), filled[next_slot]); // K-major
                    cde::cp_async_bulk_tensor_2d_global_to_shared(dst_ptr, a_tensor_map, tile_im * TILE_M, tile_ik * TILE_K, filled[next_slot]); // MN-major

                    // copy a tile from B
                    dst_ptr = b_buffer + next_slot * TILE_ELEMS_B;
                    cde::cp_async_bulk_tensor_3d_global_to_shared(dst_ptr, b_tensor_map, 0, tile_in * TILE_N, tile_ik * (TILE_K/64), filled[next_slot]);
                    (void) cuda::device::barrier_arrive_tx(filled[next_slot], 1, TILE_BYTES_A + TILE_BYTES_B);

                    nbuf++;
                }
            }
        }
    } else if (threadIdx.x < NUM_CONSUMER_THREADS){ // consumer
        #pragma unroll
        for (int i = 0; i < BUFFER_SLOTS; i++) {
            (void) drained[i].arrive();
        }

        int nbuf = 0;
        for (int tile_im = blockId; tile_im < NUM_TILES_M; tile_im += num_copy_blocks) { // partition N dimention by SM
            for (int tile_in = 0; tile_in < NUM_TILES_N; tile_in++) {
                task_wgmma_m64n256k16(a_buffer, b_buffer, c_buffer, TILE_ELEMS_A, TILE_ELEMS_B, NUM_TILES_K, tile_ik_base, tile_ik_stride, nbuf, filled, drained
    #if RECORD_NUM_CHUNKS > 0
                    ,chunk_times_end, num_chunks_recorded
    #endif
                    );
                // copy back C from shared memory to gloab mem using TMA
                if (threadIdx.x == 0) {
                    if (tile_ik_stride == 1) {
                        cde::cp_async_bulk_tensor_2d_shared_to_global(c_tensor_map, tile_im * TILE_M + M_offset, tile_in * TILE_N, c_buffer);
                    } else {
                        cp_reduce_async_bulk_tensor_2d_shared_to_gloabl (c_tensor_map, tile_im * TILE_M + M_offset, tile_in * TILE_N, c_buffer);
                    }
                    cde::cp_async_bulk_commit_group();
                    cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{}); // TODO: check if this will lose performance
                }
            }
        }
    }
}



__global__ void gemv(const __grid_constant__ CUtensorMap a_tensor_map, const __grid_constant__ CUtensorMap b_tensor_map, const __grid_constant__ CUtensorMap c_tensor_map, 
                    int M, int N, int K, 
                    int num_copy_blocks,
                    int sms_per_row
                    // , unsigned long long* d_start, unsigned long long* d_end
// #if RECORD_NUM_CHUNKS > 0
//                     unsigned long long* d_chunk_start, unsigned long long* d_chunk_end
// #endif
                    ) { 
    int sm_id = get_smid();
    if (sm_id >= num_copy_blocks){
        return;
    }

    // unsigned long long start, end;
    // if (threadIdx.x == 0) start = get_clock(true);

        
    int row_id = sm_id / sms_per_row;
    int block_size = num_copy_blocks / sms_per_row;
    int tile_ik_base = sm_id % sms_per_row;

    loop(&a_tensor_map, &b_tensor_map, &c_tensor_map, M, N, K, 0, row_id, block_size, tile_ik_base, sms_per_row
#if RECORD_NUM_CHUNKS > 0
                                               , chunk_times_start, chunk_times_end
#endif
    );

    // if (threadIdx.x == 0) { // consumer
    //     end = get_clock(true); 
    //     d_start[sm_id] = start;
    //     d_end[sm_id] = end;
    // }
}


__global__ void gemv_horizontal(const __grid_constant__ CUtensorMap h_a_tensor_map, const __grid_constant__ CUtensorMap d_a_tensor_map, const __grid_constant__ CUtensorMap b_tensor_map, const __grid_constant__ CUtensorMap c_tensor_map, 
                          int h_M, int d_M, int N, int K, 
                          int num_copy_blocks, int num_copy_host_blocks,
                          int h_sms_per_row, int d_sms_per_row
                        //   unsigned long long* d_start, unsigned long long* d_end
) {
    int sm_id = get_smid();
    if (sm_id >= num_copy_blocks){
        return;
    }

    // unsigned long long start, end;
    // if (threadIdx.x == 0) start = get_clock(true);


    if (sm_id < num_copy_host_blocks) {
        int row_id = sm_id / h_sms_per_row;
        int block_size = num_copy_host_blocks / h_sms_per_row;
        int tile_ik_base = sm_id % h_sms_per_row;

        loop(&h_a_tensor_map, &b_tensor_map, &c_tensor_map, h_M, N, K, 0, row_id, block_size, tile_ik_base, h_sms_per_row
        #if RECORD_NUM_CHUNKS > 0
            , chunk_times_start, chunk_times_end
        #endif
        );
    } else {
        int row_id = (sm_id - num_copy_host_blocks) / d_sms_per_row;
        int block_size = (num_copy_blocks - num_copy_host_blocks) / d_sms_per_row;
        int tile_ik_base = (sm_id - num_copy_host_blocks) % d_sms_per_row;

        loop(&d_a_tensor_map, &b_tensor_map, &c_tensor_map, d_M, N, K, h_M, row_id, block_size, tile_ik_base, d_sms_per_row
        #if RECORD_NUM_CHUNKS > 0
            , chunk_times_start, chunk_times_end
        #endif
        );
    }


    // if (threadIdx.x == 0) { // consumer
    //     end = get_clock(true); 
    //     d_start[sm_id] = start;
    //     d_end[sm_id] = end;
    // }                 
                          
}
