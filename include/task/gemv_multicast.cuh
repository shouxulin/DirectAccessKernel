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
#include <cutlass/arch/barrier.h>
#include <cute/tensor.hpp>
#include <cute/arch/mma_sm90.hpp>      // SM80_16x8x16_F16F16F16F16_TN
#include <cute/atom/mma_atom.hpp>      // MMA_Atom / make_tiled_mma
#include <cute/algorithm/gemm.hpp>     // cute::gemm
#include <cuda.h>
#include <curand_kernel.h>
#include "config.cuh"


using half_t = cutlass::half_t;
namespace cde = cuda::device::experimental;

using ClusterBarrier = cutlass::arch::ClusterBarrier;


__device__ __forceinline__ void cp_async_bulk_global_to_shared_multicast (
    void* dst, const void* src, int32_t size, void* mbar, uint16_t multicast_mask
){
    uint32_t dst_shared = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    uint64_t src_global = static_cast<uint64_t>(__cvta_generic_to_global(src));
    uint32_t mbar_shared = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster "
        "[%0], [%1], %2, [%3], %4;\n"
        :
        : "r"(dst_shared), "l"(src_global), "r"(size), "r"(mbar_shared), "h"(multicast_mask)
        : "memory"
    );
}

__device__ __forceinline__ void cp_async_bulk_global_to_shared (
    void* dst, const void* src, int32_t size, void* mbar
){
    uint32_t dst_shared = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    uint64_t src_global = static_cast<uint64_t>(__cvta_generic_to_global(src));
    uint32_t mbar_shared = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1], %2, [%3];\n"
        :
        : "r"(dst_shared), "l"(src_global), "r"(size), "r"(mbar_shared)
        : "memory"
    );
}


__device__ __forceinline__ void cluster_barrier_arrive_remote(
    ClusterBarrier::ValueType* barrier,
    uint32_t dst_cta_rank
) {
    ClusterBarrier::arrive(barrier, dst_cta_rank, 1);
}

__device__ __forceinline__ void cp_async_bulk_tensor_4d_global_to_shared_multicast (
    void* dst, const void* tensorMap, int d0, int d1, int d2, int d3, void* mbar, uint16_t multicast_mask
){
    uint32_t dst_shared = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    uint32_t mbar_shared = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster "
        "[%0], [%1, {%2, %3, %4, %5}], [%6], %7;\n"
        :
        : "r"(dst_shared), "l"(tensorMap), "r"(d0), "r"(d1), "r"(d2), "r"(d3), "r"(mbar_shared), "h"(multicast_mask)
        : "memory"
    );
}


template<int TILE_M_T, int TILE_N_T, int TILE_K_T, int BUFFER_SLOTS_T, int BUFFER_SLOTS_B_T, int TILE_ELEMS_A_T, int TILE_ELEMS_B_T, int GROUP_ID>
__device__ __forceinline__ void task_wgmma_local(
    half_t * __restrict__ a_buffer, half_t * __restrict__ b_buffer, half_t * __restrict__ sC,
    int thread_id, int &nbuf, int tile_ik_base, int tile_ik_stride,
    const int NUM_TILES_K, 
    cuda::barrier<cuda::thread_scope_block> filled[], cuda::barrier<cuda::thread_scope_block> drained[],
    cuda::barrier<cuda::thread_scope_block> filled_b[], cuda::barrier<cuda::thread_scope_block> drained_b[],
    ClusterBarrier::ValueType cluster_drained[]
) {
    using namespace cute;
    // static_assert(TILE_N_T == 128 || TILE_N_T % 256 == 0, "TILE_N_T must be 128 or a multiple of 256");
    // static_assert(TILE_M_T % 64 == 0, "TILE_M_T must be a multiple of 64");
    // static_assert(TILE_K_T % 16 == 0, "TILE_K_T must be a multiple of 16");
    using GmmaAtom = std::conditional_t<
        TILE_N_T == 128,
        SM90_64x128x16_F16F16F16_SS<GMMA::Major::MN, GMMA::Major::K>,
        SM90_64x256x16_F16F16F16_SS<GMMA::Major::MN, GMMA::Major::K>>;

    // Make a bigger "kernel" by repeating the atom
    auto tiled_mma = make_tiled_mma(
        MMA_Atom<GmmaAtom>{},
        make_layout(make_shape(Int<1>{}, Int<1>{}, Int<1>{})), // number of thread-parallel atoms
        make_tile(Int<TILE_M_T>{}, Int<TILE_N_T>{}, Int<TILE_K_T>{}) // number of wgmma instructions inside one warp group, MAY increase the useage of registers.
    );
    auto thr_mma = tiled_mma.get_slice(thread_id);
    
    auto layout_sA = tile_to_shape(
        GMMA::Layout_MN_SW128_Atom<half_t>{},
        make_shape(Int<TILE_M_T>{},Int<TILE_K_T>{}));


    auto layout_sB = tile_to_shape(
        GMMA::Layout_K_SW128_Atom<half_t>{},
        make_shape(Int<TILE_N_T>{}, Int<TILE_K_T>{}));

    // C must be laid out as MxN to match the GMMA accumulator fragments
    auto layout_sC = tile_to_shape(
        GMMA::Layout_MN_SW128_Atom<half_t>{},
        make_shape(Int<TILE_M_T>{}, Int<TILE_N_T>{}));

    Tensor t_sC = make_tensor(
        make_smem_ptr((half_t*)sC),
        layout_sC
    );

    auto frag_C = thr_mma.partition_fragment_C(t_sC);
    clear(frag_C);

    using CopyAtomC = Copy_Atom<SM90_U16x8_STSM_T, half_t>;
    auto smem_tiled_copy_C = make_tiled_copy_C(CopyAtomC{}, tiled_mma);
    auto smem_thr_copy_C = smem_tiled_copy_C.get_thread_slice(thread_id);

    for (int tile_ik = tile_ik_base; tile_ik < NUM_TILES_K ; tile_ik += tile_ik_stride) {
        int next_slot = nbuf % BUFFER_SLOTS_T;
        int next_slot_b = nbuf % BUFFER_SLOTS_B_T;
        if constexpr (GROUP_ID == 0) {
            filled[next_slot].arrive_and_wait();
            filled_b[next_slot_b].arrive_and_wait();
        } else {
            filled[next_slot].arrive_and_wait();
            filled_b[next_slot_b + BUFFER_SLOTS_B_T].arrive_and_wait();
        }

        half_t *sA = (half_t*)a_buffer + next_slot * TILE_ELEMS_A_T;
        half_t *sB = (half_t*)b_buffer + next_slot_b * TILE_ELEMS_B_T;

        auto t_sA = make_tensor(make_smem_ptr(sA), layout_sA);
        auto t_sB = make_tensor(make_smem_ptr(sB), layout_sB);

        auto frag_A = thr_mma.partition_fragment_A(t_sA);
        auto frag_B = thr_mma.partition_fragment_B(t_sB);

        warpgroup_arrive();
        gemm(tiled_mma, frag_A, frag_B, frag_C);   // emit multiple wgmma instructions by cute, see make_tiled_mma        
        warpgroup_commit_batch();

        if (GROUP_ID == 0 && thread_id == 0) {
            (void) filled_b[next_slot_b + BUFFER_SLOTS_B_T].arrive();
        }

        warpgroup_wait<0>();
        if (thread_id == 0) {
            cluster_barrier_arrive_remote(&cluster_drained[next_slot], 0);
        }
        (void) drained[next_slot].arrive();
        (void) drained_b[next_slot_b].arrive();
        nbuf++;
    }

    auto tCsC = smem_thr_copy_C.partition_D(t_sC);
    auto tCrC = smem_thr_copy_C.retile_S(frag_C);
    copy(smem_tiled_copy_C, tCrC, tCsC);
    cuda::ptx::fence_proxy_async();
}




template<int TILE_M_T, int TILE_N_T,int TILE_K_T, int BUFFER_SLOTS_T, int BUFFER_SLOTS_B_T, int CHUNK_SIZE_A_T, int CHUNK_SIZE_B_T, int CHUNK_SIZE_C_T>
__global__  __cluster_dims__(CLUSTER_SIZE, 1, 1) void gemv_multicast(
    const __grid_constant__ CUtensorMap a_tensor_map,
    const __grid_constant__ CUtensorMap b_tensor_map,
    const __grid_constant__ CUtensorMap c_tensor_map,
    int M,
    int N,
    int K,
    int num_blocks,
    int sms_per_row
) {
    int sm_id = blockIdx.x;
    if (sm_id >= num_blocks){
        return;
    }
    extern __shared__ __align__(128) half_t shared_mem[];


    half_t* buffer_a = shared_mem;
    half_t* buffer_a_2 = buffer_a + BUFFER_SLOTS_T * CHUNK_SIZE_A_T; // double buffer for A
    half_t* buffer_b = buffer_a_2 + BUFFER_SLOTS_T * CHUNK_SIZE_A_T;
    half_t* buffer_C = buffer_b + BUFFER_SLOTS_B_T * CHUNK_SIZE_B_T;
    half_t* buffer_C_2 = buffer_C + CHUNK_SIZE_C_T;

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ cuda::barrier<cuda::thread_scope_block> filled[BUFFER_SLOTS_T * 2], drained[BUFFER_SLOTS_T * 2];
    __shared__ cuda::barrier<cuda::thread_scope_block> filled_b[BUFFER_SLOTS_B_T * 2], drained_b[BUFFER_SLOTS_B_T];
    __shared__ __align__(8) ClusterBarrier::ValueType cluster_drained_a[BUFFER_SLOTS_T * 2];


    // A barriers
    if (threadIdx.x < BUFFER_SLOTS_T) {
        init(&filled[threadIdx.x], NUM_CONSUMER_THREADS + 1);
        init(&drained[threadIdx.x], NUM_CONSUMER_THREADS + 1);
    }

    if (threadIdx.x >= BUFFER_SLOTS_T && threadIdx.x < BUFFER_SLOTS_T * 2) {
        init(&filled[threadIdx.x], NUM_CONSUMER_THREADS + 1);
        init(&drained[threadIdx.x], NUM_CONSUMER_THREADS + 1);
    }

    // B barriers
    if (threadIdx.x < BUFFER_SLOTS_B_T) {
        init(&filled_b[threadIdx.x], NUM_CONSUMER_THREADS + 1);
        init(&drained_b[threadIdx.x], NUM_CONSUMER_THREADS * 2 + 1);
    }

    if (threadIdx.x >= BUFFER_SLOTS_B_T && threadIdx.x < BUFFER_SLOTS_B_T * 2) {
        init(&filled_b[threadIdx.x], NUM_CONSUMER_THREADS + 1);
    }

    if (threadIdx.x < BUFFER_SLOTS_T * 2) {
        ClusterBarrier::init(&cluster_drained_a[threadIdx.x], CLUSTER_SIZE);
    }

    __syncthreads();
    cute::cluster_sync();


    int num_tiles_k = (K + TILE_K_T - 1) / TILE_K_T;
    int num_tiles_m = (M + TILE_M_T - 1) / TILE_M_T / 2;

    int cluster_rank = cute::block_rank_in_cluster();
    constexpr uint16_t tma_mcast_mask = (uint16_t(1) << CLUSTER_SIZE) - 1;

    int blockId = blockIdx.x;
    int num_sms_per_m_tile_row = N / TILE_N_T;
    int tile_in = blockId % num_sms_per_m_tile_row;
    int num_copy_blocks = gridDim.x / (num_sms_per_m_tile_row * sms_per_row);
    blockId /= (num_sms_per_m_tile_row * sms_per_row);

    int tile_ik_start = blockIdx.x % sms_per_row;

    if (threadIdx.x == NUM_CONSUMER_THREADS * 2) { // producer 0: A0 + B
        uint32_t cluster_drain_phase[BUFFER_SLOTS_T] = {};
        int nbuff = 0;
        for (int tile_im = blockId; tile_im < num_tiles_m; tile_im += num_copy_blocks) {
            for (int tile_ik = tile_ik_start; tile_ik < num_tiles_k; tile_ik += sms_per_row) {
                int next_slot = nbuff % BUFFER_SLOTS_T;
                int next_slot_b = nbuff % BUFFER_SLOTS_B_T;
                drained[next_slot].arrive_and_wait();

                if (cluster_rank == 0) {
                    ClusterBarrier::wait(&cluster_drained_a[next_slot], cluster_drain_phase[next_slot]);
                    cluster_drain_phase[next_slot] ^= 1;
                }

                (void) cuda::device::barrier_arrive_tx(filled[next_slot], 1, (CHUNK_SIZE_A_T) * sizeof(half_t));

                if (cluster_rank == 0) {
                    half_t* dst_ptr = buffer_a + next_slot * CHUNK_SIZE_A_T;
                    cp_async_bulk_tensor_4d_global_to_shared_multicast(dst_ptr, &a_tensor_map, 0, 0, 2 * tile_im * (TILE_M_T / 64), tile_ik * (TILE_K_T/8), &filled[next_slot], tma_mcast_mask);
                }

                drained_b[next_slot_b].arrive_and_wait();
                (void) cuda::device::barrier_arrive_tx(filled_b[next_slot_b], 1, (CHUNK_SIZE_B_T) * sizeof(half_t));

                half_t *dst_ptr_b = buffer_b + next_slot_b * CHUNK_SIZE_B_T;
                cde::cp_async_bulk_tensor_3d_global_to_shared(dst_ptr_b, &b_tensor_map, 0, tile_in * TILE_N_T, tile_ik * (TILE_K_T/64), filled_b[next_slot_b]);


                nbuff++;
            }
        }
    } else if (threadIdx.x == NUM_CONSUMER_THREADS * 2 + NUM_PRODUCER_THREADS) { // producer 1: A1
        uint32_t cluster_drain_phase[BUFFER_SLOTS_T] = {};
        int nbuff = 0;
        for (int tile_im = blockId; tile_im < num_tiles_m; tile_im += num_copy_blocks) {
            for (int tile_ik = tile_ik_start; tile_ik < num_tiles_k; tile_ik += sms_per_row) {
                int next_slot = nbuff % BUFFER_SLOTS_T;

                drained[next_slot + BUFFER_SLOTS_T].arrive_and_wait();

                if (cluster_rank == 0) {
                    ClusterBarrier::wait(&cluster_drained_a[next_slot + BUFFER_SLOTS_T], cluster_drain_phase[next_slot]);
                    cluster_drain_phase[next_slot] ^= 1;
                }

                (void) cuda::device::barrier_arrive_tx(filled[next_slot + BUFFER_SLOTS_T], 1, (CHUNK_SIZE_A_T) * sizeof(half_t));

                if (cluster_rank == 0) {
                    half_t* dst_ptr = buffer_a_2 + next_slot * CHUNK_SIZE_A_T;
                    // const half_t* src_ptr = A + (2 * tile_im + 1 + 2 * num_tiles_m * tile_ik) * CHUNK_SIZE_A_T;
                    // cp_async_bulk_global_to_shared_multicast(dst_ptr, src_ptr, CHUNK_SIZE_A_T * sizeof(half_t), &filled[next_slot + BUFFER_SLOTS_T], tma_mcast_mask);
                    cp_async_bulk_tensor_4d_global_to_shared_multicast(dst_ptr, &a_tensor_map, 0, 0, (2 * tile_im + 1) * (TILE_M_T / 64), tile_ik * (TILE_K_T/8), &filled[next_slot+BUFFER_SLOTS_T], tma_mcast_mask);

                }

                nbuff++;
            }
        }
    } else if (threadIdx.x < NUM_CONSUMER_THREADS) {
        #pragma unroll
        for (int i = 0; i < BUFFER_SLOTS_T; i++) {
            (void) drained[i].arrive();
            if (threadIdx.x == 0) {
                cluster_barrier_arrive_remote(&cluster_drained_a[i], 0);
            }
        }

        #pragma unroll
        for (int i = 0; i < BUFFER_SLOTS_B_T; i++) {
            (void) drained_b[i].arrive();
        }
        
        int nbuf = 0;
        for (int tile_im = blockId; tile_im < num_tiles_m; tile_im += num_copy_blocks) {
            task_wgmma_local<TILE_M_T, TILE_N_T, TILE_K_T, BUFFER_SLOTS_T, BUFFER_SLOTS_B_T, CHUNK_SIZE_A_T, CHUNK_SIZE_B_T, 0>
                (buffer_a, buffer_b, buffer_C,
                 threadIdx.x, nbuf, tile_ik_start, sms_per_row,
                 num_tiles_k, filled, drained, filled_b, drained_b, cluster_drained_a);

            if (threadIdx.x == 0) {
                cde::cp_async_bulk_tensor_4d_shared_to_global(&c_tensor_map, 0, 0, (tile_im * 2) * TILE_M_T/64, tile_in * (TILE_N_T/8), buffer_C);
                cde::cp_async_bulk_commit_group();
                cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
            }

        }
    } else if (threadIdx.x < 2 * NUM_CONSUMER_THREADS) {
        #pragma unroll
        for (int i = 0; i < BUFFER_SLOTS_T; i++) {
            (void) drained[i + BUFFER_SLOTS_T].arrive();
            if (threadIdx.x == NUM_CONSUMER_THREADS) {
                cluster_barrier_arrive_remote(&cluster_drained_a[i + BUFFER_SLOTS_T], 0);
            }
        }

        #pragma unroll
        for (int i = 0; i < BUFFER_SLOTS_B_T; i++) {
            (void) drained_b[i].arrive();
        }

        int tid = threadIdx.x - NUM_CONSUMER_THREADS;
        
        int nbuf = 0;
        for (int tile_im = blockId; tile_im < num_tiles_m; tile_im += num_copy_blocks) {
            task_wgmma_local<TILE_M_T, TILE_N_T, TILE_K_T, BUFFER_SLOTS_T, BUFFER_SLOTS_B_T, CHUNK_SIZE_A_T, CHUNK_SIZE_B_T, 1>
                (buffer_a_2, buffer_b, buffer_C_2,
                 tid, nbuf, tile_ik_start, sms_per_row,
                 num_tiles_k, filled + BUFFER_SLOTS_T, drained + BUFFER_SLOTS_T, filled_b, drained_b, cluster_drained_a + BUFFER_SLOTS_T);

            if (tid == 0) {
                cde::cp_async_bulk_tensor_4d_shared_to_global(&c_tensor_map, 0, 0, (tile_im *2 + 1) * TILE_M_T/64, tile_in * (TILE_N_T/8), buffer_C_2);
                cde::cp_async_bulk_commit_group();
                cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
            }

        }
    }

}



template<int TILE_M_T, int TILE_N_T,int TILE_K_T, int BUFFER_SLOTS_T, int BUFFER_SLOTS_B_T, int CHUNK_SIZE_A_T, int CHUNK_SIZE_B_T, int CHUNK_SIZE_C_T>
__device__ void loop(
    const CUtensorMap* a_tensor_map,
    const CUtensorMap* b_tensor_map,
    const CUtensorMap* c_tensor_map,
    int M,
    int N,
    int K,
    int M_offset,
    int local_block_idx,
    int num_blocks_in_domain,
    int sms_per_row
) {
    extern __shared__ __align__(128) half_t shared_mem[];


    half_t* buffer_a = shared_mem;
    half_t* buffer_a_2 = buffer_a + BUFFER_SLOTS_T * CHUNK_SIZE_A_T; // double buffer for A
    half_t* buffer_b = buffer_a_2 + BUFFER_SLOTS_T * CHUNK_SIZE_A_T;
    half_t* buffer_C = buffer_b + BUFFER_SLOTS_B_T * CHUNK_SIZE_B_T;
    half_t* buffer_C_2 = buffer_C + CHUNK_SIZE_C_T;

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ cuda::barrier<cuda::thread_scope_block> filled[BUFFER_SLOTS_T * 2], drained[BUFFER_SLOTS_T * 2];
    __shared__ cuda::barrier<cuda::thread_scope_block> filled_b[BUFFER_SLOTS_B_T * 2], drained_b[BUFFER_SLOTS_B_T];
    __shared__ __align__(8) ClusterBarrier::ValueType cluster_drained_a[BUFFER_SLOTS_T * 2];


    // A barriers
    if (threadIdx.x < BUFFER_SLOTS_T) {
        init(&filled[threadIdx.x], NUM_CONSUMER_THREADS + 1);
        init(&drained[threadIdx.x], NUM_CONSUMER_THREADS + 1);
    }

    if (threadIdx.x >= BUFFER_SLOTS_T && threadIdx.x < BUFFER_SLOTS_T * 2) {
        init(&filled[threadIdx.x], NUM_CONSUMER_THREADS + 1);
        init(&drained[threadIdx.x], NUM_CONSUMER_THREADS + 1);
    }

    // B barriers
    if (threadIdx.x < BUFFER_SLOTS_B_T) {
        init(&filled_b[threadIdx.x], NUM_CONSUMER_THREADS + 1);
        init(&drained_b[threadIdx.x], NUM_CONSUMER_THREADS * 2 + 1);
    }

    if (threadIdx.x >= BUFFER_SLOTS_B_T && threadIdx.x < BUFFER_SLOTS_B_T * 2) {
        init(&filled_b[threadIdx.x], NUM_CONSUMER_THREADS + 1);
    }

    if (threadIdx.x < BUFFER_SLOTS_T * 2) {
        ClusterBarrier::init(&cluster_drained_a[threadIdx.x], CLUSTER_SIZE);
    }


    __syncthreads();
    cute::cluster_sync();


    int num_tiles_k = (K + TILE_K_T - 1) / TILE_K_T;
    int num_tiles_m = (M + TILE_M_T - 1) / TILE_M_T / 2;

    int cluster_rank = cute::block_rank_in_cluster();
    constexpr uint16_t tma_mcast_mask = (uint16_t(1) << CLUSTER_SIZE) - 1;

    int blockId = local_block_idx;
    int num_sms_per_m_tile_row = N / TILE_N_T;
    int tile_in = blockId % num_sms_per_m_tile_row;
    int num_copy_blocks = num_blocks_in_domain / (num_sms_per_m_tile_row * sms_per_row);
    blockId /= (num_sms_per_m_tile_row * sms_per_row);

    int tile_ik_start = local_block_idx % sms_per_row;

    if (threadIdx.x == NUM_CONSUMER_THREADS * 2) { // producer 0: A0 + B
        uint32_t cluster_drain_phase[BUFFER_SLOTS_T] = {};
        int nbuff = 0;
        for (int tile_im = blockId; tile_im < num_tiles_m; tile_im += num_copy_blocks) {
            for (int tile_ik = tile_ik_start; tile_ik < num_tiles_k; tile_ik += sms_per_row) {
                int next_slot = nbuff % BUFFER_SLOTS_T;
                int next_slot_b = nbuff % BUFFER_SLOTS_B_T;
                drained[next_slot].arrive_and_wait();

                if (cluster_rank == 0) {
                    ClusterBarrier::wait(&cluster_drained_a[next_slot], cluster_drain_phase[next_slot]);
                    cluster_drain_phase[next_slot] ^= 1;
                }

                (void) cuda::device::barrier_arrive_tx(filled[next_slot], 1, (CHUNK_SIZE_A_T) * sizeof(half_t));

                if (cluster_rank == 0) {
                    half_t* dst_ptr = buffer_a + next_slot * CHUNK_SIZE_A_T;
                    cp_async_bulk_tensor_4d_global_to_shared_multicast(dst_ptr, a_tensor_map, 0, 0, 2 * tile_im * (TILE_M_T / 64), tile_ik * (TILE_K_T/8), &filled[next_slot], tma_mcast_mask);
                }

                drained_b[next_slot_b].arrive_and_wait();
                (void) cuda::device::barrier_arrive_tx(filled_b[next_slot_b], 1, (CHUNK_SIZE_B_T) * sizeof(half_t));

                half_t *dst_ptr_b = buffer_b + next_slot_b * CHUNK_SIZE_B_T;
                cde::cp_async_bulk_tensor_3d_global_to_shared(dst_ptr_b, b_tensor_map, 0, tile_in * TILE_N_T, tile_ik * (TILE_K_T/64), filled_b[next_slot_b]);


                nbuff++;
            }
        }
    } else if (threadIdx.x == NUM_CONSUMER_THREADS * 2 + NUM_PRODUCER_THREADS) { // producer 1: A1
        uint32_t cluster_drain_phase[BUFFER_SLOTS_T] = {};
        int nbuff = 0;
        for (int tile_im = blockId; tile_im < num_tiles_m; tile_im += num_copy_blocks) {
            for (int tile_ik = tile_ik_start; tile_ik < num_tiles_k; tile_ik += sms_per_row) {
                int next_slot = nbuff % BUFFER_SLOTS_T;

                drained[next_slot + BUFFER_SLOTS_T].arrive_and_wait();

                if (cluster_rank == 0) {
                    ClusterBarrier::wait(&cluster_drained_a[next_slot + BUFFER_SLOTS_T], cluster_drain_phase[next_slot]);
                    cluster_drain_phase[next_slot] ^= 1;
                }

                (void) cuda::device::barrier_arrive_tx(filled[next_slot + BUFFER_SLOTS_T], 1, (CHUNK_SIZE_A_T) * sizeof(half_t));

                if (cluster_rank == 0) {
                    half_t* dst_ptr = buffer_a_2 + next_slot * CHUNK_SIZE_A_T;
                    cp_async_bulk_tensor_4d_global_to_shared_multicast(dst_ptr, a_tensor_map, 0, 0, (2 * tile_im + 1) * (TILE_M_T / 64), tile_ik * (TILE_K_T/8), &filled[next_slot+BUFFER_SLOTS_T], tma_mcast_mask);
                }

                nbuff++;
            }
        }
    } else if (threadIdx.x < NUM_CONSUMER_THREADS) {
        #pragma unroll
        for (int i = 0; i < BUFFER_SLOTS_T; i++) {
            (void) drained[i].arrive();
            if (threadIdx.x == 0) {
                cluster_barrier_arrive_remote(&cluster_drained_a[i], 0);
            }
        }

        #pragma unroll
        for (int i = 0; i < BUFFER_SLOTS_B_T; i++) {
            (void) drained_b[i].arrive();
        }
        
        int nbuf = 0;
        for (int tile_im = blockId; tile_im < num_tiles_m; tile_im += num_copy_blocks) {
            task_wgmma_local<TILE_M_T, TILE_N_T, TILE_K_T, BUFFER_SLOTS_T, BUFFER_SLOTS_B_T, CHUNK_SIZE_A_T, CHUNK_SIZE_B_T, 0>
                (buffer_a, buffer_b, buffer_C,
                 threadIdx.x, nbuf, tile_ik_start, sms_per_row,
                 num_tiles_k, filled, drained, filled_b, drained_b, cluster_drained_a);

            if (threadIdx.x == 0) {
                cde::cp_async_bulk_tensor_4d_shared_to_global(c_tensor_map, 0, 0, (tile_im * 2) * TILE_M_T/64 + M_offset/64, tile_in * (TILE_N_T/8), buffer_C);
                cde::cp_async_bulk_commit_group();
                cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
            }

        }
    } else if (threadIdx.x < 2 * NUM_CONSUMER_THREADS) {
        #pragma unroll
        for (int i = 0; i < BUFFER_SLOTS_T; i++) {
            (void) drained[i + BUFFER_SLOTS_T].arrive();
            if (threadIdx.x == NUM_CONSUMER_THREADS) {
                cluster_barrier_arrive_remote(&cluster_drained_a[i + BUFFER_SLOTS_T], 0);
            }
        }

        #pragma unroll
        for (int i = 0; i < BUFFER_SLOTS_B_T; i++) {
            (void) drained_b[i].arrive();
        }

        int tid = threadIdx.x - NUM_CONSUMER_THREADS;
        
        int nbuf = 0;
        for (int tile_im = blockId; tile_im < num_tiles_m; tile_im += num_copy_blocks) {
            task_wgmma_local<TILE_M_T, TILE_N_T, TILE_K_T, BUFFER_SLOTS_T, BUFFER_SLOTS_B_T, CHUNK_SIZE_A_T, CHUNK_SIZE_B_T, 1>
                (buffer_a_2, buffer_b, buffer_C_2,
                 tid, nbuf, tile_ik_start, sms_per_row,
                 num_tiles_k, filled + BUFFER_SLOTS_T, drained + BUFFER_SLOTS_T, filled_b, drained_b, cluster_drained_a + BUFFER_SLOTS_T);

            if (tid == 0) {
                cde::cp_async_bulk_tensor_4d_shared_to_global(c_tensor_map, 0, 0, (tile_im *2 + 1) * TILE_M_T/64 + M_offset/64, tile_in * (TILE_N_T/8), buffer_C_2);
                cde::cp_async_bulk_commit_group();
                cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
            }

        }
    }

}




template<int TILE_M_T, int TILE_N_T,int TILE_K_T, int BUFFER_SLOTS_T, int BUFFER_SLOTS_B_T, int CHUNK_SIZE_A_T, int CHUNK_SIZE_B_T, int CHUNK_SIZE_C_T>
__global__ __cluster_dims__(CLUSTER_SIZE, 1, 1) void gemv_horizontal_multicast (
    const __grid_constant__ CUtensorMap h_a_tensor_map,
    const __grid_constant__ CUtensorMap d_a_tensor_map,
    const __grid_constant__ CUtensorMap b_tensor_map,
    const __grid_constant__ CUtensorMap c_tensor_map, 
    int h_M,
    int d_M,
    int N,
    int K, 
    int num_copy_blocks,
    int num_copy_host_blocks,
    int h_sms_per_row,
    int d_sms_per_row
) {
    int sm_id = blockIdx.x;
    if (sm_id >= num_copy_blocks){
        return;
    }

    if (sm_id < num_copy_host_blocks) {
        loop<TILE_M_T, TILE_N_T, TILE_K_T, BUFFER_SLOTS_T, BUFFER_SLOTS_B_T, CHUNK_SIZE_A_T, CHUNK_SIZE_B_T, CHUNK_SIZE_C_T>(&h_a_tensor_map, &b_tensor_map, &c_tensor_map, h_M, N, K, 0, sm_id, num_copy_host_blocks, h_sms_per_row);
    } else {
        loop<TILE_M_T, TILE_N_T, TILE_K_T, BUFFER_SLOTS_T, BUFFER_SLOTS_B_T, CHUNK_SIZE_A_T, CHUNK_SIZE_B_T, CHUNK_SIZE_C_T>(&d_a_tensor_map, &b_tensor_map, &c_tensor_map, d_M, N, K, h_M, sm_id - num_copy_host_blocks, num_copy_blocks - num_copy_host_blocks, d_sms_per_row);
    }
}



template<int TILE_M_T, int TILE_N_T, int TILE_K_T, int BUFFER_SLOTS_T, int BUFFER_SLOTS_B_T, int TILE_ELEMS_A_T, int TILE_ELEMS_B_T, int GROUP_ID>
__device__ __forceinline__ void task_wgmma_prefill(
    half_t * __restrict__ a_buffer, half_t * __restrict__ b_buffer, half_t * __restrict__ sC,
    int thread_id, int &nbuf, int tile_ik_base, int tile_ik_stride,
    const int NUM_TILES_K, 
    cuda::barrier<cuda::thread_scope_block> filled[], cuda::barrier<cuda::thread_scope_block> drained[],
    cuda::barrier<cuda::thread_scope_block> filled_b[], cuda::barrier<cuda::thread_scope_block> drained_b[]
) {
    using namespace cute;
    static_assert(TILE_N_T == 128 || TILE_N_T % 256 == 0, "TILE_N_T must be 128 or a multiple of 256");
    static_assert(TILE_M_T % 64 == 0, "TILE_M_T must be a multiple of 64");
    static_assert(TILE_K_T % 16 == 0, "TILE_K_T must be a multiple of 16");
    using GmmaAtom = std::conditional_t<
        TILE_N_T == 128,
        SM90_64x128x16_F16F16F16_SS<GMMA::Major::MN, GMMA::Major::K>,
        SM90_64x256x16_F16F16F16_SS<GMMA::Major::MN, GMMA::Major::K>>;

    // Make a bigger "kernel" by repeating the atom
    auto tiled_mma = make_tiled_mma(
        MMA_Atom<GmmaAtom>{},
        make_layout(make_shape(Int<1>{}, Int<1>{}, Int<1>{})), // number of thread-parallel atoms
        make_tile(Int<TILE_M_T>{}, Int<TILE_N_T>{}, Int<TILE_K_T>{}) // number of wgmma instructions inside one warp group, MAY increase the useage of registers.
    );
    auto thr_mma = tiled_mma.get_slice(thread_id);
    
    auto layout_sA = tile_to_shape(
        GMMA::Layout_MN_SW128_Atom<half_t>{},
        make_shape(Int<TILE_M_T>{},Int<TILE_K_T>{}));


    auto layout_sB = tile_to_shape(
        GMMA::Layout_K_SW128_Atom<half_t>{},
        make_shape(Int<TILE_N_T>{}, Int<TILE_K_T>{}));

    // C must be laid out as MxN to match the GMMA accumulator fragments
    auto layout_sC = tile_to_shape(
        GMMA::Layout_MN_SW128_Atom<half_t>{},
        make_shape(Int<TILE_M_T>{}, Int<TILE_N_T>{}));

    Tensor t_sC = make_tensor(
        make_smem_ptr((half_t*)sC),
        layout_sC
    );

    auto frag_C = thr_mma.partition_fragment_C(t_sC);
    clear(frag_C);

    using CopyAtomC = Copy_Atom<SM90_U16x8_STSM_T, half_t>;
    auto smem_tiled_copy_C = make_tiled_copy_C(CopyAtomC{}, tiled_mma);
    auto smem_thr_copy_C = smem_tiled_copy_C.get_thread_slice(thread_id);

    for (int tile_ik = tile_ik_base; tile_ik < NUM_TILES_K ; tile_ik += tile_ik_stride) {
        int next_slot = nbuf % BUFFER_SLOTS_T;
        int next_slot_b = nbuf % BUFFER_SLOTS_B_T;
        if constexpr (GROUP_ID == 0) {
            filled[next_slot].arrive_and_wait();
            filled_b[next_slot_b].arrive_and_wait();
        } else {
            filled[next_slot].arrive_and_wait();
            filled_b[next_slot_b + BUFFER_SLOTS_B_T].arrive_and_wait();
        }

        half_t *sA = (half_t*)a_buffer + next_slot * TILE_ELEMS_A_T;
        half_t *sB = (half_t*)b_buffer + next_slot_b * TILE_ELEMS_B_T;

        auto t_sA = make_tensor(make_smem_ptr(sA), layout_sA);
        auto t_sB = make_tensor(make_smem_ptr(sB), layout_sB);

        auto frag_A = thr_mma.partition_fragment_A(t_sA);
        auto frag_B = thr_mma.partition_fragment_B(t_sB);

        warpgroup_arrive();
        gemm(tiled_mma, frag_A, frag_B, frag_C);   // emit multiple wgmma instructions by cute, see make_tiled_mma        
        warpgroup_commit_batch();

        if (GROUP_ID == 0 && thread_id == 0) {
            (void) filled_b[next_slot_b + BUFFER_SLOTS_B_T].arrive();
        }

        warpgroup_wait<0>();
        (void) drained[next_slot].arrive();
        (void) drained_b[next_slot_b].arrive();
        nbuf++;
    }

    auto tCsC = smem_thr_copy_C.partition_D(t_sC);
    auto tCrC = smem_thr_copy_C.retile_S(frag_C);
    copy(smem_tiled_copy_C, tCrC, tCsC);
    cuda::ptx::fence_proxy_async();
}


template<int TILE_M_T, int TILE_N_T,int TILE_K_T, int BUFFER_SLOTS_T, int BUFFER_SLOTS_B_T, int CHUNK_SIZE_A_T, int CHUNK_SIZE_B_T, int CHUNK_SIZE_C_T>
__device__ void loop_prefill(const CUtensorMap* a_tensor_map, const CUtensorMap* b_tensor_map, const CUtensorMap* c_tensor_map,
                     int M, int N, int K, int M_offset, int local_block_idx, int num_blocks_in_domain, int sms_per_row) {
    extern __shared__ __align__(128) half_t shared_mem[];


    half_t* buffer_a = shared_mem;
    half_t* buffer_a_2 = buffer_a + BUFFER_SLOTS_T * CHUNK_SIZE_A_T; // double buffer for A
    half_t* buffer_b = buffer_a_2 + BUFFER_SLOTS_T * CHUNK_SIZE_A_T;
    half_t* buffer_C = buffer_b + BUFFER_SLOTS_B_T * CHUNK_SIZE_B_T;
    half_t* buffer_C_2 = buffer_C + CHUNK_SIZE_C_T;

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ cuda::barrier<cuda::thread_scope_block> filled[BUFFER_SLOTS_T * 2], drained[BUFFER_SLOTS_T * 2];
    __shared__ cuda::barrier<cuda::thread_scope_block> filled_b[BUFFER_SLOTS_B_T * 2], drained_b[BUFFER_SLOTS_B_T];


    // A barriers
    if (threadIdx.x < BUFFER_SLOTS_T) {
        init(&filled[threadIdx.x], NUM_CONSUMER_THREADS + 1);
        init(&drained[threadIdx.x], NUM_CONSUMER_THREADS + 1);
    }

    if (threadIdx.x >= BUFFER_SLOTS_T && threadIdx.x < BUFFER_SLOTS_T * 2) {
        init(&filled[threadIdx.x], NUM_CONSUMER_THREADS + 1);
        init(&drained[threadIdx.x], NUM_CONSUMER_THREADS + 1);
    }

    // B barriers
    if (threadIdx.x < BUFFER_SLOTS_B_T) {
        init(&filled_b[threadIdx.x], NUM_CONSUMER_THREADS + 1);
        init(&drained_b[threadIdx.x], NUM_CONSUMER_THREADS * 2 + 1);
    }

    if (threadIdx.x >= BUFFER_SLOTS_B_T && threadIdx.x < BUFFER_SLOTS_B_T * 2) {
        init(&filled_b[threadIdx.x], NUM_CONSUMER_THREADS + 1);
    }

    // if (threadIdx.x < BUFFER_SLOTS_B_T) {
    //     init(&filled[threadIdx.x], NUM_CONSUMER_THREADS + 1); // +1 for producer
    //     init(&drained[threadIdx.x], NUM_CONSUMER_THREADS + 1); // +1 for producer
    //     // cuda::device::barrier_expect_tx(filled[threadIdx.x], (CHUNK_SIZE_A_T) * sizeof(half_t)); // only A for now

    //     init(&filled_b[threadIdx.x], NUM_CONSUMER_THREADS + 1); // +1 for producer
    //     init(&drained_b[threadIdx.x], NUM_CONSUMER_THREADS + 1); // +1 for producer
    // } else if (threadIdx.x < BUFFER_SLOTS_T) {
    //     init(&filled[threadIdx.x], NUM_CONSUMER_THREADS + 1); // +1 for producer
    //     init(&drained[threadIdx.x], NUM_CONSUMER_THREADS + 1); // +1 for producer
    // }
    __syncthreads();
    cute::cluster_sync();


    int num_tiles_k = (K + TILE_K_T - 1) / TILE_K_T;
    int num_tiles_m = (M + TILE_M_T - 1) / TILE_M_T / 2;

    int sm_id = get_smid();

    int blockId = local_block_idx;
    int num_sms_per_m_tile_row = N / TILE_N_T;
    int tile_in = blockId % num_sms_per_m_tile_row;
    int num_copy_blocks = num_blocks_in_domain / (num_sms_per_m_tile_row * sms_per_row);
    blockId /= (num_sms_per_m_tile_row * sms_per_row);

    int tile_ik_start = local_block_idx % sms_per_row;

    if (threadIdx.x == NUM_CONSUMER_THREADS * 2) { // producer 0: A0 + B
        int nbuff = 0;
        for (int tile_im = blockId; tile_im < num_tiles_m; tile_im += num_copy_blocks) {
            for (int tile_ik = tile_ik_start; tile_ik < num_tiles_k; tile_ik += sms_per_row) {
                int next_slot = nbuff % BUFFER_SLOTS_T;
                int next_slot_b = nbuff % BUFFER_SLOTS_B_T;
                drained[next_slot].arrive_and_wait();

                (void) cuda::device::barrier_arrive_tx(filled[next_slot], 1, (CHUNK_SIZE_A_T) * sizeof(half_t));

                // if (cluster_rank == 0) {
                //     half_t* dst_ptr = buffer_a + next_slot * CHUNK_SIZE_A_T;
                //     // const half_t* src_ptr = A + (2 * tile_im + 2 * num_tiles_m * tile_ik) * CHUNK_SIZE_A_T;
                //     // cp_async_bulk_global_to_shared_multicast(dst_ptr, src_ptr, CHUNK_SIZE_A_T * sizeof(half_t), &filled[next_slot], tma_mcast_mask);
                //     cp_async_bulk_tensor_4d_global_to_shared_multicast(dst_ptr, a_tensor_map, 0, 0, 2 * tile_im * (TILE_M_T / 64), tile_ik * (TILE_K_T/8), &filled[next_slot], tma_mcast_mask);
                // }
                half_t* dst_ptr = buffer_a + next_slot * CHUNK_SIZE_A_T;
                cde::cp_async_bulk_tensor_4d_global_to_shared(dst_ptr, a_tensor_map, 0, 0, 2 * tile_im * (TILE_M_T / 64), tile_ik * (TILE_K_T/8), filled[next_slot]);



                drained_b[next_slot_b].arrive_and_wait();
                (void) cuda::device::barrier_arrive_tx(filled_b[next_slot_b], 1, (CHUNK_SIZE_B_T) * sizeof(half_t));

                half_t *dst_ptr_b = buffer_b + next_slot_b * CHUNK_SIZE_B_T;
                cde::cp_async_bulk_tensor_3d_global_to_shared(dst_ptr_b, b_tensor_map, 0, tile_in * TILE_N_T, tile_ik * (TILE_K_T/64), filled_b[next_slot_b]);

                nbuff++;
            }
        }
    } else if (threadIdx.x == NUM_CONSUMER_THREADS * 2 + NUM_PRODUCER_THREADS) { // producer 1: A1
        int nbuff = 0;
        for (int tile_im = blockId; tile_im < num_tiles_m; tile_im += num_copy_blocks) {
            for (int tile_ik = tile_ik_start; tile_ik < num_tiles_k; tile_ik += sms_per_row) {
                int next_slot = nbuff % BUFFER_SLOTS_T;

                drained[next_slot + BUFFER_SLOTS_T].arrive_and_wait();

                (void) cuda::device::barrier_arrive_tx(filled[next_slot + BUFFER_SLOTS_T], 1, (CHUNK_SIZE_A_T) * sizeof(half_t));

                // if (cluster_rank == 0) {
                //     half_t* dst_ptr = buffer_a_2 + next_slot * CHUNK_SIZE_A_T;
                //     // const half_t* src_ptr = A + (2 * tile_im + 1 + 2 * num_tiles_m * tile_ik) * CHUNK_SIZE_A_T;
                //     // cp_async_bulk_global_to_shared_multicast(dst_ptr, src_ptr, CHUNK_SIZE_A_T * sizeof(half_t), &filled[next_slot + BUFFER_SLOTS_T], tma_mcast_mask);
                //     cp_async_bulk_tensor_4d_global_to_shared_multicast(dst_ptr, a_tensor_map, 0, 0, (2 * tile_im + 1) * (TILE_M_T / 64), tile_ik * (TILE_K_T/8), &filled[next_slot+BUFFER_SLOTS_T], tma_mcast_mask);
                // }
                half_t* dst_ptr = buffer_a_2 + next_slot * CHUNK_SIZE_A_T;
                cde::cp_async_bulk_tensor_4d_global_to_shared(dst_ptr, a_tensor_map, 0, 0, (2 * tile_im + 1) * (TILE_M_T / 64), tile_ik * (TILE_K_T/8), filled[next_slot+BUFFER_SLOTS_T]);

                nbuff++;
            }
        }
    } else if (threadIdx.x < NUM_CONSUMER_THREADS) {
        #pragma unroll
        for (int i = 0; i < BUFFER_SLOTS_T; i++) {
            (void) drained[i].arrive();
        }

        #pragma unroll
        for (int i = 0; i < BUFFER_SLOTS_B_T; i++) {
            (void) drained_b[i].arrive();
        }
        
        int nbuf = 0;
        for (int tile_im = blockId; tile_im < num_tiles_m; tile_im += num_copy_blocks) {
            task_wgmma_prefill<TILE_M_T, TILE_N_T, TILE_K_T, BUFFER_SLOTS_T, BUFFER_SLOTS_B_T, CHUNK_SIZE_A_T, CHUNK_SIZE_B_T, 0>
                (buffer_a, buffer_b, buffer_C,
                 threadIdx.x, nbuf, tile_ik_start, sms_per_row,
                 num_tiles_k, filled, drained, filled_b, drained_b);
            __sync_barrier<13, NUM_CONSUMER_THREADS>();
            if (threadIdx.x == 0) {
                cde::cp_async_bulk_tensor_4d_shared_to_global(c_tensor_map, 0, 0, (tile_im * 2) * TILE_M_T/64 + M_offset/64, tile_in * (TILE_N_T/8), buffer_C);
                cde::cp_async_bulk_commit_group();
                cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
            }
            __sync_barrier<13, NUM_CONSUMER_THREADS>();

        }
    } else if (threadIdx.x < 2 * NUM_CONSUMER_THREADS) {
        #pragma unroll
        for (int i = 0; i < BUFFER_SLOTS_T; i++) {
            (void) drained[i + BUFFER_SLOTS_T].arrive();
        }

        #pragma unroll
        for (int i = 0; i < BUFFER_SLOTS_B_T; i++) {
            (void) drained_b[i].arrive();
        }

        int tid = threadIdx.x - NUM_CONSUMER_THREADS;
        
        int nbuf = 0;
        for (int tile_im = blockId; tile_im < num_tiles_m; tile_im += num_copy_blocks) {
            task_wgmma_prefill<TILE_M_T, TILE_N_T, TILE_K_T, BUFFER_SLOTS_T, BUFFER_SLOTS_B_T, CHUNK_SIZE_A_T, CHUNK_SIZE_B_T, 1>
                (buffer_a_2, buffer_b, buffer_C_2,
                 tid, nbuf, tile_ik_start, sms_per_row,
                 num_tiles_k, filled + BUFFER_SLOTS_T, drained + BUFFER_SLOTS_T, filled_b, drained_b);
            __sync_barrier<15, NUM_CONSUMER_THREADS>();
            if (tid == 0) {
                cde::cp_async_bulk_tensor_4d_shared_to_global(c_tensor_map, 0, 0, (tile_im *2 + 1) * TILE_M_T/64 + M_offset/64, tile_in * (TILE_N_T/8), buffer_C_2);
                cde::cp_async_bulk_commit_group();
                cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
            }
            __sync_barrier<15, NUM_CONSUMER_THREADS>();

        }
    }
}


template<int TILE_M_T, int TILE_N_T,int TILE_K_T, int BUFFER_SLOTS_T, int BUFFER_SLOTS_B_T, int CHUNK_SIZE_A_T, int CHUNK_SIZE_B_T, int CHUNK_SIZE_C_T>
__global__  void gemv_multicast_prefill(
    const __grid_constant__ CUtensorMap a_tensor_map,
    const __grid_constant__ CUtensorMap b_tensor_map,
    const __grid_constant__ CUtensorMap c_tensor_map,
    int M,
    int N,
    int K,
    int num_blocks,
    int sms_per_row
) {
    int sm_id = blockIdx.x;
    if (sm_id >= num_blocks){
        return;
    }

    loop_prefill<TILE_M_T, TILE_N_T, TILE_K_T, BUFFER_SLOTS_T, BUFFER_SLOTS_B_T, CHUNK_SIZE_A_T, CHUNK_SIZE_B_T, CHUNK_SIZE_C_T>(&a_tensor_map, &b_tensor_map, &c_tensor_map, M, N, K, 0, sm_id, num_blocks, sms_per_row);
}


template<int TILE_M_T, int TILE_N_T,int TILE_K_T, int BUFFER_SLOTS_T, int BUFFER_SLOTS_B_T, int CHUNK_SIZE_A_T, int CHUNK_SIZE_B_T, int CHUNK_SIZE_C_T>
__global__ void gemv_horizontal_multicast_prefill (
    const __grid_constant__ CUtensorMap h_a_tensor_map,
    const __grid_constant__ CUtensorMap d_a_tensor_map,
    const __grid_constant__ CUtensorMap b_tensor_map,
    const __grid_constant__ CUtensorMap c_tensor_map, 
    int h_M,
    int d_M,
    int N,
    int K, 
    int num_copy_blocks,
    int num_copy_host_blocks,
    int h_sms_per_row,
    int d_sms_per_row
) {
    int sm_id = blockIdx.x;
    if (sm_id >= num_copy_blocks){
        return;
    }

    if (sm_id < num_copy_host_blocks) {
        loop_prefill<TILE_M_T, TILE_N_T, TILE_K_T, BUFFER_SLOTS_T, BUFFER_SLOTS_B_T, CHUNK_SIZE_A_T, CHUNK_SIZE_B_T, CHUNK_SIZE_C_T>(&h_a_tensor_map, &b_tensor_map, &c_tensor_map, h_M, N, K, 0, sm_id, num_copy_host_blocks, h_sms_per_row);
    } else {
        loop_prefill<TILE_M_T, TILE_N_T, TILE_K_T, BUFFER_SLOTS_T, BUFFER_SLOTS_B_T, CHUNK_SIZE_A_T, CHUNK_SIZE_B_T, CHUNK_SIZE_C_T>(&d_a_tensor_map, &b_tensor_map, &c_tensor_map, d_M, N, K, h_M, sm_id - num_copy_host_blocks, num_copy_blocks - num_copy_host_blocks, d_sms_per_row);
    }
}


