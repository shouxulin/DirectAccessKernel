// DAK_SM_ARCH is set by the build: `make arch=90a|120a` (forwarded to setup.py by `make pyext`)
#if !defined(DAK_SM_ARCH)
#error "DAK_SM_ARCH is not defined, build with `make arch=90a` or `make arch=120a`"
#elif DAK_SM_ARCH == 90     // GH200, sm_90a
#define NUM_SMS 132
#define MAX_COPY_SMS 128
// TILE_N=128 with 3 slots needs ~305 KB smem, over the 227 KB per-block limit
#if defined(TILE_N) && TILE_N == 128
#define BUFFER_SLOTS 2
#else
#define BUFFER_SLOTS 3
#endif
#elif DAK_SM_ARCH == 120    // RTX PRO 6000, sm_120a
#define NUM_SMS 188
#define MAX_COPY_SMS 188
#define BUFFER_SLOTS 2
#else
#error "Unsupported DAK_SM_ARCH, expected 90 or 120"
#endif


#define TILE_M 64
// sm_90a: set by `make pyext arch=90a TILE_N=<n>`, also selects the wgmma atom in gemv.cuh
#ifndef TILE_N
#define TILE_N 8
#endif
#define TILE_K 256

#define TILE_M_MULTICAST 64
#define TILE_N_MULTICAST 256
#define TILE_K_MULTICAST 64
#define BUFFER_SLOTS_MULTICAST 3
#define BUFFER_SLOTS_B_MULTICAST 3

#define CHUNK_SIZE_A (TILE_M_MULTICAST * TILE_K_MULTICAST)
#define CHUNK_SIZE_B (TILE_N_MULTICAST * TILE_K_MULTICAST)
#define CHUNK_SIZE_C (TILE_M_MULTICAST * TILE_N_MULTICAST)

#ifndef RECORD_NUM_CHUNKS
#define RECORD_NUM_CHUNKS 0
#endif

#define NUM_PRODUCER_THREADS 32
#define NUM_CONSUMER_THREADS 128

#define CLUSTER_SIZE 2