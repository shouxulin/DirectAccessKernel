from offload import runtime

TILE_M = 64
TILE_K = 256
TILE_N = 8
# arch-dependent (make arch=90a|120a), taken from the compiled extension so it matches include/task/config.cuh
BUFFER_SLOTS = runtime.BUFFER_SLOTS

TILE_M_MULTICAST = 64
TILE_N_MULTICAST = 256
TILE_K_MULTICAST = 64
BUFFER_SLOTS_MULTICAST = 3
BUFFER_SLOTS_B_MULTICAST = 3
