from offload import runtime
import torch

from .config import *


def get_smem_size(multicast=False):
    if not multicast:
        return ((TILE_M * TILE_K + TILE_N * TILE_K) * BUFFER_SLOTS + TILE_M * TILE_N) * 2 + 1024
    else:
        return (2 * TILE_M_MULTICAST * TILE_K_MULTICAST * BUFFER_SLOTS_MULTICAST + TILE_N_MULTICAST * TILE_K_MULTICAST * BUFFER_SLOTS_B_MULTICAST + 2 * TILE_M_MULTICAST * TILE_N_MULTICAST) * 2 + 1024


# get dims other than the specified index and last one
def get_other_dims(dim: int, i: int):
    return [j for j in range(dim-1) if j != i and j - dim != i]

def split_horizontal(weight_shape, offload_ratio):
    m = weight_shape[0]
    k = weight_shape[1]
    # if offload_ratio == 0.08:
    #     """ GH 200 hand-optimized config """
    #     if m == 28672 and k == 7168:
    #         return 2048, m - 2048
    #     elif m == 4096 and k == 4096:
    #         return 256, m - 256
    # if offload_ratio == 0.03:
    #     """ RTX 6000 hand-optimized config """
    #     if m == 7168 and k == 7168:
    #         return 128, m - 128
    #     elif m == 4096 and k == 28672:
    #         return 192, m - 192
    #     elif m == 28672 and k == 4096:
    #         return 832, m - 832
    #     elif m == 4096 and k == 4096:
    #         return 64, m - 64
    #     elif m == 4096 and k == 16384:
    #         return 64, m - 64
    #     elif m == 16384 and k == 4096:
    #         return 384, m - 384

    assert m % TILE_M == 0, f"Weight shape {weight_shape} is not compatible with tile size {TILE_M}"
    num_tiles = m // TILE_M
    m_h = int(num_tiles * offload_ratio) * TILE_M
    return m_h, m - m_h
    

def build_tma_wgmma_mn(mat: torch.Tensor, tileM: int, tileK: int, iK = -2, debug=False):
    assert iK != -1 and iK < len(mat.shape) - 1, "iK must not be the last dim"
    # build 4d by default
    assert len(mat.shape) >= 2, "Input matrix must be at least 2D"
    K, M = mat.shape[iK], mat.shape[-1]
    elsize = mat.element_size()

    blockM = 128 // elsize
    blockK = 8

    assert tileM % blockM == 0, f"tileM {tileM} must be multiple of blockM {blockM}"

    # if blockM != tileM:
    if debug: # TODO: check if need to have 64,8,... 
        global_dims = [blockM, blockK, M // blockM, K // blockK]
        global_strides = [mat.stride(iK), blockM, mat.stride(iK) * blockK]
        box_dims = [blockM, blockK, tileM // blockM, tileK // blockK]
    else:
        global_dims = [M, K]
        global_strides = [mat.stride(iK)]
        box_dims = [tileM, tileK]

    
    if len(mat.shape) > 2:
        # assert (False), "Currently only support 2D input, but got shape: {}".format(mat.shape)
        # collapse other dims
        size_otherthan_k_or_m = 1
        other_dims = get_other_dims(len(mat.shape), iK)
        inner_most_others = other_dims[-1]
        for i in other_dims:
            # consider M = negative index
            size_otherthan_k_or_m *= mat.shape[i]
        
        global_dims.append(size_otherthan_k_or_m)
        global_strides.append(mat.stride(inner_most_others))
        box_dims.append(1)

    rank = len(global_dims)
    box_strides = [1] * rank
    global_strides = [s * elsize for s in global_strides]


    # print(f"Data Dims: {global_dims}")
    # print(f"Data Strides: {global_strides}")
    # print(f"Box Dims: {box_dims}")

    return rank, runtime.build_tma_desc(
        mat,
        global_dims,
        global_strides,
        box_dims,
        box_strides,
        128
    )


def build_tma_wgmma_k(mat: torch.Tensor, tileK: int, tileN: int, iN: int = -2):
    assert iN != -1 and iN < len(mat.shape) - 1, "iN must not be the last dim"
    N, K = mat.shape[iN], mat.shape[-1]
    elsize = mat.element_size()
    blockK = 128 // elsize

    assert tileK % blockK == 0, "tileK must be multiple of blockK"

    glob_dims = [blockK, N, K // blockK]
    glob_strides = [mat.stride(iN), blockK]
    box_dims = [blockK, tileN, tileK // blockK]

    if len(mat.shape) > 2:
        # assert (False), "Currently only support 2D input, but got shape: {}".format(mat.shape)
        # collapse other dims
        size_otherthan_k_or_n = 1
        other_dims = get_other_dims(len(mat.shape), iN)
        inner_most_others = other_dims[-1]
        for i in other_dims:
            size_otherthan_k_or_n *= mat.shape[i]
        
        # find stride of inner most dims other than K or N
        glob_dims.append(size_otherthan_k_or_n)
        glob_strides.append(mat.stride(inner_most_others))
        box_dims.append(1)

    glob_strides = [s * elsize for s in glob_strides]
    rank = len(glob_dims)
    box_strides = [1] * rank

    # print(f"Data Dims: {glob_dims}")
    # print(f"Data Strides: {glob_strides}")
    # print(f"Box Dims: {box_dims}")

    return rank, runtime.build_tma_desc(
        mat,
        glob_dims,
        glob_strides,
        box_dims,
        box_strides,
        128
    )
