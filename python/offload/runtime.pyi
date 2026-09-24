"""Type stubs for offload.runtime C++ extension module."""

import torch

# Arch-dependent constants from include/task/config.cuh (set by `make arch=90a|120a`)
NUM_SMS: int
MAX_COPY_SMS: int
BUFFER_SLOTS: int

def print_device_name(device_id: int) -> None:
    """Print the name of the CUDA device with the given ID."""
    ...

def build_tma_desc(
    mat: torch.Tensor,
    global_dims: list[int],
    global_strides: list[int],
    box_dims: list[int],
    box_strides: list[int],
    max_shared_memory_size: int
) -> torch.Tensor:
    """Build CUtensorMap descriptor for given tensor and layout."""
    ...

def set_smem_size(smem_size: int, kernel_code: int = 0) -> int:
    """Set the maximum dynamic shared memory size for the kernel."""
    ...

def gemv(
    a_desc: torch.Tensor,
    b_desc: torch.Tensor,
    c_desc: torch.Tensor,
    M: int,
    N: int,
    K: int,
    num_copy_blocks: int,
    num_copy_host_blocks: int,
    smem_size: int,
    sms_per_row: int
) -> None:
    """Launch GEMV kernel with given TMA descriptors and parameters."""
    ...

def gemv_reduce_horizontal(
    a_desc_h: torch.Tensor,
    a_desc_d: torch.Tensor,
    b_desc: torch.Tensor,
    c_desc: torch.Tensor,
    h_M: int,
    d_M: int,
    N: int,
    K: int,
    num_copy_blocks: int,
    smem_size: int,
    h_sms_per_row: int,
    d_sms_per_row: int
) -> None:
    """Launch GEMV kernel with given TMA descriptors and parameters."""
    ...
