import torch
from torch import Tensor
from torch.nn import functional as F, init
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn.modules.module import Module
import math


from offload import runtime
from ..utils import build_tma_wgmma_mn, build_tma_wgmma_k, split_horizontal
from ..config import TILE_M, TILE_K, TILE_N


class Linear(Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        smem_size: int,
        num_copy_blocks: int,
        bias: bool = False,
        device=None,
        dtype=None,
        skip_init: bool = False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.smem_size = smem_size
        self.num_copy_blocks = num_copy_blocks
        self.sms_per_row = 1
        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        if not skip_init:
            self.reset_parameters()
        self.w_desc = None

        self.output = None
        self.output_shape = None
        self.input_desc = None
        self.output_desc = None

        # self.num_iter = 0
        # self.starts = [torch.cuda.Event(enable_timing=True) for _ in range(128)]
        # self.ends = [torch.cuda.Event(enable_timing=True) for _ in range(128)]


    def reset_parameters(self) -> None:
        """
        Resets parameters based on their initialization used in ``__init__``.
        """
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def move_weight_to_device(self, device):
        if device=="cpu":
            weight_h = torch.empty(self.in_features, self.out_features, dtype=torch.float16, device='cpu', pin_memory=True).contiguous()
        else:
            weight_h = torch.empty(self.in_features, self.out_features, dtype=torch.float16, device='cuda')
        weight_h.copy_((self.weight.data.T), non_blocking=False)
        del self.weight
        self.weight = None
        self.weight = Parameter(weight_h)

        # print(f"\t {self.weight_h.shape} {self.weight_h.device} {self.weight_d.shape} {self.weight_d.device}")
        # print(f"\t out_features: {self.out_features}, out_features_h: {self.out_features_h}, out_features_d: {self.out_features_d}, num_copy_blocks_host: {num_host_blocks}, offload ratio: {self.out_features_h / self.out_features * 100:.2f}%")



    def config(self, num_blocks: int, sms_per_row: int):
        self.num_copy_blocks = num_blocks
        self.sms_per_row = sms_per_row

    def build_tma_desc(self):
        if self.w_desc is None:
            # _, self.w_desc = build_tma_wgmma_k(self.weight, TILE_K, TILE_M)
            _, self.w_desc = build_tma_wgmma_mn(self.weight, TILE_M, TILE_K)

    def forward(self, input: Tensor) -> Tensor:
        """
        Runs the forward pass.
        """
        # self.starts[self.num_iter].record()
        self.build_tma_desc()

        # Reuse input tensormap if shape matches; otherwise rebuild.
        if self.input_desc is None or self.input_shape != input.shape:
            self.input_shape = input.shape
            _, self.input_desc = build_tma_wgmma_k(input.view(-1, self.in_features), TILE_K, TILE_N)
        else:
            runtime.tensormap_replace_address(self.input_desc, input)
        # _, tma_desc_x = build_tma_wgmma_k(input.view(-1, self.in_features), TILE_K, TILE_N)

        output = torch.zeros(*input.shape[:-1], self.out_features, device=input.device, dtype=input.dtype).contiguous()
        expected_output_shape = input.shape[:-1] + (self.out_features,)
        if self.output_desc is None or self.output_shape != expected_output_shape:
            self.output_shape = output.shape
            _, self.output_desc = build_tma_wgmma_mn(output.view(-1, self.out_features), TILE_M, TILE_N)
        else:
            runtime.tensormap_replace_address(self.output_desc, output)
        # _, tma_desc_y = build_tma_wgmma_mn(y.view(-1, self.out_features), TILE_M, TILE_N)

        # torch.cuda.nvtx.range_push("GxEMV")
        # runtime.gemv(self.w_desc, tma_desc_x, tma_desc_y, self.out_features, input.shape[:-1].numel(), self.in_features, self.num_copy_blocks, self.smem_size, self.sms_per_row)
        runtime.gemv(self.w_desc, self.input_desc, self.output_desc, self.out_features, input.shape[:-1].numel(), self.in_features, self.num_copy_blocks, self.smem_size, self.sms_per_row)

        # torch.cuda.nvtx.range_pop()

        # self.ends[self.num_iter].record()
        # self.num_iter += 1

        if self.bias is not None:
            output += self.bias
        return output
    


    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
    

class LinearHorizontal(Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        smem_size: int,
        num_copy_blocks: int,
        bias: bool = False,
        device=None,
        dtype=None,
        skip_init: bool = False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.smem_size = smem_size
        self.num_copy_blocks = num_copy_blocks
        self.h_sms_per_row = 1
        self.d_sms_per_row = 1
        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        if not skip_init:
            self.reset_parameters()
        self.w_desc_h = None
        self.w_desc_d = None

        self.output = None
        self.output_shape = None
        self.input_desc = None
        self.output_desc = None

        # self.num_iter = 0
        # self.starts = [torch.cuda.Event(enable_timing=True) for _ in range(128)]
        # self.ends = [torch.cuda.Event(enable_timing=True) for _ in range(128)]

    # def horizontal_split_weight(self, host_out_features: int, num_host_blocks: int) -> None:
    #     """
    #     Horizontally split the weight matrix into two parts: [0:host_out_features, :] and [host_out_features:,].
    #     The first part is kept on host memory for CPU computation, and the second part is kept on device memory for GPU computation.
    #     """
    #     self.weight_h = self.weight.data[:host_out_features, :].cpu().contiguous().pin_memory()
    #     self.weight_d = self.weight.data[host_out_features:, :]
    #     self.out_features_h = host_out_features
    #     self.out_features_d = self.out_features - host_out_features
    #     self.num_copy_blocks_host = num_host_blocks
    #     # release the original weight to save memory, since we have split it into two parts
    #     del self.weight
    #     self.weight = None
    #     # torch.cuda.empty_cache()

    # def set_blocks(self, num_copy_blocks: int, num_copy_blocks_host: int):
    #     self.num_copy_blocks = num_copy_blocks
    #     self.num_copy_blocks_host = num_copy_blocks_host

    def horizontal_split_weight(self, offload: float, num_blocks: int, num_host_blocks: int, h_sms_per_row: int, d_sms_per_row: int, h_m: int = 0, d_m: int = 0) -> None:
        """
        Horizontally split the weight matrix into two parts: [0:host_out_features, :] and [host_out_features:,].
        The first part is kept on host memory for CPU computation, and the second part is kept on device memory for GPU computation.
        """
        self.num_copy_blocks = num_blocks
        self.num_copy_blocks_host = num_host_blocks
        self.h_sms_per_row = h_sms_per_row
        self.d_sms_per_row = d_sms_per_row

        if h_m == 0 and d_m == 0:
            self.out_features_h, self.out_features_d = split_horizontal((self.out_features, self.in_features), offload)
        else:
            self.out_features_h = h_m
            self.out_features_d = d_m

        # weight_h = (self.weight.data[:self.out_features_h, :].T).cpu().contiguous().pin_memory()
        # weight_d = self.weight.data[self.out_features_h:, :].T.contiguous()

        weight_h = torch.empty(self.in_features, self.out_features_h, dtype=torch.float16, device='cpu', pin_memory=True).contiguous()
        weight_d = torch.empty(self.in_features, self.out_features_d, dtype=torch.float16, device='cuda')
        weight_h.copy_((self.weight.data[:self.out_features_h, :].T), non_blocking=False)
        weight_d.copy_((self.weight.data[self.out_features_h:, :].T), non_blocking=False)


        self.weight_h = Parameter(weight_h)
        self.weight_d = Parameter(weight_d)

        # print(f"\t {self.weight_h.shape} {self.weight_h.device} {self.weight_d.shape} {self.weight_d.device}")
        # print(f"\t out_features: {self.out_features}, out_features_h: {self.out_features_h}, out_features_d: {self.out_features_d}, num_copy_blocks_host: {num_host_blocks}, offload ratio: {self.out_features_h / self.out_features * 100:.2f}%")

        # release the original weight to save memory, since we have split it into two parts
        del self.weight
        self.weight = None
        # torch.cuda.empty_cache()
        # pass

        # self.register_parameter("weight_h", weight_h)
        # self.register_parameter("weight_d", weight_d)


    def reset_parameters(self) -> None:
        """
        Resets parameters based on their initialization used in ``__init__``.
        """
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def build_tma_desc(self):
        if self.w_desc_h is None:
            # _, self.w_desc_h = build_tma_wgmma_k(self.weight_h, TILE_K, TILE_M)
            _, self.w_desc_h = build_tma_wgmma_mn(self.weight_h, TILE_M, TILE_K)
        if self.w_desc_d is None:
            # _, self.w_desc_d = build_tma_wgmma_k(self.weight_d, TILE_K, TILE_M)
            _, self.w_desc_d = build_tma_wgmma_mn(self.weight_d, TILE_M, TILE_K)

    def forward_naieve(self, input: Tensor) -> Tensor:
        """
        Runs the forward pass.
        """
        # self.starts[self.num_iter].record()

        # self.build_tma_desc()
        _, tma_desc_x = build_tma_wgmma_k(input.view(-1, self.in_features), TILE_K, TILE_N)

        y = torch.zeros(*input.shape[:-1], self.out_features, device=input.device, dtype=input.dtype).contiguous()
        _, tma_desc_y = build_tma_wgmma_mn(y.view(-1, self.out_features), TILE_M, TILE_N)


        # torch.cuda.nvtx.range_push("Horizontal GEMV")
        runtime.gemv_horizontal(self.w_desc_h, self.w_desc_d, tma_desc_x, tma_desc_y, self.out_features_h, self.out_features_d, input.shape[:-1].numel(), self.in_features, self.num_copy_blocks, self.num_copy_blocks_host, self.smem_size, self.h_sms_per_row, self.d_sms_per_row)
        # torch.cuda.nvtx.range_pop()

        # self.ends[self.num_iter].record()
        # self.num_iter += 1

        if self.bias is not None:
            y += self.bias
        return y

    def forward(self, input: Tensor) -> Tensor:
        """
        Runs the forward pass, reusing the built TMA descriptors and output buffer when possible.
        """
        # self.starts[self.num_iter].record()

        # Build weight descriptors once and reuse them across calls.
        self.build_tma_desc()

        # Reuse input tensormap if shape matches; otherwise rebuild.
        if self.input_desc is None or self.input_shape != input.shape:
            self.input_shape = input.shape
            _, self.input_desc = build_tma_wgmma_k(input.view(-1, self.in_features), TILE_K, TILE_N)
        else:
            runtime.tensormap_replace_address(self.input_desc, input)

        # Reuse output buffer and tensormap when the leading shape is unchanged.
        expected_output_shape = input.shape[:-1] + (self.out_features,)
        output = torch.zeros(*expected_output_shape, device=input.device, dtype=input.dtype).contiguous()
        if self.output_desc is None or self.output_shape != expected_output_shape:
            # self.output = torch.zeros(*expected_output_shape, device=input.device, dtype=input.dtype).contiguous()
            self.output_shape = output.shape
            _, self.output_desc = build_tma_wgmma_mn(output.view(-1, self.out_features), TILE_M, TILE_N)
        else:
            runtime.tensormap_replace_address(self.output_desc, output)


        # torch.cuda.nvtx.range_push("Horizontal GEMV")
        runtime.gemv_horizontal(self.w_desc_h, self.w_desc_d, self.input_desc, self.output_desc, self.out_features_h, self.out_features_d, input.shape[:-1].numel(), self.in_features, self.num_copy_blocks, self.num_copy_blocks_host, self.smem_size, self.h_sms_per_row, self.d_sms_per_row)
        # torch.cuda.nvtx.range_pop()

        # print(f"host weight shape: {self.weight_h.shape}, device weight shape: {self.weight_d.shape}, num_copy_blocks: {self.num_copy_blocks}, num_copy_blocks_host: {self.num_copy_blocks_host}, smem_size: {self.smem_size}")
        # torch.cuda.synchronize()

        # self.ends[self.num_iter].record()
        # self.num_iter += 1

        if self.bias is not None:
            output += self.bias
        return output

    


    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
