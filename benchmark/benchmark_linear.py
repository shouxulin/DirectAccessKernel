import argparse
import os
import random
import sys

import numpy as np
import torch
from torch import nn

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.python.config import (
    TILE_K,
    TILE_K_MULTICAST,
    TILE_M,
    TILE_M_MULTICAST,
    TILE_N,
    TILE_N_MULTICAST,
)
from app.python.utils import (
    build_tma_wgmma_k,
    build_tma_wgmma_mn,
    get_smem_size,
    split_horizontal,
)
from offload import runtime

device = "cuda"
dtype = torch.float16


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark different linear implementations")
    parser.add_argument(
        "--type",
        type=str,
        choices=["linear", "horizontal", "vanila", "multicast", "horizontal_multicast"],
        default="vanila",
        help="Benchmark type: linear, horizontal, vanila, multicast, or horizontal_multicast",
    )
    parser.add_argument("--m", type=int, default=4096, help="Weight M dimension")
    parser.add_argument("--h_m", type=int, default=0, help="Host M dimension")
    parser.add_argument("--k", type=int, default=4096, help="Weight K dimension")
    parser.add_argument("--n", type=int, default=8, help="N")
    parser.add_argument(
        "--h_blocks",
        type=int,
        default=16,
        help="number of sms reading from host",
    )
    parser.add_argument(
        "--d_blocks",
        type=int,
        default=16,
        help="number of sms reading from gpu",
    )
    parser.add_argument(
        "--h_sms_per_row",
        type=int,
        default=1,
        help="number of sms per row for horizontal kernel",
    )
    parser.add_argument(
        "--d_sms_per_row",
        type=int,
        default=1,
        help="number of sms per row for diagonal kernel",
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        default=False,
        help="Whether to use cuda graph.",
    )
    parser.add_argument(
        "--offload",
        type=float,
        default=0.0,
        help=(
            "Offload ratio for weights. If > 0, it will override the weight "
            "percentage specified by --percent."
        ),
    )
    args = parser.parse_args()

    m = args.m
    k = args.k
    seq_len = 1
    bsz = args.n

    num_blocks = args.h_blocks + args.d_blocks
    num_blocks_host = args.h_blocks
    assert (
        num_blocks_host <= num_blocks
    ), "num_blocks_host should be less than or equal to num_blocks"
    offload = args.offload

    if (args.type == "horizontal" or args.type == "horizontal_multicast") and args.offload > 0:
        if args.h_m == 0:
            m_h, m_d = split_horizontal((m, k), offload)
        else:
            m_h = args.h_m
            m_d = m - m_h
        offload_ratio = m_h / m

        print(
            f"num_blocks: {num_blocks}, num_blocks_host: {num_blocks_host}, "
            f"M: {m}, h_M: {m_h}, d_M: {m_d}, k: {k} "
            f"offload ratio: {m_h / m * 100:.2f}%"
        )

    num_warmup = 5
    num_measure = 10

    # torch.manual_seed(42)

    """ verify correctness """

    input = torch.randn(bsz, seq_len, k, device=device, dtype=dtype).contiguous()
    linear_ref = nn.Linear(k, m, bias=False, device=device, dtype=dtype)
    output_ref = linear_ref(input)

    if args.type == "vanila":
        linear = nn.Linear(k, m, bias=False, device=device, dtype=dtype)
        linear.weight.data.copy_(linear_ref.weight.data)
    elif args.type == "linear":
        from app.python.layers.linearLayer import Linear

        smem_size = get_smem_size()
        smem_size = runtime.set_smem_size(smem_size, 0)
        linear = Linear(
            k,
            m,
            smem_size=smem_size,
            num_copy_blocks=num_blocks,
            device=device,
            dtype=dtype,
        )
        linear.weight.data.copy_(linear_ref.weight.data)
        linear.config(num_blocks, args.d_sms_per_row)
        print(f"Offloading weight to {'CPU' if offload > 0 else 'GPU'}...")
        linear.move_weight_to_device("cpu" if offload > 0 else "cuda")
        linear.build_tma_desc()
    elif args.type == "multicast":
        assert (
            args.offload == 0 or args.offload == 1
        ), "Offload ratio should be either 0 or 1 for multicast kernel"
        from app.python.layers.linearLayerMulticast import LinearMulticast

        smem_size = get_smem_size(multicast=True)
        smem_size = runtime.set_smem_size(smem_size, 2)
        smem_size = runtime.set_smem_size(smem_size, 4)
        linear = LinearMulticast(
            k,
            m,
            smem_size=smem_size,
            num_copy_blocks=num_blocks,
            device=device,
            dtype=dtype,
        )
        linear.weight.data.copy_(linear_ref.weight.data)
        linear.config(num_blocks, args.d_sms_per_row)
        target_device = "cpu" if offload == 1 else "cuda"
        print(f"Offloading weight to {target_device.upper()}")
        linear.move_weight_to_device(target_device)
        linear.build_tma_desc()
    elif args.type == "horizontal_multicast":
        from app.python.layers.linearLayerMulticast import LinearHorizontalMulticast

        smem_size = get_smem_size(multicast=True)
        smem_size = runtime.set_smem_size(smem_size, 3)
        smem_size = runtime.set_smem_size(smem_size, 5)
        linear = LinearHorizontalMulticast(
            k,
            m,
            smem_size=smem_size,
            num_copy_blocks=num_blocks,
            device=device,
            dtype=dtype,
        )
        linear.weight.data.copy_(linear_ref.weight.data)
        linear.horizontal_split_weight(
            offload,
            num_blocks,
            num_blocks_host,
            args.h_sms_per_row,
            args.d_sms_per_row,
            m_h,
            m_d,
        )

        # horizontal_output = torch.zeros(bsz, seq_len, m, device=device, dtype=dtype).contiguous()
        # _, tma_desc_y = build_tma_wgmma_mn(
        #     horizontal_output.view(-1, m), TILE_M_MULTICAST, TILE_N_MULTICAST, debug=True
        # )

    elif args.type == "horizontal":
        from app.python.layers.linearLayer import LinearHorizontal

        smem_size = get_smem_size()
        smem_size = runtime.set_smem_size(smem_size, 1)
        linear = LinearHorizontal(
            k,
            m,
            smem_size=smem_size,
            num_copy_blocks=num_blocks,
            device=device,
            dtype=dtype,
        )
        linear.weight.data.copy_(linear_ref.weight.data)
        linear.horizontal_split_weight(
            offload,
            num_blocks,
            num_blocks_host,
            args.h_sms_per_row,
            args.d_sms_per_row,
            m_h,
            m_d,
        )
        linear.build_tma_desc()

        _, tma_desc_x = build_tma_wgmma_k(input.view(-1, k), TILE_K, TILE_N)
        horizontal_output = torch.zeros(bsz, seq_len, m, device=device, dtype=dtype).contiguous()
        _, tma_desc_y = build_tma_wgmma_mn(horizontal_output.view(-1, m), TILE_M, TILE_N)
    else:
        raise ValueError("Invalid type")

    torch.cuda.synchronize()

    def run():
        if args.type == "linear" or args.type == "vanila" or args.type == "multicast":
            return linear.forward(input)
        elif args.type == "horizontal":
            # runtime.gemv_horizontal(
            #     linear.w_desc_h,
            #     linear.w_desc_d,
            #     tma_desc_x,
            #     tma_desc_y,
            #     linear.out_features_h,
            #     linear.out_features_d,
            #     bsz * seq_len,
            #     linear.in_features,
            #     linear.num_copy_blocks,
            #     linear.num_copy_blocks_host,
            #     linear.smem_size,
            #     args.h_sms_per_row,
            #     args.d_sms_per_row,
            # )
            # return horizontal_output
            return linear.forward(input)
            # return linear.forward_naieve(input)
        elif args.type == "horizontal_multicast":
            return linear.forward(input)
        else:
            raise ValueError("Invalid type")

    # output = linear.forward(input)
    output = run()
    # print(f"output: {output.shape} {output}")
    torch.cuda.synchronize()

    eps = 1e-6
    mape = (torch.abs(output - output_ref) / (torch.abs(output_ref) + eps)).mean() * 100
    print(f"MAPE %: {mape.item():.2f}")
    # print(f"output_ref: {output_ref.shape} {output_ref}")
    # print(f"output: {output.shape} {output}")

    ref_starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_measure)]
    ref_ends = [torch.cuda.Event(enable_timing=True) for _ in range(num_measure)]

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_measure)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(num_measure)]

    dummy = torch.randn(100 * 1024 * 1024, dtype=torch.float32, device="cuda")
    sink = torch.zeros(1, dtype=torch.float32, device="cuda")

    matrix_size = m * k * input.element_size()
    intput_size = bsz * seq_len * k * input.element_size()
    output_size = bsz * seq_len * m * input.element_size()
    print(
        f"matrix size: {matrix_size / (1024**2):.2f} MB "
        f"input size: {intput_size / (1024**2):.2f} MB "
        f"output size: {output_size / (1024**2):.2f} MB"
    )

    weight_bytes = matrix_size

    # """ vanila """
    #  warm up
    # for i in range(num_warmup):
    #     output_ref= linear_ref(input)
    # torch.cuda.synchronize()

    # with torch.no_grad():
    #     for i in range(num_measure):
    #         runtime.flush_l2_cache(dummy, sink)
    #         ref_starts[i].record()
    #         ref_output = linear_ref(input)
    #         ref_ends[i].record()
    #     torch.cuda.synchronize()
    # avg_ref_time = (
    #     sum([ref_starts[i].elapsed_time(ref_ends[i]) for i in range(num_measure)])
    #     / num_measure
    # )
    # ref_bandwith = weight_bytes / (avg_ref_time / 1000) / (1e9)
    # print("="*60)
    # print(f"Reference matmul timing")
    # print(
    #     f"Average reference time: {avg_ref_time*1000:.2f} us, "
    #     f"Effective Bandwidth: {ref_bandwith:.2f} GB/s"
    # )

    with torch.no_grad():
        # warmup
        for i in range(num_warmup):
            # output = linear.forward(input)
            output = run()

        # cuda graph
        if args.graph:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                # output = linear.forward(input)
                output = run()

        torch.cuda.synchronize()

        for i in range(num_measure):
            runtime.flush_l2_cache(dummy, sink)
            starts[i].record()
            if not args.graph:
                # output = linear.forward(input)
                output = run()
            else:
                g.replay()
            ends[i].record()
        torch.cuda.synchronize()
        avg_time = sum([starts[i].elapsed_time(ends[i]) for i in range(num_measure)]) / num_measure
        bandwith = weight_bytes / (avg_time / 1000) / (1e9)
        print("=" * 60)
        print(f"{args.type} timing")
        print(f"Average time: {avg_time*1000:.2f} us, Effective Bandwidth: {bandwith:.2f} GB/s")

    if args.type == "horizontal":
        if offload_ratio < 0.08:
            theoretical_bw = 4000 * 0.8 / (1 - offload_ratio)
        else:
            theoretical_bw = 450 * 0.8 / offload_ratio
        print(
            f"Split ratio: {offload_ratio * 100:.2f}%, "
            f"Theoretical Bandwidth: {theoretical_bw:.2f} GB/s "
            f"diff: {abs(bandwith - theoretical_bw) / theoretical_bw * 100:.2f}%"
        )
