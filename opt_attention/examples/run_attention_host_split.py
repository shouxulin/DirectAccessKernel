import argparse
import math
import os
import statistics

import torch
from transformers import OPTConfig, StaticCache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.opt.modeling_opt import OPTAttention, eager_attention_forward

import opt_attention


def _build_split_kv_with_static_cache(
    config: OPTConfig,
    key_full: torch.Tensor,
    value_full: torch.Tensor,
    host_size: int,
):
    seq_len = key_full.shape[2]
    gpu_batch = key_full.shape[0] - host_size

    if gpu_batch > 0:
        gpu_cache = StaticCache(config=config, max_cache_len=seq_len)
        key_gpu_in = key_full[host_size:].contiguous()
        value_gpu_in = value_full[host_size:].contiguous()
        key_gpu, value_gpu = gpu_cache.update(
            key_gpu_in,
            value_gpu_in,
            layer_idx=0,
            cache_kwargs={"cache_position": torch.arange(seq_len, device=key_gpu_in.device)},
        )
    else:
        key_gpu = key_full[host_size:].contiguous()
        value_gpu = value_full[host_size:].contiguous()

    if host_size > 0:
        host_cache = StaticCache(config=config, max_cache_len=seq_len)
        key_host_in = key_full[:host_size].contiguous().to("cpu", non_blocking=True).pin_memory()
        value_host_in = value_full[:host_size].contiguous().to("cpu", non_blocking=True).pin_memory()
        key_host, value_host = host_cache.update(
            key_host_in,
            value_host_in,
            layer_idx=0,
            cache_kwargs={"cache_position": torch.arange(seq_len, device=key_host_in.device)},
        )
        key_host = key_host.pin_memory()
        value_host = value_host.pin_memory()
    else:
        key_host = torch.empty((0, *key_full.shape[1:]), dtype=key_full.dtype, device="cpu").pin_memory()
        value_host = torch.empty((0, *value_full.shape[1:]), dtype=value_full.dtype, device="cpu").pin_memory()

    return key_gpu, value_gpu, key_host, value_host


def benchmark_with_bw(label: str, fn, kv_total_bytes: int, warmup: int, measure_iters: int):
    if warmup < 0 or measure_iters <= 0:
        raise ValueError("warmup must be >= 0 and measure_iters must be > 0")

    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        times_ms = []
        bw_gbps = []
        for _ in range(measure_iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            elapsed_ms = start.elapsed_time(end)
            times_ms.append(elapsed_ms)
            # GB/s where GB uses 1e9 bytes.
            bw_gbps.append(kv_total_bytes / (elapsed_ms * 1e-3) / 1e9)

    print(f"[{label}] warmup={warmup} measure_iters={measure_iters}")
    print(
        f"[{label}] latency_ms mean={statistics.mean(times_ms):.4f} "
        f"median={statistics.median(times_ms):.4f} min={min(times_ms):.4f}"
    )
    print(
        f"[{label}] kv_bw_gbps mean={statistics.mean(bw_gbps):.2f} "
        f"median={statistics.median(bw_gbps):.2f} max={max(bw_gbps):.2f}"
    )


def run_case(
    batch: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    host_size: int,
    dtype: torch.dtype,
    warmup: int,
    measure_iters: int,
):
    device = torch.device("cuda")
    torch.manual_seed(1234)

    config = OPTConfig(
        hidden_size=heads * head_dim,
        num_attention_heads=heads,
        num_hidden_layers=1,
    )
    config._attn_implementation = "vdcores_opt"

    module = OPTAttention(config, layer_idx=0).to(device=device, dtype=dtype)
    module.eval()

    query = torch.randn(batch, heads, 1, head_dim, device=device, dtype=dtype) / math.sqrt(head_dim)
    key_full = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    value_full = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)

    key_gpu, value_gpu, key_host, value_host = _build_split_kv_with_static_cache(
        config,
        key_full,
        value_full,
        host_size,
    )
    kv_total_bytes = (key_full.numel() + value_full.numel()) * key_full.element_size()

    vd_iface = ALL_ATTENTION_FUNCTIONS.get_interface("vdcores_opt", eager_attention_forward)
    sdpa_iface = ALL_ATTENTION_FUNCTIONS.get_interface("sdpa", eager_attention_forward)

    with torch.no_grad():
        got, _ = vd_iface(
            module,
            query,
            key_gpu,
            value_gpu,
            None,
            dropout=0.0,
            scaling=1.0,
            key_host=key_host,
            value_host=value_host,
            host_size=host_size,
        )
        ref, _ = eager_attention_forward(
            module,
            query,
            key_full,
            value_full,
            None,
            dropout=0.0,
            scaling=1.0,
        )
        torch.cuda.synchronize()

    diff = (got - ref).abs().float()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    ok = torch.allclose(got, ref, rtol=4e-2, atol=4e-2)

    print(
        f"case batch={batch} heads={heads} seq_len={seq_len} head_dim={head_dim} host_size={host_size} "
        f"dtype={dtype}"
    )
    print(f"gpu_kv_batch={key_gpu.shape[0]} host_kv_batch={key_host.shape[0]}")
    print(f"kv_total_bytes={kv_total_bytes} ({kv_total_bytes / (1024 ** 2):.2f} MiB)")
    print(f"max_err={max_err:.8f} mean_err={mean_err:.8f} allclose={ok}")

    benchmark_with_bw(
        "vdcores_opt_host_split",
        lambda: vd_iface(
            module,
            query,
            key_gpu,
            value_gpu,
            None,
            dropout=0.0,
            scaling=1.0,
            key_host=key_host,
            value_host=value_host,
            host_size=host_size,
        ),
        kv_total_bytes=kv_total_bytes,
        warmup=warmup,
        measure_iters=measure_iters,
    )
    benchmark_with_bw(
        "pytorch_sdpa",
        lambda: sdpa_iface(
            module,
            query,
            key_full,
            value_full,
            None,
            dropout=0.0,
            scaling=1.0,
        ),
        kv_total_bytes=kv_total_bytes,
        warmup=warmup,
        measure_iters=measure_iters,
    )

    return ok


def main():
    parser = argparse.ArgumentParser(description="Test opt_attention mixed host/gpu KV split")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--host-size", type=int, default=2)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--split-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--measure-iters", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this test")

    if args.host_size < 0 or args.host_size > args.batch:
        raise ValueError("host_size must be in [0, batch]")

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    opt_attention.register()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Keep split_size aligned with the extension's static-cache tile assumptions.
    os.environ["OPT_ATTENTION_SPLIT_SIZE"] = str(args.split_size)

    ok = run_case(
        batch=args.batch,
        heads=args.heads,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        host_size=args.host_size,
        dtype=dtype,
        warmup=args.warmup,
        measure_iters=args.measure_iters,
    )
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
