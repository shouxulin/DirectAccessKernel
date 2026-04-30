import os
import sys

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.cache_utils import StaticCache

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from app.python.linear_replacement import (
    LinearReplacementConfig,
    load_replaced_llama_for_causal_lm,
)
from app.python.placement import place_model


device = "cuda"
dtype = torch.float16


def _format_gb(num_bytes: float) -> str:
    return f"{num_bytes / 1e9:.2f} GB"


def _format_ratio(value: float) -> str:
    return f"{value * 100:.2f}%"


def _progress(iterable, *, desc: str):
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, unit="step", dynamic_ncols=True)


def get_test_inputs(prompt, prompt_len, num_prompts, tokenizer):
    prompts = [prompt] * num_prompts
    inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=prompt_len,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    return input_ids, attention_mask


def estimate_kv_cache_size_gb(cfg, batch_size: int, seq_len: int, dtype: torch.dtype) -> float:
    # Per-token KV elements per layer = 2 * num_kv_heads * head_dim (K and V).
    # For MHA, num_kv_heads == num_attention_heads, so this reduces to 2 * hidden_size.
    num_layers = int(getattr(cfg, "num_hidden_layers"))
    hidden_size = int(getattr(cfg, "hidden_size"))
    num_attention_heads = int(getattr(cfg, "num_attention_heads"))
    num_kv_heads = int(getattr(cfg, "num_key_value_heads", num_attention_heads))
    head_dim = hidden_size // num_attention_heads
    dtype_bytes = torch.tensor([], dtype=dtype).element_size()

    total_bytes = (
        2
        * batch_size
        * seq_len
        * num_layers
        * num_kv_heads
        * head_dim
        * dtype_bytes
    )
    return total_bytes / 1e9


def _make_cache(config: LlamaConfig, max_cache_len: int) -> StaticCache:
    return StaticCache(config=config, max_cache_len=max_cache_len)


horizontal_config = {}
horizontal_config["GH200"] = {}
bsz = 8
horizontal_config["GH200"].update(
    {
        # {h_m, d_m, h_blocks, d_blocks, h_sms_per_row, d_sms_per_row}
        (4096, 4096): (320, 4096 - 320, 10, 120, 2, 2),
        (4096, 11008): (320, 4096 - 320, 10, 120, 2, 2),
        (11008, 4096): (768, 11008 - 768, 12, 80, 1, 1),
        (32000, 4096): (2560, 32000 - 2560, 10, 115, 1, 1),
    }
)
default_horizontal_config = {
    "GH200": (0, 0, 24, 104, 1, 1),
    "RTX6000": (0, 0, 8, 180, 1, 1),
}


@torch.inference_mode()
def run(
    gpu_name: str,
    model_path: str,
    prompt: str,
    prompt_len: int,
    bsz: int,
    max_new_tokens: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    graph_capture: bool = False,
    offload_ratio: float = 0.0,
    use_config: bool = False,
    attn_impl: str | None = None,
    host_cache_size: int = 0,
    output_file: str = f"{ROOT_DIR}/eval/result/SplitKernel_mem.csv",
    drop_bias: bool = False,
):
    cfg = AutoConfig.from_pretrained(model_path)
    if attn_impl:
        if attn_impl == "vdcores_opt":
            print("[Setup] Registering attention implementation: vdcores_opt")
            import opt_attention

            opt_attention.register(name=attn_impl)
        cfg._attn_implementation = attn_impl
        print(f"[Setup] Attention implementation: {attn_impl}")

        # host_cache_size = 80
        # if attn_impl == "vdcores_opt" and host_cache_size == bsz and bsz > 0:
        #     # vdcores_opt currently expects a non-empty device-side partition.
        #     host_cache_size = 64
        #     print(
        #         f"[warn] vdcores_opt requires host_size < batch_size; "
        #         f"clamped to {host_cache_size}."
        #     )
        print(
            f"[Setup] Batch size: {bsz}, host cache size: {host_cache_size}, "
            f"target weight offload: {_format_ratio(offload_ratio)}"
        )

    # Load tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    input_ids, attention_mask = get_test_inputs(prompt, prompt_len, bsz, tokenizer)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    if offload_ratio > 0:
        from app.python.utils import get_smem_size
        from offload import runtime

        smem_size = get_smem_size()
        runtime.set_smem_size(smem_size, 0 if offload_ratio == 1.0 else 1)

    if offload_ratio == 0:
        from app.python.llama.vanila_llama import LlamaForCausalLM

        model = LlamaForCausalLM.from_pretrained(
            model_path,
            config=cfg,
            torch_dtype=dtype,
            device_map=None,
        ).to(device)
    else:
        linear_type = "host" if offload_ratio == 1.0 else "horizontal"
        model = load_replaced_llama_for_causal_lm(
            model_path,
            LinearReplacementConfig(
                linear_type,
                device="cpu",
                dtype=dtype,
                multicast_kernel=False,
                drop_bias=drop_bias,
            ),
            config=cfg,
            torch_dtype=dtype,
            device_map=None,
        )
    model.eval()

    if offload_ratio > 0:
        print(
            "[Setup] Placing replaced LLaMA linears with target offload: "
            f"{_format_ratio(offload_ratio)}"
        )
        placement_stats = place_model(
            model,
            offload_ratio,
            gpu_name=gpu_name,
            use_config=use_config,
            horizontal_config=horizontal_config,
            default_horizontal_config=default_horizontal_config,
            device=device,
        )
        model_size_h = placement_stats.host_bytes
        model_size_d = placement_stats.device_bytes
    else:
        model_size_h = 0
        model_size_d = sum(p.numel() * p.element_size() for p in model.parameters())

    model_size = (
        sum(p.numel() for p in model.parameters())
        * model.parameters().__next__().element_size()
    )
    assert model_size == model_size_h + model_size_d, (
        f"Model size mismatch: total {model_size} != "
        f"host {model_size_h} + device {model_size_d}"
    )
    print(f"[Setup] Model weight size: {_format_gb(model_size)}")

    B, L = input_ids.shape
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    prompt_lengths = attention_mask.sum(dim=1)

    # 1) Preallocate a static KV cache covering the prompt and generation.
    max_cache_len = L + max_new_tokens
    cache = StaticCache(
        config=model.config,
        batch_size=B,
        max_cache_len=max_cache_len,
        device=device,
        dtype=dtype,
    )

    host_cache = _make_cache(model.config, max_cache_len)

    kv_est_gb = estimate_kv_cache_size_gb(model.config, B, max_cache_len, dtype)
    print(
        f"[Setup] Estimated KV cache size: {kv_est_gb:.2f} GB "
        f"(B={B}, L={max_cache_len}, layers={model.config.num_hidden_layers}, "
        f"hidden={model.config.hidden_size}, dtype={dtype})"
    )

    out_ids = torch.empty((B, max_cache_len), device=device, dtype=input_ids.dtype)
    out_ids[:, :L] = input_ids  # Copy the prompt first.

    # 2) Prefill. With left padding, LLaMA RoPE positions should follow real
    # prompt tokens rather than padded storage positions.
    prefill_cache_pos = torch.arange(L, device=device, dtype=torch.long)
    prefill_position_ids = attention_mask.long().cumsum(dim=1) - 1
    prefill_position_ids.masked_fill_(attention_mask == 0, 0)
    prefill_out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=prefill_position_ids,
        use_cache=True,
        past_key_values=cache,
        past_key_values_host=host_cache,
        host_size=host_cache_size,
        cache_position=prefill_cache_pos,
        logits_to_keep=1,
    )

    logits = prefill_out.logits  # [B, 1, V]
    next_token = torch.argmax(logits[:, -1, :], dim=-1)  # [B]
    del prefill_out, logits
    out_ids[:, L] = next_token  # Store the first generated token from prefill.

    # vocab_size = model.config.vocab_size
    # fake_prefill_logits = torch.randn((B, 1, vocab_size), device=device, dtype=dtype)
    # prefill_out = SimpleNamespace(logits=fake_prefill_logits)
    # logits = prefill_out.logits  # [B, L, V]
    # next_token = torch.argmax(logits[:, -1, :], dim=-1)
    # del prefill_out, logits, fake_prefill_logits
    # out_ids[:, L] = next_token

    # 3) Prepare fixed-shape decode buffers. Each step uses [B, 1].
    cur_input_ids = torch.empty((B, 1), device=device, dtype=input_ids.dtype)
    cur_input_ids[:, 0] = next_token

    # The attention mask also has a fixed shape. Preallocate [B, max_cache_len]
    # and update one position to 1 at each step.
    attn_buf = torch.zeros((B, max_cache_len), device=device, dtype=attention_mask.dtype)
    attn_buf[:, :L] = attention_mask
    attn_buf[:, L] = 1

    # Keep position_ids fixed at [B, 1] and update in place.
    pos_buf = torch.empty((B, 1), device=device, dtype=torch.long)
    pos_buf[:, 0] = prompt_lengths

    g = torch.cuda.CUDAGraph()

    # 6) Decode loop: update buffers, replay or run, argmax, then update buffers.
    cur_pos = L
    cache_pos_buf = torch.empty((1,), device=device, dtype=torch.long)

    # Warm up so kernel selection, lazy init, and memory pool setup happen before capture.
    torch.cuda.synchronize()
    for _ in range(2):
        cache_pos_buf[0] = cur_pos
        out = model(
            input_ids=cur_input_ids,
            attention_mask=attn_buf,
            position_ids=pos_buf,
            use_cache=True,
            past_key_values=cache,
            cache_position=cache_pos_buf,
            past_key_values_host=host_cache,
            host_size=host_cache_size,
            logits_to_keep=1,
        )
    torch.cuda.synchronize()

    print("[Benchmark] Warmup done. Starting decode benchmark ...")

    # capture cuda graph
    if graph_capture:
        print("[Benchmark] Capturing CUDA graph ...")
        with torch.cuda.graph(g):
            out = model(
                input_ids=cur_input_ids,
                attention_mask=attn_buf,
                position_ids=pos_buf,
                use_cache=True,
                past_key_values=cache,
                cache_position=cache_pos_buf,
                past_key_values_host=host_cache,
                host_size=host_cache_size,
                logits_to_keep=1,
            )

    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(max_new_tokens - 1)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(max_new_tokens - 1)]

    decode_steps = max_new_tokens - 1
    for step in _progress(range(decode_steps), desc="[Benchmark] Decoding"):
        cache_pos_buf[0] = cur_pos
        start_events[step].record()
        if graph_capture:
            g.replay()
        else:
            out = model(
                input_ids=cur_input_ids,
                attention_mask=attn_buf,
                position_ids=pos_buf,
                use_cache=True,
                past_key_values=cache,
                cache_position=cache_pos_buf,
                past_key_values_host=host_cache,
                host_size=host_cache_size,
                logits_to_keep=1,
            )
        end_events[step].record()
        step_logits = out.logits  # [B, 1, V]

        # Use the final position from step_logits.
        next_token = torch.argmax(step_logits[:, -1, :], dim=-1)  # [B]

        # Update the input token buffer in place while preserving shape.
        cur_input_ids[:, 0] = next_token
        # generated.append(cur_input_ids)

        # Update the attention mask buffer by setting the next position to 1.
        cur_pos += 1
        if cur_pos >= max_cache_len:
            break
        attn_buf[:, cur_pos] = 1

        # Update the position id buffer.
        pos_buf[:, 0] += 1

        out_ids[:, cur_pos] = next_token  # Store the generated token.

    torch.cuda.synchronize()
    out_tokens = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
    print("\n[Samples] Generated text:")
    for i in [0, len(out_tokens) - 1]:
        print(f"{i}: {out_tokens[i]}")
        print("-" * 70)

    avg_latency = []
    for step in range(max_new_tokens - 1):
        latency = start_events[step].elapsed_time(end_events[step])
        # print(f"Step {step+1} latency: {latency:.2f} ms")
        avg_latency.append(latency)
    mean_latency = np.mean(avg_latency)
    print(f"[Metrics] TBT: {mean_latency:.2f} ms")

    decode_throughput = (B * (max_new_tokens - 1)) / (np.sum(avg_latency) / 1000)
    assert (decode_throughput - 1 / (mean_latency / 1000) * B) < 1e-5, (
        f"Throughput calculation mismatch: {decode_throughput} vs "
        f"{1 / (mean_latency / 1000) * B}"
    )
    print(f"[Metrics] Decode throughput: {decode_throughput:.2f} tokens/s")

    effective_bw = model_size / (mean_latency / 1000) / 1e9
    print(f"[Metrics] Effective memory bandwidth: {effective_bw:.2f} GB/s")

    actual_ratio = model_size_h / model_size

    cache_ratio = float(host_cache_size) / float(bsz)
    kv_est_bytes = kv_est_gb * 1e9
    host_cache_bytes = cache_ratio * kv_est_bytes
    device_cache_bytes = kv_est_bytes - host_cache_bytes
    overal_ratio = (model_size_h + host_cache_bytes) / (model_size + kv_est_bytes)
    overall_total_bytes = model_size + kv_est_bytes
    overall_host_bytes = model_size_h + host_cache_bytes
    overall_device_bytes = model_size_d + device_cache_bytes

    print("[Offload] Weight:")
    print(
        f"  ratio: {_format_ratio(actual_ratio)}, total: {_format_gb(model_size)}, "
        f"host: {_format_gb(model_size_h)}, gpu: {_format_gb(model_size_d)}"
    )
    print("[Offload] KV cache:")
    print(
        f"  ratio: {_format_ratio(cache_ratio)}, total: {_format_gb(kv_est_bytes)}, "
        f"host: {_format_gb(host_cache_bytes)}, gpu: {_format_gb(device_cache_bytes)}"
    )
    print("[Offload] Overall:")
    print(
        f"  ratio: {_format_ratio(overal_ratio)}, total: {_format_gb(overall_total_bytes)}, "
        f"host: {_format_gb(overall_host_bytes)}, gpu: {_format_gb(overall_device_bytes)}"
    )


    # write to outputfile
    with open(output_file, "a") as f:
        if os.stat(output_file).st_size == 0:
            f.write("gpu,model,prompt_len,bsz,offload_ratio,actual_offloading_ratio,weight_ratio,cache_ratio,tpot,decode_throughput,bw\n")
        f.write(f"{gpu_name},{model_path.split('/')[-1]},{prompt_len},{bsz},{offload_ratio*100:.2f},{overal_ratio*100:.2f},{actual_ratio*100:.2f},{cache_ratio*100:.2f},{mean_latency:.2f},{decode_throughput:.2f},{effective_bw:.2f}\n")




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare different attention implementations")
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument(
        "--input_text",
        type=str,
        default="Paris is the capital city of",
        help="Input text",
    )
    parser.add_argument("--prompt_len", type=int, default=2, help="Prompt length")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--host_cache_size",
        type=int,
        default=0,
        help="Host cache size",
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
    parser.add_argument("--gpu", type=str, default="GH200", help="GPU name")
    parser.add_argument(
        "--use_config",
        action="store_true",
        default=False,
        help="Whether to use predefined config.",
    )
    parser.add_argument(
        "--attn_impl",
        type=str,
        default=None,
        help="Attention implementation, e.g. sdpa/eager/vdcores_opt",
    )
    parser.add_argument(
        "--drop_bias",
        action="store_true",
        default=False,
        help="Drop linear bias in replaced custom linears.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=f"{ROOT_DIR}/benchmark/SplitKernel_mem.csv",
        help="Path to append benchmark CSV results.",
    )
    args = parser.parse_args()

    run(
        args.gpu,
        args.model_path,
        args.input_text,
        args.prompt_len,
        args.batch_size,
        args.max_new_tokens,
        device="cuda",
        dtype=torch.float16,
        graph_capture=args.graph,
        offload_ratio=args.offload,
        use_config=args.use_config,
        attn_impl=args.attn_impl,
        host_cache_size=args.host_cache_size,
        output_file=args.output_file,
        drop_bias=args.drop_bias,
    )
