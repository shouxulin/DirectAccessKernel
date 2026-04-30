from __future__ import annotations

import os

import torch


def _valid_cache_layout(tensor: torch.Tensor) -> bool:
    if tensor.dim() != 4:
        return False
    if tensor.stride(-1) != 1:
        return False
    if tensor.stride(-2) != tensor.shape[-1]:
        return False
    if tensor.stride(-3) != tensor.shape[-2] * tensor.shape[-1]:
        return False
    if tensor.stride(-4) != tensor.shape[-3] * tensor.shape[-2] * tensor.shape[-1]:
        return False
    return True


def _pop_host_kv_kwargs(kwargs: dict) -> tuple[torch.Tensor | None, torch.Tensor | None, int]:
    key_host = kwargs.pop("key_host", None)
    value_host = kwargs.pop("value_host", None)
    host_size = int(kwargs.pop("host_size", 0))
    return key_host, value_host, host_size


def _fallback_attention(
    module,
    query,
    key,
    value,
    attention_mask,
    dropout,
    scaling,
    key_host=None,
    value_host=None,
    host_size=0,
    **kwargs,
):
    if host_size > 0:
        if key_host is None or value_host is None:
            raise ValueError("key_host/value_host are required when host_size > 0")

        # Fallback expects full [B, H, S, D] tensors matching query batch.
        # Rebuild full KV by prepending host split and appending GPU split.
        key_host_full = key_host.to(device=query.device, dtype=key.dtype, non_blocking=True)
        value_host_full = value_host.to(device=query.device, dtype=value.dtype, non_blocking=True)
        if key.shape[0] == 0:
            key = key_host_full
            value = value_host_full
        else:
            key = torch.cat((key_host_full, key), dim=0)
            value = torch.cat((value_host_full, value), dim=0)

    from transformers.models.opt.modeling_opt import eager_attention_forward

    return eager_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=dropout,
        scaling=scaling,
        **kwargs,
    )


def _supported(module, query, key, value, attention_mask, dropout, key_host, value_host, host_size, **kwargs) -> bool:
    if kwargs.get("output_attentions", False):
        return False
    if dropout != 0.0:
        return False
    if getattr(getattr(module, "config", None), "model_type", None) != "opt":
        return False
    if host_size < 0 or host_size > query.shape[0]:
        return False
    if not (query.is_cuda and key.is_cuda and value.is_cuda):
        return False
    if query.dtype not in (torch.float16, torch.bfloat16):
        return False
    if key.dtype != query.dtype or value.dtype != query.dtype:
        return False
    if query.dim() != 4:
        return False
    if query.shape[2] != 1 or query.shape[-1] != 128:
        return False
    if key.shape != value.shape:
        return False
    if key.shape[0] != query.shape[0] - host_size or key.shape[1] != query.shape[1] or key.shape[-1] != query.shape[-1]:
        return False
    gpu_seq_len = int(key.shape[2]) if int(key.shape[0]) > 0 else 0
    if gpu_seq_len > 0 and gpu_seq_len % 64 != 0:
        return False
    if query.stride(-1) != 1:
        return False
    if not _valid_cache_layout(key) or not _valid_cache_layout(value):
        return False
    effective_seq_len = gpu_seq_len
    if host_size > 0:
        if key_host is None or value_host is None:
            return False
        if not isinstance(key_host, torch.Tensor) or not isinstance(value_host, torch.Tensor):
            return False
        if key_host.is_cuda or value_host.is_cuda:
            return False
        if key_host.dtype != query.dtype or value_host.dtype != query.dtype:
            return False
        if key_host.shape != value_host.shape:
            return False
        if key_host.dim() != 4:
            return False
        if key_host.shape[0] != host_size or key_host.shape[1] != query.shape[1]:
            return False
        if key_host.shape[-1] != query.shape[-1]:
            return False
        if not _valid_cache_layout(key_host) or not _valid_cache_layout(value_host):
            return False
        host_seq_len = int(key_host.shape[2])
        if host_seq_len % 64 != 0:
            return False
        if gpu_seq_len == 0:
            effective_seq_len = host_seq_len
        elif host_seq_len != gpu_seq_len:
            return False
    if effective_seq_len == 0:
        return False
    if int(os.environ.get("OPT_ATTENTION_SPLIT_SIZE", "256")) % 64 != 0:
        return False
    if attention_mask is not None:
        if not attention_mask.is_cuda or attention_mask.dim() != 4:
            return False
        if attention_mask.shape[0] != query.shape[0] or attention_mask.shape[-1] != effective_seq_len:
            return False
        if attention_mask.shape[1] != 1 or attention_mask.shape[2] != 1:
            return False
    return True


def opt_attention_forward(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    runtime_kwargs = dict(kwargs)
    key_host, value_host, host_size = _pop_host_kv_kwargs(runtime_kwargs)
    if host_size > 0 and (key_host is None or value_host is None):
        raise ValueError("host_size > 0 requires key_host and value_host")
    scale = 1.0 if scaling is None else float(scaling)
    if not _supported(
        module,
        query,
        key,
        value,
        attention_mask,
        float(dropout),
        key_host,
        value_host,
        host_size,
        **runtime_kwargs,
    ):
        return _fallback_attention(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout,
            scale,
            key_host=key_host,
            value_host=value_host,
            host_size=host_size,
            **runtime_kwargs,
        )

    from . import _C

    mask = attention_mask
    if mask is not None and mask.dtype is not torch.float32:
        mask = mask.to(torch.float32)
    split_size = int(os.environ.get("OPT_ATTENTION_SPLIT_SIZE", "256"))
    return _C.decode(query, key, value, key_host, value_host, host_size, mask, scale, split_size), None


def register(name: str = "vdcores_opt") -> None:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS.register(name, opt_attention_forward)
    ALL_MASK_ATTENTION_FUNCTIONS.register(name, ALL_MASK_ATTENTION_FUNCTIONS["eager"])
