from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


LinearType = Literal["vanila", "host", "horizontal", "multicast", "horizontal_multicast"]

OPT_TARGET_LINEARS = ("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2")
OPT_PROJECTION_LINEARS = ("project_in", "project_out")
LLAMA_ATTENTION_LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj")
LLAMA_MLP_LINEARS = ("gate_proj", "up_proj", "down_proj")
LLAMA_HEAD_LINEARS = ("lm_head",)
LLAMA_TARGET_LINEARS = LLAMA_ATTENTION_LINEARS + LLAMA_MLP_LINEARS + LLAMA_HEAD_LINEARS


@dataclass(frozen=True)
class LinearReplacementConfig:
    linear_type: LinearType
    smem_size: int | None = None
    num_copy_blocks: int = 128
    device: torch.device | str | None = None
    dtype: torch.dtype | None = torch.float16
    multicast_kernel: bool | None = None
    include_embedding_projections: bool = False
    drop_bias: bool = True
    skip_init: bool = True


def set_opt_linear_config(config, replacement: LinearType | LinearReplacementConfig):
    return _set_linear_config(config, "opt", replacement)


def set_llama_linear_config(config, replacement: LinearType | LinearReplacementConfig):
    return _set_linear_config(config, "llama", replacement)


def _set_linear_config(config, prefix: str, replacement: LinearType | LinearReplacementConfig):
    replacement = (
        replacement if isinstance(replacement, LinearReplacementConfig) else LinearReplacementConfig(replacement)
    )
    setattr(config, f"{prefix}_linear_type", replacement.linear_type)
    setattr(config, f"{prefix}_linear_smem_size", replacement.smem_size)
    setattr(config, f"{prefix}_linear_num_copy_blocks", replacement.num_copy_blocks)
    setattr(config, f"{prefix}_linear_device", str(replacement.device) if replacement.device is not None else None)
    setattr(config, f"{prefix}_linear_dtype", _dtype_to_name(replacement.dtype))
    setattr(config, f"{prefix}_linear_multicast_kernel", replacement.multicast_kernel)
    setattr(config, f"{prefix}_linear_drop_bias", replacement.drop_bias)
    setattr(config, f"{prefix}_linear_skip_init", replacement.skip_init)
    return config


def get_opt_linear_config(config) -> LinearReplacementConfig:
    return _get_linear_config(config, "opt")


def get_llama_linear_config(config) -> LinearReplacementConfig:
    return _get_linear_config(config, "llama")


def _get_linear_config(config, prefix: str) -> LinearReplacementConfig:
    return LinearReplacementConfig(
        linear_type=getattr(config, f"{prefix}_linear_type", "vanila"),
        smem_size=getattr(config, f"{prefix}_linear_smem_size", None),
        num_copy_blocks=getattr(config, f"{prefix}_linear_num_copy_blocks", 128),
        device=getattr(config, f"{prefix}_linear_device", None),
        dtype=_name_to_dtype(getattr(config, f"{prefix}_linear_dtype", "float16")),
        multicast_kernel=getattr(config, f"{prefix}_linear_multicast_kernel", None),
        drop_bias=getattr(config, f"{prefix}_linear_drop_bias", True),
        skip_init=getattr(config, f"{prefix}_linear_skip_init", True),
    )


def make_replacement_linear(
    in_features: int,
    out_features: int,
    bias: bool,
    config: LinearReplacementConfig,
) -> nn.Module:
    linear_cls = _resolve_linear_class(config.linear_type)
    return linear_cls(in_features, out_features, **_linear_kwargs(config, bias))


def _resolve_linear_class(linear_type: LinearType):
    if linear_type == "vanila":
        return nn.Linear
    if linear_type == "host":
        from app.python.layers.linearLayer import Linear

        return Linear
    if linear_type == "horizontal":
        from app.python.layers.linearLayer import LinearHorizontal

        return LinearHorizontal
    if linear_type == "multicast":
        from app.python.layers.linearLayerMulticast import LinearMulticast

        return LinearMulticast
    if linear_type == "horizontal_multicast":
        from app.python.layers.linearLayerMulticast import LinearHorizontalMulticast

        return LinearHorizontalMulticast
    raise ValueError(f"Unsupported linear type: {linear_type}")


def _linear_kwargs(config: LinearReplacementConfig, bias: bool) -> dict:
    if config.linear_type == "vanila":
        return {
            "bias": bias,
            "device": config.device,
            "dtype": config.dtype,
        }

    multicast_kernel = config.multicast_kernel
    if multicast_kernel is None:
        multicast_kernel = config.linear_type in {"multicast", "horizontal_multicast"}

    return {
        "smem_size": config.smem_size
        if config.smem_size is not None
        else _get_smem_size(multicast=multicast_kernel),
        "num_copy_blocks": config.num_copy_blocks,
        "bias": bias and not config.drop_bias,
        "device": config.device,
        "dtype": config.dtype,
        "skip_init": config.skip_init,
    }


def _get_smem_size(multicast: bool) -> int:
    from app.python.utils import get_smem_size

    return get_smem_size(multicast=multicast)


def _dtype_to_name(dtype: torch.dtype | None) -> str | None:
    if dtype is None:
        return None
    return str(dtype).removeprefix("torch.")


def _name_to_dtype(dtype: str | torch.dtype | None) -> torch.dtype | None:
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    return getattr(torch, dtype.removeprefix("torch."))


def _replace_one_linear(old_linear: nn.Linear, config: LinearReplacementConfig) -> nn.Module:
    if config.device is None or config.dtype is None:
        config = LinearReplacementConfig(
            linear_type=config.linear_type,
            smem_size=config.smem_size,
            num_copy_blocks=config.num_copy_blocks,
            device=config.device or old_linear.weight.device,
            dtype=config.dtype or old_linear.weight.dtype,
            multicast_kernel=config.multicast_kernel,
            include_embedding_projections=config.include_embedding_projections,
            drop_bias=config.drop_bias,
            skip_init=config.skip_init,
        )
    new_linear = make_replacement_linear(
        old_linear.in_features,
        old_linear.out_features,
        old_linear.bias is not None,
        config,
    )

    with torch.no_grad():
        new_linear.weight.copy_(old_linear.weight.to(device=new_linear.weight.device, dtype=new_linear.weight.dtype))
        if getattr(new_linear, "bias", None) is not None and old_linear.bias is not None:
            new_linear.bias.copy_(old_linear.bias.to(device=new_linear.bias.device, dtype=new_linear.bias.dtype))

    return new_linear


def replace_opt_linears(
    model: nn.Module,
    linear_type: LinearType | LinearReplacementConfig,
    target_names: Iterable[str] = OPT_TARGET_LINEARS,
) -> nn.Module:
    """Replace OPT decoder q/k/v/o and MLP linears after loading a vanilla model."""
    config = linear_type if isinstance(linear_type, LinearReplacementConfig) else LinearReplacementConfig(linear_type)
    targets = set(target_names)
    if config.include_embedding_projections:
        targets.update(OPT_PROJECTION_LINEARS)

    for layer in model.model.decoder.layers:
        for name in targets:
            if name in OPT_PROJECTION_LINEARS:
                continue
            parent = layer.self_attn if name in {"q_proj", "k_proj", "v_proj", "out_proj"} else layer
            old_linear = getattr(parent, name)
            if old_linear is not None:
                setattr(parent, name, _replace_one_linear(old_linear, config))

    if config.include_embedding_projections:
        decoder = model.model.decoder
        for name in OPT_PROJECTION_LINEARS:
            old_linear = getattr(decoder, name)
            if old_linear is not None:
                setattr(decoder, name, _replace_one_linear(old_linear, config))

    return model


def replace_llama_linears(
    model: nn.Module,
    linear_type: LinearType | LinearReplacementConfig,
    target_names: Iterable[str] = LLAMA_TARGET_LINEARS,
) -> nn.Module:
    """Replace LLaMA attention, MLP, and lm_head linears after loading a vanilla model."""
    config = linear_type if isinstance(linear_type, LinearReplacementConfig) else LinearReplacementConfig(linear_type)
    targets = set(target_names)

    for layer in model.model.layers:
        for name in targets:
            if name in LLAMA_HEAD_LINEARS:
                continue
            parent = layer.self_attn if name in LLAMA_ATTENTION_LINEARS else layer.mlp
            old_linear = getattr(parent, name)
            if old_linear is not None:
                setattr(parent, name, _replace_one_linear(old_linear, config))

    if "lm_head" in targets:
        old_linear = getattr(model, "lm_head", None)
        if old_linear is not None:
            setattr(model, "lm_head", _replace_one_linear(old_linear, config))

    return model


def load_replaced_opt_for_causal_lm(
    pretrained_model_name_or_path,
    linear_type: LinearType | LinearReplacementConfig,
    *args,
    **kwargs,
):
    from app.python.opt.vanila_opt import OPTForCausalLM

    config = kwargs.get("config")
    if config is None:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        kwargs["config"] = config
    set_opt_linear_config(config, linear_type)
    return OPTForCausalLM.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)


def load_replaced_llama_for_causal_lm(
    pretrained_model_name_or_path,
    linear_type: LinearType | LinearReplacementConfig,
    *args,
    **kwargs,
):
    from app.python.llama.vanila_llama import LlamaForCausalLM

    config = kwargs.get("config")
    if config is None:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        kwargs["config"] = config
    set_llama_linear_config(config, linear_type)
    return LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
