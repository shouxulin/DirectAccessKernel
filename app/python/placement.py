from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


DEFAULT_HORIZONTAL_CONFIG = {
    "GH200": {
        (7168, 7168): (2560, 4608, 40, 90, 1, 1),
        (7168, 28672): (2560, 4608, 40, 90, 1, 1),
        (28672, 7168): (10240, 18432, 40, 90, 1, 1),
    },
    "RTX6000": {
        (7168, 7168): (128, 7040, 2, 128, 1, 1),
        (7168, 28672): (192, 6976, 3, 128, 1, 1),
        (28672, 7168): (832, 27840, 8, 160, 1, 1),
        (4096, 4096): (64, 4032, 2, 128, 2, 2),
        (4096, 16384): (64, 4032, 2, 128, 2, 2),
        (16384, 4096): (384, 16000, 3, 140, 1, 1),
    },
}


@dataclass(frozen=True)
class ModelPlacementStats:
    host_bytes: int
    device_bytes: int
    actual_offload_ratio: float


def place_model(
    model: nn.Module,
    offload_ratio: float,
    *,
    gpu_name: str,
    use_config: bool = False,
    horizontal_config: Mapping[str, Mapping[tuple[int, int], tuple[int, int, int, int, int, int]]] | None = None,
    default_horizontal_config: Mapping[str, tuple[int, int, int, int, int, int]] | None = None,
    device: torch.device | str = "cuda",
    build_tma_desc: bool = True,
) -> ModelPlacementStats:
    """Place non-custom model weights on GPU and place replaced Linear weights by offload mode."""
    custom_types = _custom_linear_types()

    _move_non_custom_direct_tensors(model, custom_types, device)

    for name, module in model.named_modules():
        if not isinstance(module, custom_types):
            continue

        if _is_horizontal_linear(module):
            _place_horizontal_linear(
                name,
                module,
                offload_ratio,
                gpu_name=gpu_name,
                use_config=use_config,
                horizontal_config=horizontal_config or DEFAULT_HORIZONTAL_CONFIG,
                default_horizontal_config=default_horizontal_config,
            )
            _move_custom_linear_bias(module, device)
        elif _is_host_linear(module):
            target = "cpu" if offload_ratio >= 1.0 else device
            module.move_weight_to_device(target)
            _move_custom_linear_bias(module, device)
            if build_tma_desc:
                module.build_tma_desc()
        else:
            raise TypeError(f"Unsupported replaced Linear module at {name}: {type(module)!r}")

    return get_model_placement_stats(model)


def set_linear_prefill(model: nn.Module, prefill: bool) -> None:
    for module in model.modules():
        if hasattr(module, "prefill"):
            module.prefill = prefill


def get_model_placement_stats(model: nn.Module) -> ModelPlacementStats:
    host_bytes = 0
    device_bytes = 0
    for param in model.parameters():
        param_bytes = param.numel() * param.element_size()
        if param.device.type == "cpu":
            host_bytes += param_bytes
        else:
            device_bytes += param_bytes

    total_bytes = host_bytes + device_bytes
    actual_offload_ratio = host_bytes / total_bytes if total_bytes else 0.0
    return ModelPlacementStats(host_bytes, device_bytes, actual_offload_ratio)


def _place_horizontal_linear(
    name: str,
    module: nn.Module,
    offload_ratio: float,
    *,
    gpu_name: str,
    use_config: bool,
    horizontal_config: Mapping[str, Mapping[tuple[int, int], tuple[int, int, int, int, int, int]]],
    default_horizontal_config: Mapping[str, tuple[int, int, int, int, int, int]] | None,
) -> None:
    if not 0.0 < offload_ratio < 1.0:
        raise ValueError(
            f"{name} is a horizontal Linear, so offload_ratio must be between 0 and 1; got {offload_ratio}"
        )

    h_m, d_m, h_blocks, d_blocks, h_sms_per_row, d_sms_per_row = _resolve_horizontal_split_config(
        module,
        offload_ratio,
        gpu_name=gpu_name,
        use_config=use_config,
        horizontal_config=horizontal_config,
        default_horizontal_config=default_horizontal_config,
    )
    module.horizontal_split_weight(
        offload_ratio,
        h_blocks + d_blocks,
        h_blocks,
        h_sms_per_row,
        d_sms_per_row,
        h_m,
        d_m,
    )


def _resolve_horizontal_split_config(
    module: nn.Module,
    offload_ratio: float,
    *,
    gpu_name: str,
    use_config: bool,
    horizontal_config: Mapping[str, Mapping[tuple[int, int], tuple[int, int, int, int, int, int]]],
    default_horizontal_config: Mapping[str, tuple[int, int, int, int, int, int]] | None,
) -> tuple[int, int, int, int, int, int]:
    if default_horizontal_config is not None and gpu_name in default_horizontal_config:
        config = default_horizontal_config[gpu_name]
    elif gpu_name == "GH200":
        config = (0, 0, 16, 112, 1, 1)
    elif gpu_name == "RTX6000":
        config = (0, 0, 8, 180, 1, 1)
    else:
        raise NotImplementedError(f"GPU {gpu_name} not supported for horizontal split config")

    if not use_config:
        return config

    weight_shape = tuple(module.weight.shape)
    gpu_config = horizontal_config.get(gpu_name)
    if gpu_config is None:
        raise KeyError(f"Horizontal config for GPU {gpu_name} not found")
    if weight_shape not in gpu_config:
        raise KeyError(f"Weight shape {weight_shape} not found in horizontal config for {gpu_name}")
    return gpu_config[weight_shape]


def _move_non_custom_direct_tensors(
    module: nn.Module,
    custom_types: tuple[type[nn.Module], ...],
    device: torch.device | str,
) -> None:
    if isinstance(module, custom_types):
        return

    _move_direct_parameters_and_buffers(module, device)
    for child in module.children():
        _move_non_custom_direct_tensors(child, custom_types, device)


def _move_direct_parameters_and_buffers(module: nn.Module, device: torch.device | str) -> None:
    with torch.no_grad():
        for param in module._parameters.values():
            if param is not None:
                param.data = param.data.to(device, non_blocking=True)
                if param.grad is not None:
                    param.grad.data = param.grad.data.to(device, non_blocking=True)

        for name, buffer in module._buffers.items():
            if buffer is not None:
                module._buffers[name] = buffer.to(device, non_blocking=True)


def _move_custom_linear_bias(module: nn.Module, device: torch.device | str) -> None:
    bias = getattr(module, "bias", None)
    if bias is None:
        return
    with torch.no_grad():
        bias.data = bias.data.to(device, non_blocking=True)
        if bias.grad is not None:
            bias.grad.data = bias.grad.data.to(device, non_blocking=True)


def _custom_linear_types() -> tuple[type[nn.Module], ...]:
    from app.python.layers.linearLayer import Linear, LinearHorizontal
    from app.python.layers.linearLayerMulticast import LinearMulticast
    from app.python.layers.linearLayerMulticast import LinearHorizontalMulticast

    return (Linear, LinearHorizontal, LinearMulticast, LinearHorizontalMulticast)


def _is_horizontal_linear(module: Any) -> bool:
    return hasattr(module, "horizontal_split_weight")


def _is_host_linear(module: Any) -> bool:
    return hasattr(module, "move_weight_to_device")
