from __future__ import annotations

from typing import Any

import torch


class PinnedStaticCache:
    """Minimal host-side StaticCache backed by pinned CPU tensors.

    This intentionally implements only the small subset of the Transformers
    cache API used by the OPT host-KV benchmark.
    """

    def __init__(
        self,
        config,
        batch_size: int,
        max_cache_len: int,
        dtype: torch.dtype,
    ) -> None:
        if batch_size < 0:
            raise ValueError("batch_size must be non-negative")

        self.max_cache_len = int(max_cache_len)
        self.batch_size = int(batch_size)
        self.dtype = dtype
        self.device = torch.device("cpu")

        self.num_layers = int(getattr(config, "num_hidden_layers"))
        hidden_size = int(getattr(config, "hidden_size"))
        num_attention_heads = int(getattr(config, "num_attention_heads"))
        self.num_heads = int(getattr(config, "num_key_value_heads", num_attention_heads))
        self.head_dim = hidden_size // num_attention_heads

        shape = (self.batch_size, self.num_heads, self.max_cache_len, self.head_dim)
        self.key_cache = [
            torch.zeros(shape, dtype=self.dtype, device="cpu", pin_memory=True)
            for _ in range(self.num_layers)
        ]
        self.value_cache = [
            torch.zeros(shape, dtype=self.dtype, device="cpu", pin_memory=True)
            for _ in range(self.num_layers)
        ]
        self._seq_lengths = [0 for _ in range(self.num_layers)]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not (0 <= layer_idx < self.num_layers):
            raise IndexError(f"layer_idx {layer_idx} is out of range for {self.num_layers} layers")
        # if key_states.is_cuda or value_states.is_cuda:
        #     raise ValueError("PinnedStaticCache expects CPU key/value states")
        if key_states.shape != value_states.shape:
            raise ValueError("key_states and value_states must have the same shape")
        if key_states.dim() != 4:
            raise ValueError("key_states must have shape [B, H, T, D]")
        if key_states.shape[0] != self.batch_size:
            raise ValueError(f"key batch {key_states.shape[0]} does not match cache batch {self.batch_size}")
        if key_states.shape[1] != self.num_heads or key_states.shape[-1] != self.head_dim:
            raise ValueError("key/value head shape does not match cache shape")

        cache_position = cache_kwargs.get("cache_position") if cache_kwargs is not None else None
        if cache_position is None:
            cache_position = torch.arange(key_states.shape[2], device="cpu", dtype=torch.long)
        elif cache_position.is_cuda or cache_position.dtype != torch.long:
            cache_position = cache_position.to(device="cpu", dtype=torch.long)

        if cache_position.dim() != 1:
            raise ValueError("cache_position must be a 1D tensor")
        if cache_position.numel() != key_states.shape[2]:
            raise ValueError("cache_position length must match key/value sequence length")

        keys = self.key_cache[layer_idx]
        values = self.value_cache[layer_idx]
        self._copy_into_cache(keys, cache_position, key_states)
        self._copy_into_cache(values, cache_position, value_states)

        if cache_position.numel() > 0:
            self._seq_lengths[layer_idx] = max(
                self._seq_lengths[layer_idx],
                int(cache_position.max().item()) + 1,
            )
        return keys, values

    @staticmethod
    def _copy_into_cache(cache: torch.Tensor, cache_position: torch.Tensor, states: torch.Tensor) -> None:
        non_blocking = states.is_cuda
        if cache_position.numel() == 0:
            return

        start = int(cache_position[0].item())
        length = cache_position.numel()
        end = start + length
        if length == 1 or torch.equal(cache_position, torch.arange(start, end, device="cpu", dtype=torch.long)):
            cache[:, :, start:end, :].copy_(states, non_blocking=non_blocking)
            return

        for src_idx, dst_idx in enumerate(cache_position.tolist()):
            cache[:, :, dst_idx : dst_idx + 1, :].copy_(
                states[:, :, src_idx : src_idx + 1, :],
                non_blocking=non_blocking,
            )

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if not (0 <= layer_idx < self.num_layers):
            return 0
        return self._seq_lengths[layer_idx]

    def get_max_cache_shape(self) -> int:
        return self.max_cache_len
