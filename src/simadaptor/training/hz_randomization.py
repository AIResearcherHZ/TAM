from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp

from simadaptor.config import TrainConfig


def get_hz_randomization_settings(cfg: TrainConfig) -> tuple[int, tuple[int, ...]]:
    base_hz = int(getattr(cfg, "hz_randomization_base_hz", 1000) or 1000)
    raw_choices = tuple(int(hz) for hz in getattr(cfg, "hz_randomization_choices", (base_hz,)))
    if not raw_choices:
        raise ValueError("hz_randomization_choices must be non-empty.")
    for hz in raw_choices:
        if hz <= 0:
            raise ValueError(f"hz_randomization_choices must be positive, got {raw_choices}.")
        if base_hz % hz != 0:
            raise ValueError(
                f"hz_randomization_base_hz={base_hz} must be divisible by each sampled Hz, got {raw_choices}."
            )
    return base_hz, raw_choices


def sample_hz_per_batch(
    rng: jax.Array,
    batch_size: int,
    *,
    base_hz: int,
    choices: tuple[int, ...],
    enabled: bool,
) -> jax.Array:
    if not enabled:
        return jnp.full((batch_size,), int(base_hz), dtype=jnp.int32)
    choices_arr = jnp.asarray(choices, dtype=jnp.int32)
    idx = jax.random.randint(rng, shape=(batch_size,), minval=0, maxval=choices_arr.shape[0])
    return choices_arr[idx]


def build_history_bernoulli_keep_mask(
    rng: jax.Array,
    sampled_hz: jax.Array,
    time_steps: int,
    *,
    base_hz: int,
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    keep_prob = (sampled_hz.astype(jnp.float32) / float(base_hz))[:, None, None]
    mask = jax.random.bernoulli(rng, p=keep_prob, shape=(sampled_hz.shape[0], int(time_steps), 1))
    return mask.astype(dtype)


def build_exact_step_keep_mask(
    sampled_hz: jax.Array,
    window_len: int,
    *,
    base_hz: int,
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    period = (int(base_hz) // sampled_hz.astype(jnp.int32))[:, None]
    idx_from_end = (int(window_len) - 1) - jnp.arange(int(window_len), dtype=jnp.int32)[None, :]
    keep = jnp.mod(idx_from_end, period) == 0
    return keep[..., None].astype(dtype)


def compute_history_token_time(
    hist_idx: jax.Array,
    *,
    n_tokens: int,
    total_steps: int,
    patch_size: int,
    patch_stride: int,
) -> jax.Array:
    stride = max(int(patch_stride), 1)
    patch = max(int(patch_size), 1)
    trim_len = (max(int(n_tokens), 1) - 1) * stride + patch
    first_start = max(int(total_steps) - trim_len, 0)
    hist_token_time = first_start + hist_idx.astype(jnp.int32) * stride + (patch - 1)
    return jnp.clip(hist_token_time, 0, max(int(total_steps) - 1, 0))


def select_source_by_token_mask(
    x_orig: Optional[jax.Array],
    x_shuf: Optional[jax.Array],
    use_original_source: jax.Array,
):
    if x_orig is None and x_shuf is None:
        return None
    if x_orig is None:
        return x_shuf
    if x_shuf is None:
        return x_orig
    source_mask = jnp.asarray(use_original_source)
    while source_mask.ndim < x_orig.ndim:
        source_mask = source_mask[..., None]
    return jnp.where(source_mask, x_orig, x_shuf)


def apply_time_keep_mask(x: jax.Array, keep_mask: Optional[jax.Array]) -> jax.Array:
    if keep_mask is None:
        return x
    mask = jnp.asarray(keep_mask, dtype=x.dtype)
    while mask.ndim < x.ndim:
        mask = mask[..., None]
    return x * mask


def mix_time_series_by_cut(
    x_orig: jax.Array,
    x_shuf: jax.Array,
    cut_t: jax.Array,
) -> jax.Array:
    if x_orig.shape != x_shuf.shape:
        raise ValueError(f"orig/shuf shapes must match, got {x_orig.shape} vs {x_shuf.shape}")
    if x_orig.shape[0] != cut_t.shape[0]:
        raise ValueError(f"cut_t batch {cut_t.shape[0]} must match inputs batch {x_orig.shape[0]}")
    source_mask = jnp.arange(x_orig.shape[1], dtype=jnp.int32)[None, :] < cut_t[:, None]
    while source_mask.ndim < x_orig.ndim:
        source_mask = source_mask[..., None]
    return jnp.where(source_mask, x_orig, x_shuf)
