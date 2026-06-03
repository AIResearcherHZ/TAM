from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

from flax.struct import dataclass as flax_dataclass
import jax
import jax.numpy as jnp

import simadaptor.models.transformer as models_transformer


_NORM_STAT_DEFAULTS = {
    "mean_q": 0.0,
    "mean_dq": 0.0,
    "mean_u": 0.0,
    "var_q": 1.0,
    "var_dq": 1.0,
    "var_u": 1.0,
}


@flax_dataclass
class OnlineHistoryState:
    history_emb: jax.Array
    cache: Any
    q_buf: jax.Array
    qd_buf: jax.Array
    tau_buf: jax.Array
    keep_buf: jax.Array
    sample_count: jax.Array
    next_emit_idx: jax.Array
    has_embedding: jax.Array


@dataclass(frozen=True)
class OnlineHistoryRuntimeConfig:
    patch_size: int
    patch_stride: int
    context_half: int
    decode_patch_size: int
    arm_dof: int
    emb_dim: int
    jointwise: bool


@dataclass(frozen=True)
class OnlineHistoryRuntime:
    config: OnlineHistoryRuntimeConfig
    cache0: Any
    decode_step: Callable[..., tuple[jax.Array, Any]]


def _resize_norm_stat_leaf(value: Any, dof: int, fill_value: float) -> Any:
    arr = jnp.asarray(value)
    if arr.ndim == 0 or int(arr.shape[-1]) == int(dof):
        return value
    if int(arr.shape[-1]) > int(dof):
        return arr[..., : int(dof)]
    pad_width = [(0, 0)] * arr.ndim
    pad_width[-1] = (0, int(dof) - int(arr.shape[-1]))
    return jnp.pad(arr, tuple(pad_width), mode="constant", constant_values=fill_value)


def align_norm_stats_to_dof(norm_stats: Any, dof: int) -> Any:
    """Make saved per-DoF norm stats usable by narrower online eval controllers."""
    if norm_stats is None:
        return None
    dof = int(dof)
    if dof <= 0:
        return norm_stats
    if isinstance(norm_stats, Mapping):
        out = dict(norm_stats)
        for field, fill in _NORM_STAT_DEFAULTS.items():
            if field in out:
                out[field] = _resize_norm_stat_leaf(out[field], dof, fill)
        return out
    updates = {}
    for field, fill in _NORM_STAT_DEFAULTS.items():
        if hasattr(norm_stats, field):
            updates[field] = _resize_norm_stat_leaf(getattr(norm_stats, field), dof, fill)
    if not updates:
        return norm_stats
    if hasattr(norm_stats, "replace"):
        return norm_stats.replace(**updates)
    return norm_stats.__class__(**{**vars(norm_stats), **updates})


def push_window(window: jax.Array, new_val: jax.Array) -> jax.Array:
    new_val = jnp.asarray(new_val, dtype=window.dtype)
    return jnp.concatenate([window[1:], new_val[None, ...]], axis=0)


def zero_torque_history_keep_mask(
    raw_tau: jax.Array,
    *,
    threshold: float = 1e-5,
) -> jax.Array:
    raw_tau = jnp.asarray(raw_tau)
    return jnp.where(
        jnp.all(jnp.abs(raw_tau) <= float(threshold), axis=-1),
        0.0,
        1.0,
    ).astype(raw_tau.dtype)


def mask_zero_torque_history_sample(
    q: jax.Array,
    qd: jax.Array,
    tau_model: jax.Array,
    raw_tau: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    keep = zero_torque_history_keep_mask(raw_tau)
    return q * keep, qd * keep, tau_model * keep, keep


def runtime_history_embedding_from_sequence(history_seq: jax.Array) -> jax.Array:
    history_seq = jnp.asarray(history_seq)
    if history_seq.ndim == 1:
        return history_seq[None, :]
    if history_seq.ndim == 2:
        return history_seq[-1:, :]
    if history_seq.ndim == 3:
        return history_seq[-1:, :, :]
    if history_seq.ndim == 4:
        return history_seq[:, -1, ...]
    raise ValueError(
        f"Unexpected history embedding rank: {history_seq.ndim}, shape={history_seq.shape}"
    )


def build_online_history_runtime(
    *,
    hist_model,
    params_hist_example,
    emb_dim: int,
    arm_dof: int,
) -> OnlineHistoryRuntime:
    hist_cfg = getattr(hist_model, "cfg", hist_model)
    patch_size = int(getattr(hist_cfg, "patch_size"))
    patch_stride = int(getattr(hist_cfg, "patch_stride"))
    masked_fit_half = int(
        getattr(hist_cfg, "masked_fit_max_neighbors_each_side", 50) or 0
    )
    context_half = masked_fit_half if masked_fit_half > 0 else 0
    decode_patch_size = patch_size + 2 * context_half if context_half > 0 else patch_size
    cfg = OnlineHistoryRuntimeConfig(
        patch_size=patch_size,
        patch_stride=patch_stride,
        context_half=context_half,
        decode_patch_size=decode_patch_size,
        arm_dof=int(arm_dof),
        emb_dim=int(emb_dim),
        jointwise=True,
    )
    cache0 = models_transformer.init_infer_state(
        params_hist_example,
        hist_model,
        batch_size=1,
    )

    @jax.jit
    def decode_step(
        params_hist,
        cache,
        q_patch,
        qd_patch,
        tau_patch,
        input_keep_mask,
        norm_stats,
    ):
        valid_mask = jnp.ones(q_patch.shape[:2], dtype=jnp.float32)
        emb, cache_out = models_transformer.step_decode(
            params=params_hist,
            cache=cache,
            model=hist_model,
            chunk_q=q_patch,
            chunk_qd=qd_patch,
            chunk_u=tau_patch,
            valid_mask=valid_mask,
            key=None,
            norm_stats=norm_stats,
            input_keep_mask=input_keep_mask,
        )
        return emb[:, 0, ...], cache_out

    return OnlineHistoryRuntime(config=cfg, cache0=cache0, decode_step=decode_step)


def init_online_history_state(
    runtime: OnlineHistoryRuntime,
    *,
    dtype=jnp.float32,
) -> OnlineHistoryState:
    cfg = runtime.config
    dummy_emb = (
        jnp.zeros((1, cfg.arm_dof, cfg.emb_dim), dtype=dtype)
        if cfg.jointwise
        else jnp.zeros((1, cfg.emb_dim), dtype=dtype)
    )
    q0 = jnp.zeros((cfg.decode_patch_size, cfg.arm_dof), dtype=dtype)
    qd0 = jnp.zeros((cfg.decode_patch_size, cfg.arm_dof), dtype=dtype)
    tau0 = jnp.zeros((cfg.decode_patch_size, cfg.arm_dof), dtype=dtype)
    keep0 = jnp.zeros((cfg.decode_patch_size,), dtype=dtype)
    return OnlineHistoryState(
        history_emb=dummy_emb,
        cache=runtime.cache0,
        q_buf=q0,
        qd_buf=qd0,
        tau_buf=tau0,
        keep_buf=keep0,
        sample_count=jnp.asarray(1, dtype=jnp.int32),
        next_emit_idx=jnp.asarray(cfg.patch_size - 1, dtype=jnp.int32),
        has_embedding=jnp.asarray(False),
    )


def advance_online_history_state(
    runtime: OnlineHistoryRuntime,
    state: OnlineHistoryState,
    *,
    q_arm: jax.Array,
    qd_arm: jax.Array,
    tau_arm: jax.Array,
    raw_tau_arm: jax.Array,
    params_hist,
    norm_stats,
) -> OnlineHistoryState:
    q_masked, qd_masked, tau_masked, keep = mask_zero_torque_history_sample(
        q_arm,
        qd_arm,
        tau_arm,
        raw_tau_arm,
    )
    pushed = state.replace(
        q_buf=push_window(state.q_buf, q_masked),
        qd_buf=push_window(state.qd_buf, qd_masked),
        tau_buf=push_window(state.tau_buf, tau_masked),
        keep_buf=push_window(state.keep_buf, keep),
        sample_count=state.sample_count + 1,
    )
    should_emit = (pushed.sample_count - 1) >= (
        pushed.next_emit_idx + runtime.config.context_half
    )

    def emit_token(st: OnlineHistoryState) -> OnlineHistoryState:
        q_patch = st.q_buf[None, None, ...]
        qd_patch = st.qd_buf[None, None, ...]
        tau_patch = st.tau_buf[None, None, ...]
        keep_patch = st.keep_buf[None, None, ...]
        history_emb, cache = runtime.decode_step(
            params_hist,
            st.cache,
            q_patch,
            qd_patch,
            tau_patch,
            keep_patch,
            align_norm_stats_to_dof(norm_stats, runtime.config.arm_dof),
        )
        return st.replace(
            history_emb=history_emb,
            cache=cache,
            next_emit_idx=st.next_emit_idx + runtime.config.patch_stride,
            has_embedding=jnp.asarray(True),
        )

    return jax.lax.cond(should_emit, emit_token, lambda st: st, pushed)


def apply_online_adaptor(
    *,
    adaptor_model,
    adaptor_apply_fn: Callable[..., tuple[jax.Array, Any]] | None = None,
    params_adaptor,
    q_window: jax.Array,
    qd_window: jax.Array,
    tau_window: jax.Array,
    history_emb: jax.Array,
    norm_stats,
) -> tuple[jax.Array, jax.Array]:
    norm_stats = align_norm_stats_to_dof(norm_stats, int(tau_window.shape[-1]))
    if adaptor_apply_fn is None:
        delta_tau, _ = adaptor_model.apply(
            params_adaptor,
            q_window[None, ...],
            qd_window[None, ...],
            tau_window[None, ...],
            history_emb,
            norm_stats=norm_stats,
        )
    else:
        delta_tau, _ = adaptor_apply_fn(
            params_adaptor,
            q_window[None, ...],
            qd_window[None, ...],
            tau_window[None, ...],
            history_emb,
            jax.random.PRNGKey(0),
            False,
            norm_stats,
        )
    delta_tau = delta_tau[0]
    return tau_window[-1] + delta_tau, delta_tau
