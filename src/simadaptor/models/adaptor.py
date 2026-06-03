
import flax.linen as nn
import einops
import jax
import jax.numpy as jnp

from typing import Any, Optional, Tuple
import numpy as np
from simadaptor.physics import smoothing as smoothing_util
from simadaptor.core.structs import NormStats
import simadaptor.physics.dynamics as dynamics
from mujoco import mjx
from functools import partial

# =========================
# Utilities
# =========================
def _broadcast_keep_mask(mask: jax.Array, ref: jax.Array) -> jax.Array:
    """Broadcast a time-step keep mask to match a reference tensor with trailing DoF dim."""
    mask = jnp.asarray(mask, dtype=ref.dtype)
    if mask.shape[:2] != ref.shape[:2]:
        raise ValueError(f"keep mask leading dims {mask.shape[:2]} must match input {ref.shape[:2]}")
    while mask.ndim < ref.ndim:
        mask = mask[..., None]
    try:
        return jnp.broadcast_to(mask, ref.shape[:-1] + (1,))
    except ValueError as exc:
        raise ValueError(
            f"keep mask shape {mask.shape} is not broadcastable to input prefix {ref.shape[:-1]}"
        ) from exc


def chunk_time(x: jnp.ndarray, patch: int, stride: Optional[int] = None, pad_value: float = 0.0) -> jnp.ndarray:
    """
    Overlapping temporal chunking.
    Args:
        x: [B, T, D]
        patch: window length (P)
        stride: hop size (S). Default = patch//2 (50% overlap)
        pad_value: constant pad at tail if needed
    Returns:
        [B, Np, patch*D] where Np = 1 + ceil((T - P)/S)
    """
    if x.ndim == 4:
        # already patchified
        assert x.shape[-3] == 1
        return x
    B, T, D = x.shape
    if stride is None:
        stride = max(1, patch // 2)

    # n_windows = 1 + np.ceil((T - patch) / stride).astype(int)
    n_windows = 1 + np.floor((T - patch) / stride).astype(int)
    n_windows = np.maximum(n_windows, 1)

    last_start = (n_windows - 1) * stride
    trim_len = last_start + patch
    # pad_len = np.maximum(0, total_needed - T)
    
    # trim_len = n_windows

    def _pad(a, pad_len_):
        pad_width = ((0, 0), (0, int(pad_len_)), (0, 0))
        return jnp.pad(a, pad_width, mode="constant", constant_values=pad_value)

    x_pad = x[:,-trim_len:]
    # x_pad = jax.lax.cond(pad_len > 0, lambda a: a[], lambda a: a, x)

    starts = jnp.arange(n_windows) * stride

    def take_window(start_idx):
        start_idx = jnp.asarray(start_idx, dtype=jnp.int32)
        return jax.lax.dynamic_slice(x_pad, (0, start_idx, 0), (B, patch, D))  # [B, P, D]

    windows = jax.vmap(take_window)(starts)      # [Np, B, P, D]
    windows = jnp.swapaxes(windows, 0, 1)        # [B, Np, P, D]
    # return windows.reshape(B, n_windows, patch * D)
    return windows

# =========================
# Models
# =========================
class HistoryEmbed(nn.Module):
    """Encode recent robot history into adaptor conditioning tokens.

    Inputs are aligned joint position `q`, velocity `qd`, and torque `u`
    histories sampled at the controller rate.

    Torque semantics:
    - `u` is the logged torque that was actually applied to the plant
      (`tau_cmd`).
    - In online deployment this should be the post-adaptor command sent to the
      actuator model, not the raw pre-adaptor desired torque.

    Accepted input shapes:
    - Offline / training path: `[B, T, DoF]`
    - Decode path with one already-built patch: `[B, 1, P, DoF]`

    Output shapes depend on `jointwise`:
    - Global encoder: `[B, N, C]`
    - Jointwise encoder: `[B, N, DoF, C]`

    `N` is the number of temporal patches after chunking, and is `1` for the
    decode path.
    """
    emb_dim: int
    patch_size: int
    patch_stride: int
    jointwise: bool = False
    ideal_mjx_model: Optional[mjx.Model] = None
    masked_fit_max_neighbors_each_side: int = 50
    masked_fit_q_weight: float = 2.0
    masked_fit_qd_weight: float = 1.0
    dropout:float = 0.3

    @nn.compact
    def __call__(
        self,
        q,
        qd,
        u,
        *,
        deterministic: bool = False,
        norm_stats: Optional[dict] = None,
        input_keep_mask: Optional[jax.Array] = None,
    ):
        """
        Args:
            q, qd:
                Joint position / velocity history. Shapes are either
                `[B, T, DoF]` or decode-time `[B, 1, P, DoF]`.
            u:
                Logged applied torque history with the same shape as `q`.
                This corresponds to `tau_cmd`, not just the raw desired torque.
            norm_stats:
                Optional dict with per-DoF mean/var for q/qd/u.
            input_keep_mask:
                Optional keep mask aligned with the leading batch/time axes.
        Returns:
            History tokens with shape `[B, N, C]` for global encoders or
            `[B, N, DoF, C]` for jointwise encoders.
        """
        assert self.ideal_mjx_model is not None, "HistoryEmbed requires ideal_mjx_model."

        # Calculate accelerations with the local masked fit. Decode-time patches
        # may include symmetric extra context; trim it away after fitting.
        dt = 0.001  # assume 1kHz control freq
        if q.ndim == 4 and q.shape[-3] == 1:
            extra = int(q.shape[-2] - self.patch_size)
            half = extra // 2 if (extra >= 2 and (extra % 2) == 0) else 0
            q_seq = q[:, 0, :, :]
            qd_seq = qd[:, 0, :, :]
            if input_keep_mask is None:
                fit_keep_mask = jnp.ones((q_seq.shape[0], q_seq.shape[1], 1), dtype=q_seq.dtype)
            else:
                fit_keep_mask = jnp.asarray(input_keep_mask, dtype=q_seq.dtype)
                if fit_keep_mask.ndim == 2:
                    if fit_keep_mask.shape != q_seq.shape[:2]:
                        raise ValueError(
                            f"decode-time input_keep_mask must match [B,P]={q_seq.shape[:2]}, got {fit_keep_mask.shape}"
                        )
                    fit_keep_mask = fit_keep_mask[..., None]
                elif fit_keep_mask.ndim == 3:
                    if fit_keep_mask.shape[:2] == (q_seq.shape[0], 1) and fit_keep_mask.shape[2] == q_seq.shape[1]:
                        fit_keep_mask = jnp.swapaxes(fit_keep_mask, 1, 2)
                    elif fit_keep_mask.shape != (q_seq.shape[0], q_seq.shape[1], 1):
                        raise ValueError(
                            "decode-time input_keep_mask must be [B,P], [B,1,P], or [B,P,1]; "
                            f"got {fit_keep_mask.shape}"
                        )
                else:
                    raise ValueError(
                        "decode-time input_keep_mask must be rank 2 or 3; "
                        f"got {fit_keep_mask.shape}"
                    )
            q_fit, qd_fit, qdd_fit = smoothing_util.estimate_masked_state_derivatives(
                q_seq,
                qd_seq,
                fit_keep_mask,
                base_dt=dt,
                max_neighbors_each_side=self.masked_fit_max_neighbors_each_side,
                q_weight=self.masked_fit_q_weight,
                qd_weight=self.masked_fit_qd_weight,
            )
            if half > 0:
                q_fit = q_fit[:, half:-half, :]
                qd_fit = qd_fit[:, half:-half, :]
                qdd_fit = qdd_fit[:, half:-half, :]
                u = u[..., half:-half, :]
                if input_keep_mask is not None:
                    input_keep_mask = input_keep_mask[..., half:-half]
            q = q_fit[:, None, :, :]
            qd = qd_fit[:, None, :, :]
            qdd = qdd_fit[:, None, :, :]
        elif q.ndim == 3:
            fit_keep_mask = input_keep_mask
            if fit_keep_mask is None:
                fit_keep_mask = jnp.ones((q.shape[0], q.shape[1], 1), dtype=q.dtype)
            q, qd, qdd = smoothing_util.estimate_masked_state_derivatives(
                q,
                qd,
                fit_keep_mask,
                base_dt=dt,
                max_neighbors_each_side=self.masked_fit_max_neighbors_each_side,
                q_weight=self.masked_fit_q_weight,
                qd_weight=self.masked_fit_qd_weight,
            )
        else:
            raise ValueError(f"Unsupported HistoryEmbed input shape for local fitting: q={q.shape}, qd={qd.shape}")

        # Public TAM always uses the real-gravity ideal model and includes the
        # ideal-model torque residual feature.
        id_model = self.ideal_mjx_model.replace(
            body_gravcomp=jnp.zeros_like(self.ideal_mjx_model.body_gravcomp)
        )
        required_tau = jnp.vectorize(
            partial(dynamics.mjx_inverse_dynamics_rne, id_model),
            signature='(d),(d),(d)->(d)',
        )(q, qd, qdd)  # [B, T, D]
        # u is the raw logged commanded torque for both modes.
        u_diff = u - required_tau

        # Apply time-step dropout on raw signals before time-patch embedding.
        use_dropout = self.dropout > 0 and not deterministic
        keep_mask = None
        if input_keep_mask is not None:
            keep_mask = _broadcast_keep_mask(input_keep_mask, q)
        elif use_dropout:
            rng = self.make_rng("dropout")
            mask_shape = q.shape[:-1] + (1,)
            keep_mask = jax.random.bernoulli(rng, p=1.0 - self.dropout, shape=mask_shape)
        else:
            keep_mask = jnp.all(jnp.abs(u) > 1e-5, axis=-1, keepdims=True).astype(q.dtype)
        q = q * keep_mask
        qd = qd * keep_mask
        u = u * keep_mask
        u_diff = u_diff * keep_mask

        # Apply per-DoF normalization after computing the torque-residual feature.
        if norm_stats is not None:
            eps = 1e-6
            getter = (lambda field, default: norm_stats.get(field, default)) if isinstance(norm_stats, dict) else (lambda field, default: getattr(norm_stats, field, default))
            mean_q = getter("mean_q", 0.0)
            mean_dq = getter("mean_dq", 0.0)
            mean_u = getter("mean_u", 0.0)
            std_q = jnp.sqrt(getter("var_q", 1.0) + eps)
            std_dq = jnp.sqrt(getter("var_dq", 1.0) + eps)
            std_u = jnp.sqrt(getter("var_u", 1.0) + eps)
            q = (q - mean_q) / std_q
            qd = (qd - mean_dq) / std_dq
            u = (u - mean_u) / std_u
            u_diff = u_diff / std_u

        if self.jointwise:
            x_q = chunk_time(q, self.patch_size, self.patch_stride)   # [B, N, P, D]
            x_qd = chunk_time(qd, self.patch_size, self.patch_stride) # [B, N, P, D]
            x_u = chunk_time(u, self.patch_size, self.patch_stride)   # [B, N, P, D]
            x_u_diff = chunk_time(u_diff, self.patch_size, self.patch_stride)
            feats = [x_q, x_qd, x_u, jax.lax.stop_gradient(x_u_diff)]

            # Flatten patch+modality for each joint token, preserving [B, N, DoF].
            x = jnp.concatenate(feats, axis=2)  # [B, N, P*M, D]
            x = einops.rearrange(x, "b n pm d -> b n d pm")
            x = nn.Dense(self.emb_dim, name="jointwise_input_proj")(x)
            x = nn.gelu(x)
            x = nn.Dense(self.emb_dim, name="jointwise_fusion")(x)
            x = nn.gelu(x)
            x = nn.LayerNorm(name="jointwise_final_ln")(x)
            return x  # [B, N, DoF, C]

        # 1) Patchify each modality independently: [B, Np, P, D]
        x_q_windows = chunk_time(q, self.patch_size, self.patch_stride)
        x_qd_windows = chunk_time(qd, self.patch_size, self.patch_stride)
        x_u_windows = chunk_time(u, self.patch_size, self.patch_stride)
        x_u_diff = jax.lax.stop_gradient(
            chunk_time(u_diff, self.patch_size, self.patch_stride)
        )

        x = jnp.concatenate([q, qd, u, u_diff], axis=-1)  # [B, T, 4D]
        for _ in range(3):
            x = nn.Dense(self.emb_dim)(x)
            x = nn.gelu(x)
        x = nn.LayerNorm()(x)
        if keep_mask is not None:
            x = x * keep_mask

        x = chunk_time(x, self.patch_size, self.patch_stride)
        x = jnp.mean(x, axis=2)  # [B, Np, C]

        # 4) FFT for each patch and modality (over time axis P)
        def fft_embed(x_windows, name: str):
            """
            Per-joint FFT over time axis P, retain D until projection.
            x_windows: [B, Np, P, D]
            Returns: [B, Np, emb_dim]
            """
            x_fft = jnp.fft.rfft(x_windows, axis=2)           # [B, Np, F, D]
            x_real = jnp.real(x_fft)
            x_imag = jnp.imag(x_fft)
            x_mag = jnp.log1p(jnp.abs(x_fft))                 # [B, Np, F, D]

            # Normalize spectral energy per window so tokens emphasize relative structure.
            energy = jnp.sum(x_mag, axis=2, keepdims=True)
            x_mag = x_mag / (energy + 1e-6)

            # Stack magnitude, phase (through real/imag) as features before flattening.
            x_feat = jnp.stack([x_real, x_imag, x_mag], axis=-1)  # [B, Np, F, D, 3]
            x_feat = einops.rearrange(x_feat, "... f d c -> ... (d f c)")  # [B, Np, F*D*3]

            x_emb = nn.Dense(self.emb_dim, name=f"{name}_fft_input_proj")(x_feat)  # [B, Np, emb_dim]
            x_emb = nn.LayerNorm(name=f"{name}_fft_ln")(x_emb)

            return x_emb  # [B, Np, C]

        q_fft = fft_embed(x_q_windows, name="q")        # [B, Np, C]
        qd_fft = fft_embed(x_qd_windows, name="qd")     # [B, Np, C]
        u_fft = fft_embed(x_u_windows, name="u")        # [B, Np, C]
        u_diff_fft = fft_embed(x_u_diff, name="u_diff")  # [B, Np, C]

        x = jnp.concat([x, q_fft, qd_fft, u_fft, u_diff_fft], axis=-1)
        
        for i in range(2):
            x = nn.Dense(self.emb_dim, name=f"final_fusion_{i}")(x)
            x = nn.gelu(x)
        
        x = nn.LayerNorm(name="final_ln")(x)
 
        return x


class AdaLN(nn.Module):
    emb_dim: int

    @nn.compact
    def __call__(self, x, cond):
        if x.ndim - cond.ndim == 1:
            cond = cond[..., None, :]
        scale_shift = nn.Dense(self.emb_dim * 2)(cond)
        gamma, beta = jnp.split(scale_shift, 2, axis=-1)
        x_norm = nn.LayerNorm()(x)
        return x_norm * (1.0 + gamma) + beta


class SimAdaptorBlock(nn.Module):
    hidden_dim: int
    out_dim: int
    final_kernel_init: Any = nn.initializers.lecun_normal()
    final_bias_init: Any = nn.initializers.zeros

    @nn.compact
    def __call__(self, x, cond):
        x = AdaLN(self.hidden_dim)(x, cond)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.gelu(x)
        x = AdaLN(self.hidden_dim)(x, cond)
        x = nn.Dense(self.out_dim, kernel_init=self.final_kernel_init, bias_init=self.final_bias_init)(x)
        return x


class SimAdaptorJointwiseFlat(nn.Module):
    """Predict per-joint torque corrections from short jointwise histories.

    This adaptor consumes a short recent window of `(q, qd, tau)` together with
    a jointwise history embedding and predicts `delta_tau` for each joint.

    Runtime torque convention for `tau`:
    - Earlier samples (`tau[..., :-1, :]`) should be the actually applied torque
      history from previous control steps, i.e. `tau_cmd`.
    - The newest sample (`tau[..., -1, :]`) should be the current desired torque
      before adaptation, i.e. `tau_des` / `tau_plain`.

    This mixed convention matches online deployment: the model conditions on the
    recent applied-torque history while correcting the current desired command.
    The newest desired sample is peeled off as the command input and removed
    from the history stem before the direct residual head is evaluated.
    """
    emb_dim: int
    hidden: int = 256
    depth: int = 3
    dropout: float = 0.1

    @nn.compact
    def __call__(
        self,
        q,
        qd,
        tau,
        history_emb,
        train: bool = False,
        norm_stats: Optional[NormStats] = None,
        input_keep_mask: Optional[jax.Array] = None,
        tau_des_override: Optional[jax.Array] = None,
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """
        Args:
            q, qd, tau:
                Recent controller window with shape `[B, T, DoF]`.
                `tau[..., :-1, :]` is past applied torque history (`tau_cmd`).
                `tau[..., -1, :]` is the current desired torque to be corrected.
            tau_des_override:
                Optional desired torque override with shape `[B, DoF]` or
                `[B, S, DoF]`. This reuses the same history features while
                evaluating one or more desired torques through the command head.
            history_emb:
                Jointwise history embedding shaped `[B, DoF, C]`. Global
                embeddings `[B, C]` or `[B, 1, C]` are broadcast across joints.
        Returns:
            `delta_tau` with shape `[B, DoF]` or `[B, S, DoF]` when
            `tau_des_override` carries sampled desired torques, plus optional
            auxiliary outputs.
        """
        action_output_dim = tau.shape[-1]

        mean_u = 0.0
        std_u = 1.0
        if norm_stats is not None:
            eps = 1e-6
            getter = (lambda field, default: norm_stats.get(field, default)) if isinstance(norm_stats, dict) else (lambda field, default: getattr(norm_stats, field, default))
            mean_q = getter("mean_q", 0.0)
            mean_dq = getter("mean_dq", 0.0)
            mean_u = getter("mean_u", 0.0)
            std_q = jnp.sqrt(getter("var_q", 1.0) + eps)
            std_dq = jnp.sqrt(getter("var_dq", 1.0) + eps)
            std_u = jnp.sqrt(getter("var_u", 1.0) + eps)
            q = (q - mean_q) / std_q
            qd = (qd - mean_dq) / std_dq
            tau = (tau - mean_u) / std_u

        tau_des = tau[..., -1, :] if tau_des_override is None else (tau_des_override - mean_u) / std_u
        if tau_des.shape[0] != tau.shape[0] or tau_des.shape[-1] != action_output_dim:
            raise ValueError(
                f"tau_des_override must align with batch/DoF of tau; got tau={tau.shape}, tau_des={tau_des.shape}"
            )
        tau = tau.at[..., -1, :].set(0.0)

        if input_keep_mask is not None:
            data_mask = _broadcast_keep_mask(input_keep_mask, q)
            q = jnp.where(data_mask, q, 0.0)
            qd = jnp.where(data_mask, qd, 0.0)
            tau = jnp.where(data_mask, tau, 0.0)
        elif train:
            rng = self.make_rng("dropout")
            nT = q.shape[-2]
            data_mask = jnp.arange(nT) >= jax.random.randint(rng, shape=q.shape[:-1], minval=0, maxval=nT)
            q = jnp.where(data_mask[..., None], q, 0.0)
            qd = jnp.where(data_mask[..., None], qd, 0.0)
            tau = jnp.where(data_mask[..., None], tau, 0.0)

        if history_emb.ndim == 2:
            history_emb = history_emb[:, None, :]
        if history_emb.ndim != 3:
            raise ValueError(f"history_emb must be rank-2/3, got shape={history_emb.shape}")
        if history_emb.shape[-2] == 1:
            history_emb = jnp.repeat(history_emb, action_output_dim, axis=-2)
        elif history_emb.shape[-2] != action_output_dim:
            raise ValueError(
                f"history_emb joint axis {history_emb.shape[-2]} must match DoF={action_output_dim}"
            )

        qj = einops.rearrange(q, "b t d -> b d t")
        qdj = einops.rearrange(qd, "b t d -> b d t")
        tauj = einops.rearrange(tau, "b t d -> b d t")

        q_stem = nn.LayerNorm(name="q_stem_ln")(nn.Dense(self.hidden, use_bias=False, name="q_stem")(qj))
        qd_stem = nn.LayerNorm(name="qd_stem_ln")(nn.Dense(self.hidden, use_bias=False, name="qd_stem")(qdj))
        tau_stem = nn.LayerNorm(name="tau_stem_ln")(nn.Dense(self.hidden, use_bias=False, name="tau_stem")(tauj))
        h = q_stem + qd_stem + tau_stem # [B, DoF, hidden]

        # Per-joint feedforward refinement before cross-joint attention.
        for didx in range(self.depth - 1):
            h = SimAdaptorBlock(self.hidden, self.hidden)(h, history_emb)
            h = nn.gelu(h)
            h_global = jnp.mean(h, axis=-2, keepdims=True)  # [B, 1, C]
            h_global = nn.Dense(self.hidden, name=f"joint_global_proj_{didx}")(h_global)
            h = h + h_global

        # History-conditioned per-joint hypernetwork for the direct residual
        # head. The same per-joint affine basis is reused across sampled
        # desired torques, preserving the [B, S, DoF] command-map contract.
        proj_dim = 16
        tau_basis_params = nn.Dense(2 * proj_dim, name="joint_direct_tau_hyper")(h)
        tau_basis_scale = tau_basis_params[..., :proj_dim]  # [B, DoF, P]
        tau_basis_bias = tau_basis_params[..., proj_dim:]  # [B, DoF, P]

        if tau_des.ndim == 2:
            tau_for_proj = tau_des[..., None]  # [B, DoF, 1]
        elif tau_des.ndim == 3:
            tau_for_proj = tau_des[..., None]  # [B, S, DoF, 1]
            tau_basis_scale = tau_basis_scale[:, None, :, :]
            tau_basis_bias = tau_basis_bias[:, None, :, :]
            tau_basis_bias = jnp.broadcast_to(
                tau_basis_bias,
                tau_des.shape + (proj_dim,),
            )  # [B, S, DoF, P]
        else:
            raise ValueError(
                f"tau_des rank must be 2 or 3 for jointwise direct head; got shape={tau_des.shape}"
            )

        tau_projected = tau_basis_scale * tau_for_proj + tau_basis_bias  # [B, S?, DoF, P]
        tau_projected = nn.gelu(tau_projected)

        ndof = tau_projected.shape[-2]
        exclude_self_mask = 1 - np.eye(ndof, dtype=tau_projected.dtype)  # [DoF, DoF]
        other_joint_tau_context = jnp.einsum(
            "...dp,kd->...kp",
            tau_projected,
            exclude_self_mask,
        )  # [B, S?, DoF, P]

        direct_param_input = jnp.concat(
            [tau_projected, other_joint_tau_context],
            axis=-1,
        )  # [B, S?, DoF, 2P]
        direct_features = nn.Dense(proj_dim, name="joint_direct_projected2")(direct_param_input)
        direct_features = nn.gelu(direct_features)
        delta_tau = nn.Dense(
            1,
            name="joint_direct_projected_out",
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(direct_features)[..., 0]

        return delta_tau, None
