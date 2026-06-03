from typing import Tuple, Optional
import numpy as np
import jax
import jax.numpy as jnp


# ============================================================
# 1) Resample to a uniform grid at target_dt (Catmull–Rom Hermite)
# ============================================================

def _catmull_rom_slopes(t_seq: jnp.ndarray, q_seq: jnp.ndarray) -> jnp.ndarray:
    """Per-knot slopes for Catmull–Rom with chordal weights; (T, nq) -> (T, nq)."""
    h = jnp.diff(t_seq)                      # (T-1,)
    h_safe = jnp.where(h <= 0, jnp.inf, h)   # guard
    delta = jnp.diff(q_seq, axis=0) / h_safe[:, None]  # (T-1, nq)
    delta = jnp.where(jnp.isfinite(delta), delta, 0.0)

    m0 = delta[0]
    mN = delta[-1]
    w0 = h[:-1][:, None]
    w1 = h[1:][:, None]
    denom = jnp.where((w0 + w1) == 0.0, 1.0, w0 + w1)
    mid = (w1 * delta[:-1] + w0 * delta[1:]) / denom
    return jnp.concatenate([m0[None], mid, mN[None]], axis=0)  # (T, nq)

def _cubic_interp_with_derivs(t_query, t_seq, q_seq):
    """Evaluate cubic Hermite spline at t_query → (q, qd, qdd)."""
    slopes = _catmull_rom_slopes(t_seq, q_seq)
    eps = jnp.finfo(t_seq.dtype).eps

    def _interp_one(tu):
        idx = jnp.clip(jnp.searchsorted(t_seq, tu, side="right") - 1, 0, t_seq.shape[0] - 2)
        t0 = t_seq[idx]
        t1 = t_seq[idx + 1]
        h = jnp.maximum(t1 - t0, eps)
        inv_h  = 1.0 / h
        inv_h2 = inv_h * inv_h

        s  = (tu - t0) * inv_h
        s2 = s * s
        s3 = s2 * s

        p0 = q_seq[idx]
        p1 = q_seq[idx + 1]
        m0 = slopes[idx]
        m1 = slopes[idx + 1]

        q_val  = ((2*s3 - 3*s2 + 1) * p0
                  + (-2*s3 + 3*s2) * p1
                  + (s3 - 2*s2 + s) * (h * m0)
                  + (s3 - s2)       * (h * m1))

        qd_val = ((6*s2 - 6*s) * inv_h * p0
                  + (-6*s2 + 6*s) * inv_h * p1
                  + (3*s2 - 4*s + 1) * m0
                  + (3*s2 - 2*s)     * m1)

        qdd_val = ((12*s - 6) * inv_h2 * p0
                   + (-12*s + 6) * inv_h2 * p1
                   + (6*s - 4)   * inv_h  * m0
                   + (6*s - 2)   * inv_h  * m1)
        return q_val, qd_val, qdd_val

    return jax.vmap(_interp_one)(t_query)



def q_traj_to_traj(
    q_traj: jax.Array,
    u_traj: Optional[jax.Array],
    times: jax.Array,
    *,
    dt: float=0.001,
) -> Tuple[jax.Array, jax.Array, jax.Array, Optional[jax.Array], jax.Array]:
    '''
    Convert a joint trajectory to a trajectory of joint positions, velocities and accelerations.

    :param q_traj: Joint positions over time (..., T, nq)
    :type q_traj: jax.Array
    :param u_traj: Optional control inputs over time (..., T, nu)
    :type u_traj: Optional[jax.Array]
    :param times: Time points (..., T)
    :type times: jax.Array
    :param dt: Target time step for resampling
    :type dt: float
    :return: Trajectory of joint positions, velocities, accelerations, resampled u (or None), and uniform times
    '''
    # inputs could have batch dimensions
    # times could not have uniform spacing, so we need to resample to uniform spacing first by dt
    # together, we want to get qd, qdd from q_traj and times with uniform spacing dt
    q_traj = jnp.asarray(q_traj)
    times = jnp.asarray(times)
    if u_traj is not None:
        u_traj = jnp.asarray(u_traj)

    if q_traj.shape[-2] != times.shape[-1]:
        raise ValueError(f"Mismatch between q_traj time dimension {q_traj.shape[-2]} and times {times.shape[-1]}.")

    batch_shape = q_traj.shape[:-2]
    T, nq = q_traj.shape[-2:]
    if u_traj is not None:
        if u_traj.shape[-2] != T:
            raise ValueError(f"u_traj time dimension {u_traj.shape[-2]} must match q_traj time dimension {T}.")
        if u_traj.shape[:-2] != batch_shape:
            raise ValueError(f"u_traj batch shape {u_traj.shape[:-2]} must match q_traj batch shape {batch_shape}.")

    # Flatten batch dims for simpler vectorized processing.
    flat_q = q_traj.reshape((-1, T, nq))
    flat_t = jnp.broadcast_to(times, batch_shape + (T,)).reshape((-1, T))
    if u_traj is not None:
        nu = u_traj.shape[-1]
        flat_u = u_traj.reshape((-1, T, nu))
    else:
        nu = None

    # Decide the resampled length using the shortest trajectory (by duration).
    def _calc_target_len(t_np: np.ndarray) -> int:
        if t_np.size < 2:
            return max(1, t_np.size)
        duration = float(t_np[-1] - t_np[0])
        if duration <= 0:
            return 1
        return max(1, int(round(duration / dt)) + 1)

    flat_t_np = np.asarray(flat_t)
    target_len = min(_calc_target_len(t_np) for t_np in flat_t_np)
    if T < 4:
        raise ValueError("Need at least 4 samples to build a cubic spline trajectory.")
    if target_len < 2:
        raise ValueError("target_len must be at least 2 for cubic interpolation; check times/dt.")

    def _resample_one(q_seq, t_seq):
        # Uniform grid covering the observed duration.
        t_uniform = jnp.linspace(t_seq[0], t_seq[-1], target_len)
        q_uniform, qd_uniform, qdd_uniform = _cubic_interp_with_derivs(t_uniform, t_seq, q_seq)
        return q_uniform, qd_uniform, qdd_uniform, t_uniform

    q_uniform_flat, qd_flat, qdd_flat, t_uniform_flat = jax.vmap(_resample_one)(flat_q, flat_t)
    q_uniform_flat = q_uniform_flat.astype(q_traj.dtype)
    qd_flat = qd_flat.astype(q_traj.dtype)
    qdd_flat = qdd_flat.astype(q_traj.dtype)
    t_uniform_flat = t_uniform_flat.astype(times.dtype)

    u_uniform_flat = None
    if u_traj is not None:
        def _hold_previous(u_seq, t_seq, t_uniform):
            idx = jnp.searchsorted(t_seq, t_uniform, side="right") - 1
            idx = jnp.clip(idx, 0, u_seq.shape[0] - 1)
            return jnp.take(u_seq, idx, axis=0)
        u_uniform_flat = jax.vmap(_hold_previous)(flat_u, flat_t, t_uniform_flat).astype(u_traj.dtype)

    out_shape = batch_shape + (target_len, nq)
    q_uniform = q_uniform_flat.reshape(out_shape)
    qd = qd_flat.reshape(out_shape)
    qdd = qdd_flat.reshape(out_shape)
    t_uniform = t_uniform_flat.reshape(batch_shape + (target_len,))
    if u_uniform_flat is not None:
        u_uniform = u_uniform_flat.reshape(batch_shape + (target_len, nu))
    else:
        u_uniform = None
    return q_uniform, qd, qdd, u_uniform, t_uniform



def resample_to_uniform(
    times: jnp.ndarray,       # (..., T)
    q_traj: jnp.ndarray,      # (..., T, dof)
    target_dt: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Build a uniform grid at `target_dt` starting at t0 and length T, then
    evaluate q on that grid with a cubic Hermite (Catmull–Rom) interpolant.
    Returns (t_uniform, q_uniform), both shaped (..., T, *).
    """
    times = jnp.asarray(times)
    q_traj = jnp.asarray(q_traj)
    assert q_traj.shape[-2] == times.shape[-1], "time length mismatch"

    batch_shape = q_traj.shape[:-2]
    T, dof = q_traj.shape[-2:]

    # Flatten batch dims
    flat_q = q_traj.reshape((-1, T, dof))
    flat_t = jnp.broadcast_to(times, batch_shape + (T,)).reshape((-1, T))

    # Per-item uniform grid: t0 + k*dt, clipped to last time to avoid extrapolation
    k = jnp.arange(T, dtype=flat_t.dtype)
    def _one_grid(t_seq):
        t0 = t_seq[0]
        tN = t_seq[-1]
        tu = t0 + k * target_dt
        return jnp.clip(tu, t0, tN)
    flat_tu = jax.vmap(_one_grid)(flat_t)  # (B, T)

    # Evaluate q only (we’ll re-derive derivatives later per pipeline)
    def _resample_one(q_seq, t_seq, t_uniform):
        q_uniform, _, _ = _cubic_interp_with_derivs(t_uniform, t_seq, q_seq)
        return q_uniform
    flat_qu = jax.vmap(_resample_one)(flat_q, flat_t, flat_tu)  # (B, T, dof)

    t_uniform = flat_tu.reshape(batch_shape + (T,))
    q_uniform = flat_qu.reshape(batch_shape + (T, dof)).astype(q_traj.dtype)
    return t_uniform, q_uniform

def _take_nearest_valid_indices(
    valid_mask: jnp.ndarray,
    score: jnp.ndarray,
    *,
    num_select: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return up to `num_select` indices with the lowest score among valid entries."""
    large = jnp.asarray(valid_mask.shape[0] + 1, dtype=score.dtype)
    masked_score = jnp.where(valid_mask, score, large)
    order = jnp.argsort(masked_score)
    idx = order[:num_select]
    keep = masked_score[idx] < large
    return idx, keep


def _masked_median(values: jnp.ndarray, valid_mask: jnp.ndarray, default: float) -> jnp.ndarray:
    """Median of the masked 1D values, or `default` when no entries are valid."""
    masked = jnp.where(valid_mask, values, jnp.inf)
    sorted_vals = jnp.sort(masked)
    count = jnp.sum(valid_mask.astype(jnp.int32))
    lo = jnp.maximum((count - 1) // 2, 0)
    hi = jnp.maximum(count // 2, 0)
    med = 0.5 * (sorted_vals[lo] + sorted_vals[hi])
    return jnp.where(count > 0, med, jnp.asarray(default, dtype=values.dtype))


def _interp_at_target(
    values: jnp.ndarray,
    *,
    target_idx: int,
    before_idx: jnp.ndarray,
    before_valid: jnp.ndarray,
    center_idx: jnp.ndarray,
    center_valid: jnp.ndarray,
    after_idx: jnp.ndarray,
    after_valid: jnp.ndarray,
    base_dt: float,
) -> jnp.ndarray:
    """Interpolate a [T, D] sequence at the target time using nearby valid samples."""
    dtype = values.dtype
    base_dt_arr = jnp.asarray(base_dt, dtype=dtype)
    tau_before = (before_idx.astype(dtype) - jnp.asarray(target_idx, dtype=dtype)) * base_dt_arr
    tau_after = (after_idx.astype(dtype) - jnp.asarray(target_idx, dtype=dtype)) * base_dt_arr
    v_before = values[before_idx]
    v_center = values[center_idx]
    v_after = values[after_idx]

    denom = tau_after - tau_before
    eps = jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype)
    interp = v_before + ((-tau_before) / jnp.where(jnp.abs(denom) > eps, denom, 1.0)) * (v_after - v_before)
    nearest = jnp.where(before_valid, v_before, jnp.where(after_valid, v_after, jnp.zeros_like(v_center)))
    out = jnp.where(before_valid & after_valid, interp, nearest)
    return jnp.where(center_valid, v_center, out)


def estimate_masked_state_derivatives(
    q: jnp.ndarray,
    qd: jnp.ndarray,
    keep_mask: jnp.ndarray,
    *,
    base_dt: float = 1e-3,
    max_neighbors_each_side: int = 50,
    q_weight: float = 2.0,
    qd_weight: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Recover `(q, qd, qdd)` from masked `(q, qd)` histories on a uniform base grid.

    `max_neighbors_each_side` is interpreted as a fixed window radius in base-grid
    steps, not as a count of valid samples. For example, at 1 kHz a value of `20`
    means the local fit considers samples from `t-20 ms` to `t+20 ms` and ignores
    masked rows inside that window.

    `q_weight` and `qd_weight` tune the relative least-squares weight between
    the position and velocity equations after the velocity term has been
    unit-aligned by the local `t_scale`.
    """
    q = jnp.asarray(q)
    qd = jnp.asarray(qd)
    keep_mask = jnp.asarray(keep_mask)
    if q.ndim != 3 or qd.ndim != 3:
        raise ValueError(f"estimate_masked_state_derivatives expects [B, T, D], got q={q.shape}, qd={qd.shape}.")
    if q.shape != qd.shape:
        raise ValueError(f"q and qd must have identical shapes, got {q.shape} vs {qd.shape}.")
    if keep_mask.shape[:2] != q.shape[:2]:
        raise ValueError(f"keep_mask leading dims must match q/qd, got {keep_mask.shape} vs {q.shape}.")
    if keep_mask.ndim == 3 and keep_mask.shape[-1] != 1:
        raise ValueError(f"keep_mask trailing dim must be 1 when rank-3, got {keep_mask.shape}.")

    keep_mask_bool = keep_mask[..., 0] > 0.5 if keep_mask.ndim == 3 else keep_mask > 0.5
    n_steps = q.shape[1]
    q_dtype = q.dtype
    if q_weight < 0.0 or qd_weight < 0.0:
        raise ValueError(f"q_weight and qd_weight must be >= 0, got q_weight={q_weight}, qd_weight={qd_weight}.")
    if q_weight == 0.0 and qd_weight == 0.0:
        raise ValueError("At least one of q_weight or qd_weight must be > 0.")
    q_weight_arr = jnp.asarray(q_weight, dtype=q_dtype)
    qd_weight_arr = jnp.asarray(qd_weight, dtype=q_dtype)
    use_q = q_weight > 0.0
    use_qd = qd_weight > 0.0
    window_steps_each_side = max(int(max_neighbors_each_side), 1)
    ridge = jnp.asarray(1e-6, dtype=q_dtype)
    eye3 = jnp.eye(3, dtype=q_dtype)
    time_idx = jnp.arange(n_steps, dtype=jnp.int32)
    base_dt_arr = jnp.asarray(base_dt, dtype=q_dtype)
    local_offsets = jnp.arange(-window_steps_each_side, window_steps_each_side + 1, dtype=jnp.int32)

    def _estimate_one_batch(
        q_seq: jnp.ndarray,
        qd_seq: jnp.ndarray,
        keep_seq: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        def _estimate_at_index(target_idx: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            local_idx_raw = target_idx + local_offsets
            local_in_bounds = (local_idx_raw >= 0) & (local_idx_raw < n_steps)
            local_idx = jnp.clip(local_idx_raw, 0, n_steps - 1)
            local_valid = local_in_bounds & keep_seq[local_idx]
            local_pos = jnp.arange(local_offsets.shape[0], dtype=jnp.int32)
            center_pos = jnp.asarray(window_steps_each_side, dtype=jnp.int32)

            before_pos = jnp.max(jnp.where(local_valid & (local_offsets < 0), local_pos, -1))
            before_valid = before_pos >= 0
            before_idx = local_idx[jnp.clip(before_pos, 0, local_offsets.shape[0] - 1)]

            center_valid = local_valid[center_pos]
            center_idx = local_idx[center_pos]

            after_sentinel = jnp.asarray(local_offsets.shape[0], dtype=jnp.int32)
            after_pos = jnp.min(jnp.where(local_valid & (local_offsets > 0), local_pos, after_sentinel))
            after_valid = after_pos < after_sentinel
            after_idx = local_idx[jnp.clip(after_pos, 0, local_offsets.shape[0] - 1)]

            local_score_large = jnp.asarray(local_offsets.shape[0] + 1, dtype=jnp.int32)
            local_score = jnp.where(local_valid, jnp.abs(local_offsets).astype(jnp.int32), local_score_large)
            overall_pos = jnp.argsort(local_score)[:3]
            overall_valid = local_score[overall_pos] < local_score_large
            overall_idx = local_idx[overall_pos]

            tau = local_offsets.astype(q_dtype) * base_dt_arr
            abs_tau_nonzero = jnp.abs(tau)
            nonzero_valid = local_valid & (abs_tau_nonzero > 0)
            t_scale = jnp.maximum(base_dt_arr, _masked_median(abs_tau_nonzero, nonzero_valid, float(base_dt)))
            s = tau / t_scale

            q_local = q_seq[local_idx]
            qd_local = qd_seq[local_idx]
            taper = 1.0 / (1.0 + jnp.abs(s))
            w_q = jnp.where(local_valid, q_weight_arr * taper, 0.0).astype(q_dtype)
            w_qd = jnp.where(local_valid, qd_weight_arr * taper, 0.0).astype(q_dtype)

            a_q = jnp.stack(
                [
                    jnp.ones_like(s, dtype=q_dtype),
                    s.astype(q_dtype),
                    0.5 * s.astype(q_dtype) * s.astype(q_dtype),
                ],
                axis=-1,
            )
            a_qd = jnp.stack(
                [
                    jnp.zeros_like(s, dtype=q_dtype),
                    jnp.ones_like(s, dtype=q_dtype),
                    s.astype(q_dtype),
                ],
                axis=-1,
            )
            a = jnp.concatenate([a_q, a_qd], axis=0)
            y = jnp.concatenate([q_local, qd_local * t_scale], axis=0)
            w = jnp.concatenate([w_q, w_qd], axis=0)
            sqrt_w = jnp.sqrt(w)[:, None]
            a_w = a * sqrt_w
            y_w = y * sqrt_w

            ata = a_w.T @ a_w + ridge * eye3
            aty = a_w.T @ y_w
            coeff = jnp.linalg.solve(ata, aty)
            q_fit_main = coeff[0]
            qd_fit_main = coeff[1] / t_scale
            qdd_fit_main = coeff[2] / (t_scale * t_scale)
            n_local_valid = jnp.sum(local_valid.astype(jnp.int32))
            if use_q and use_qd:
                fit_ok = (n_local_valid >= 2) & jnp.any(nonzero_valid)
            elif use_q:
                fit_ok = n_local_valid >= 3
            else:
                fit_ok = (n_local_valid >= 2) & jnp.any(nonzero_valid)

            q_interp = _interp_at_target(
                q_seq,
                target_idx=target_idx,
                before_idx=before_idx,
                before_valid=before_valid,
                center_idx=center_idx,
                center_valid=center_valid,
                after_idx=after_idx,
                after_valid=after_valid,
                base_dt=base_dt,
            )

            fd_idx = overall_idx[:2]
            fd_valid = overall_valid[:2]
            fd_tau = (fd_idx.astype(q_dtype) - target_idx.astype(q_dtype)) * base_dt_arr
            qd_fd = qd_seq[fd_idx]
            fd_denom = fd_tau[1] - fd_tau[0]
            fd_ok = (jnp.all(fd_valid) & (jnp.abs(fd_denom) > 0)) if use_qd else jnp.asarray(False)
            qdd_fit_fd = (qd_fd[1] - qd_fd[0]) / jnp.where(jnp.abs(fd_denom) > 0, fd_denom, 1.0)
            qd_fit_fd = qd_fd[0] + (-fd_tau[0]) * qdd_fit_fd

            q_fit_q = q_interp
            qd_fit_q = _interp_at_target(
                qd_seq,
                target_idx=target_idx,
                before_idx=before_idx,
                before_valid=before_valid,
                center_idx=center_idx,
                center_valid=center_valid,
                after_idx=after_idx,
                after_valid=after_valid,
                base_dt=base_dt,
            )

            q3_idx = overall_idx
            q3_valid = overall_valid
            tau3 = (q3_idx.astype(q_dtype) - target_idx.astype(q_dtype)) * base_dt_arr
            a3 = jnp.stack(
                [
                    jnp.ones_like(tau3, dtype=q_dtype),
                    tau3,
                    0.5 * tau3 * tau3,
                ],
                axis=-1,
            )
            y3 = q_seq[q3_idx]
            w3 = q3_valid.astype(q_dtype)
            a3_w = a3 * w3[:, None]
            y3_w = y3 * w3[:, None]
            coeff_q = jnp.linalg.solve(a3_w.T @ a3_w + ridge * eye3, a3_w.T @ y3_w)
            q_fit_q = coeff_q[0]
            qd_fit_q = coeff_q[1]
            qdd_fit_q = coeff_q[2]
            q_fallback_ok = (jnp.sum(q3_valid.astype(jnp.int32)) >= 3) if use_q else jnp.asarray(False)

            qd_interp = _interp_at_target(
                qd_seq,
                target_idx=target_idx,
                before_idx=before_idx,
                before_valid=before_valid,
                center_idx=center_idx,
                center_valid=center_valid,
                after_idx=after_idx,
                after_valid=after_valid,
                base_dt=base_dt,
            )

            q_out_main = q_fit_main if use_q else q_interp

            q_out = jnp.where(
                fit_ok,
                q_out_main,
                jnp.where(q_fallback_ok, q_fit_q, q_interp),
            )
            qd_out = jnp.where(
                fit_ok,
                qd_fit_main,
                jnp.where(
                    fd_ok,
                    qd_fit_fd,
                    jnp.where(
                        q_fallback_ok,
                        qd_fit_q,
                        jnp.where(jnp.asarray(use_qd), qd_interp, jnp.zeros_like(qd_fit_main)),
                    ),
                ),
            )
            qdd_out = jnp.where(
                fit_ok,
                qdd_fit_main,
                jnp.where(fd_ok, qdd_fit_fd, jnp.where(q_fallback_ok, qdd_fit_q, jnp.zeros_like(qdd_fit_main))),
            )
            return q_out, qd_out, qdd_out

        return jax.vmap(_estimate_at_index)(time_idx)

    return jax.vmap(_estimate_one_batch)(q, qd, keep_mask_bool)

# ============================================================
# 3) Cubic smoothing (Whittaker / second-derivative penalty)
#    Solve: (I + alpha_eff * D2^T D2) q_smooth = q
# ============================================================

def _conv1d_reflect_k5(x: jnp.ndarray) -> jnp.ndarray:
    """
    1D symmetric convolution along time with kernel [1, -4, 6, -4, 1] (k5),
    using reflect padding. x: (..., T, C) or (T,) -> (..., T, C) / (T,).
    """
    k5 = jnp.array([1., -4., 6., -4., 1.], dtype=x.dtype)  # symmetric
    W = k5.shape[0]
    m = (W - 1) // 2
    x_in = jnp.asarray(x)
    added_channel = False
    if x_in.ndim < 2:
        # Ensure we have an explicit channel dim so axis=-2 is valid.
        x_in = x_in[..., None]
        added_channel = True

    T = x_in.shape[-2]
    t = jnp.arange(T)
    offs = jnp.arange(-m, m + 1)
    idx = _reflect_indices(t[:, None] + offs[None, :], 0, T - 1)  # (T, W)
    xw = jnp.take(x_in, idx, axis=-2)                            # (..., T, W, C)
    y = jnp.tensordot(xw, k5, axes=([ -2 ], [0]))                # (..., T, C)
    if added_channel:
        y = jnp.squeeze(y, axis=-1)
    return y

def _apply_A_whittaker(x: jnp.ndarray, alpha_eff: float) -> jnp.ndarray:
    """A x = x + alpha_eff * (D2^T D2) x, implemented via k5 convolution."""
    return x + alpha_eff * _conv1d_reflect_k5(x)

def _cg_solve_whittaker(b: jnp.ndarray, alpha_eff: float, maxiter: int = 200, tol: float = 1e-6) -> jnp.ndarray:
    """
    Conjugate Gradient solve for (I + alpha_eff * D2^T D2) x = b.
    b: (T,)  -> returns x: (T,)
    Fully JAX-compatible (jit/vmap).
    """
    def body_fun(state, _):
        x, r, p, rsold = state
        Ap = _apply_A_whittaker(p, alpha_eff)
        alpha = rsold / (jnp.dot(p, Ap) + 1e-30)
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        rsnew = jnp.dot(r_new, r_new)
        beta = rsnew / (rsold + 1e-30)
        p_new = r_new + beta * p
        return (x_new, r_new, p_new, rsnew), rsnew

    # Initial
    x0 = jnp.zeros_like(b)
    r0 = b - _apply_A_whittaker(x0, alpha_eff)
    p0 = r0
    rs0 = jnp.dot(r0, r0)
    state = (x0, r0, p0, rs0)

    # Fixed-iter loop; we’ll stop “effectively” by not changing much once converged.
    state, rs_hist = jax.lax.scan(body_fun, state, None, length=maxiter)
    x, r, p, rs = state
    # Optional: single pass “soft stop” by choosing the best iterate via minimal residual
    # (Here we just return the final x; for most ID lengths it converges well before maxiter.)
    return x

def cubic_smoothing_whittaker(
    q: jnp.ndarray,   # (..., T, dof) on a uniform grid
    dt: float,
    alpha: float = 1e-1,
    maxiter: int = 200,
    tol: float = 1e-6
) -> jnp.ndarray:
    """
    Global smoother approximating a cubic smoothing spline on a uniform grid.
    Solves (I + alpha_eff * D2^T D2) q_smooth = q with reflect edges.

    alpha is dimensionless; internally alpha_eff = alpha / dt^3,
    which makes the smoothing roughly sampling-rate independent.

    Returns q_smooth with the same shape as q.
    """
    alpha_eff = alpha / (dt**3)
    # Flatten batch+DOF to run CG per series
    *batch, T, dof = q.shape
    x = q.reshape((-1, T, dof))
    # Solve each channel independently
    def solve_one(series_1c):
        # series_1c: (T,)
        return _cg_solve_whittaker(series_1c, alpha_eff, maxiter=maxiter, tol=tol)
    solve_channel = jax.vmap(solve_one, in_axes=1, out_axes=1)       # over dof
    solve_batch   = jax.vmap(solve_channel, in_axes=0, out_axes=0)   # over batch
    qs = solve_batch(x)                                              # (B, T, dof)
    return qs.reshape(tuple(batch) + (T, dof)).astype(q.dtype)
