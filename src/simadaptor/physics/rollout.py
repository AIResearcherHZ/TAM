import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax, vmap, jit, random
import einops

import mujoco
import mujoco.mjx as mjx

import simadaptor.core.structs as structs
import simadaptor.physics.actuator as actuator_util

INTERP = "catmull_rom"         # "catmull_rom" | "linear"
# -----------------------
# Utilities
# -----------------------
def deg_to_rad(x): return x * jnp.pi / 180.0


def catmull_rom_spline(waypoints: jnp.ndarray, T: int, duration:float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Uniform Catmull–Rom spline through waypoints in joint space.
    Returns q_ref (T, D), qd_ref (T, D).
    """
    # waypoints: (M, D)
    M, D = waypoints.shape
    # Duplicate endpoints for boundary conditions
    P = jnp.concatenate([waypoints[0:1], waypoints, waypoints[-1:]], axis=0)  # (M+2, D)
    # time parameterization: sample T points across M-1 segments
    t = jnp.linspace(0.0, M - 1, T, endpoint=False)  # [0, M-1)
    seg = jnp.clip(jnp.floor(t).astype(jnp.int32), 0, M - 2)  # which segment
    local_t = t - seg  # [0,1) within segment

    # For segment i, control points are: P[i], P[i+1], P[i+2], P[i+3] in the padded array
    # But seg is in [0, M-2] corresponding to raw waypoint indices. P is shifted by +1.
    i0 = seg + 0  # corresponds to P[i], which is waypoints[i-1] due to pad
    i1 = seg + 1
    i2 = seg + 2
    i3 = seg + 3

    # Gather control points
    P0 = P[i0, :]
    P1 = P[i1, :]
    P2 = P[i2, :]
    P3 = P[i3, :]

    # Catmull-Rom basis (centripetal alpha=0.5 could be used; here uniform for simplicity)
    tt = local_t
    tt2 = tt * tt
    tt3 = tt2 * tt

    # Position
    q = 0.5 * (
        (2.0 * P1)
        + (-P0 + P2) * tt[:, None]
        + (2.0 * P0 - 5.0 * P1 + 4.0 * P2 - P3) * tt2[:, None]
        + (-P0 + 3.0 * P1 - 3.0 * P2 + P3) * tt3[:, None]
    )

    # Velocity w.r.t. segment param (not physical time yet)
    dq_dtau = 0.5 * (
        (-P0 + P2)
        + 2.0 * (2.0 * P0 - 5.0 * P1 + 4.0 * P2 - P3) * tt[:, None]
        + 3.0 * (-P0 + 3.0 * P1 - 3.0 * P2 + P3) * tt2[:, None]
    )
    # Map tau to real time: each segment spans duration SEG_DUR = 5s/(M-1)
    SEG_DUR = duration / (M - 1)
    qd = dq_dtau / SEG_DUR
    return q, qd


def linear_interp(waypoints: jnp.ndarray, T: int, duration:float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Piecewise linear through waypoints across 5 s.
    """
    M, D = waypoints.shape
    t = jnp.linspace(0.0, duration, T, endpoint=False)
    knot_times = jnp.linspace(0.0, duration, M)
    # For each t, find the segment index k such that knot[k] <= t < knot[k+1]
    idx = jnp.clip(jnp.searchsorted(knot_times, t, side="right") - 1, 0, M - 2)
    t0 = knot_times[idx]
    t1 = knot_times[idx + 1]
    w0 = (t1 - t) / (t1 - t0 + 1e-9)
    w1 = 1.0 - w0
    q = waypoints[idx] * w0[:, None] + waypoints[idx + 1] * w1[:, None]
    qd = (waypoints[idx + 1] - waypoints[idx]) / (t1 - t0 + 1e-9)
    return q, qd

def build_traj_from_waypoints(waypoints: jnp.ndarray, T: int, duration:float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    if INTERP == "catmull_rom":
        return catmull_rom_spline(waypoints, T, duration)
    else:
        return linear_interp(waypoints, T, duration)

def guess_arm_joint_ids(m: mujoco.MjModel, dof_target=7) -> np.ndarray:
    # Try by known names
    names = [m.joint(i).name for i in range(m.njnt)]
    idx = []
    for i, nm in enumerate(names):
        if nm is None:
            continue
        if nm.startswith("panda_joint"):
            idx.append(i)
    if len(idx) >= dof_target:
        return np.array(idx[:dof_target], dtype=np.int32)
    # Fallback: take first 'dof_target' hinge joints
    hinge_ids = [i for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
    return np.array(hinge_ids[:dof_target], dtype=np.int32)

# -----------------------
# MJX simulation (JAX)
# -----------------------


def joint_indices_to_dofs(m: mujoco.MjModel, joint_ids: np.ndarray) -> np.ndarray:
    """Return velocity dof indices for those joints (for hinge joints, 1 dof each)."""
    # In MuJoCo, for hinge joints, dof address = jnt_dofadr[jid]
    dof_idx = [m.jnt_dofadr[jid] for jid in joint_ids]
    return np.array(dof_idx, dtype=np.int32)


def _sample_external_force_impulse_components(
    key: jax.Array,
    *,
    batch_n: int,
    total_steps: int,
    dt: float,
    num_impulses: int,
    magnitude_min_n: float,
    magnitude_max_n: float,
    duration_min_s: float,
    duration_max_s: float,
    dtype=jnp.float32,
) -> jax.Array:
    """Sample per-impulse world-frame force components."""
    if total_steps <= 0:
        return jnp.zeros((batch_n, max(int(num_impulses), 0), 0, 3), dtype=dtype)
    if num_impulses <= 0:
        return jnp.zeros((batch_n, 0, total_steps, 3), dtype=dtype)
    if magnitude_max_n <= 0.0:
        return jnp.zeros((batch_n, num_impulses, total_steps, 3), dtype=dtype)

    mag_lo = float(min(magnitude_min_n, magnitude_max_n))
    mag_hi = float(max(magnitude_min_n, magnitude_max_n))
    dur_lo_s = float(min(duration_min_s, duration_max_s))
    dur_hi_s = float(max(duration_min_s, duration_max_s))
    dur_lo_steps = max(2, int(np.round(dur_lo_s / dt)))
    dur_hi_steps = max(dur_lo_steps, int(np.round(dur_hi_s / dt)))

    k_dir, k_mag, k_dur, k_ctr = random.split(key, 4)
    dirs = random.normal(k_dir, (batch_n, num_impulses, 3), dtype=dtype)
    dirs = dirs / jnp.maximum(jnp.linalg.norm(dirs, axis=-1, keepdims=True), 1e-6)
    mags = random.uniform(k_mag, (batch_n, num_impulses, 1), minval=mag_lo, maxval=mag_hi, dtype=dtype)
    durs = random.randint(
        k_dur,
        (batch_n, num_impulses),
        minval=dur_lo_steps,
        maxval=dur_hi_steps + 1,
        dtype=jnp.int32,
    )

    ctr_keys = random.split(k_ctr, num_impulses)
    centers = []
    for i in range(num_impulses):
        lo = (i * total_steps) // num_impulses
        hi = ((i + 1) * total_steps) // num_impulses
        if hi <= lo:
            hi = min(total_steps, lo + 1)
        c = random.randint(ctr_keys[i], (batch_n,), minval=lo, maxval=hi, dtype=jnp.int32)
        centers.append(c)
    centers = jnp.stack(centers, axis=1)

    t = jnp.arange(total_steps, dtype=jnp.int32)[None, None, :]
    starts = centers - durs // 2
    rel = t - starts[..., None]
    valid = (rel >= 0) & (rel < durs[..., None])
    denom = jnp.maximum(durs[..., None] - 1, 1).astype(dtype)
    phase = rel.astype(dtype) / denom
    envelope = 0.5 * (1.0 - jnp.cos(2.0 * jnp.pi * phase))
    envelope = jnp.where(valid, envelope, 0.0)

    pulse = envelope[..., None] * dirs[:, :, None, :] * mags[:, :, None, :]
    return pulse.astype(dtype)


def sample_external_force_impulses(
    key: jax.Array,
    *,
    batch_n: int,
    total_steps: int,
    dt: float,
    num_impulses: int,
    magnitude_min_n: float,
    magnitude_max_n: float,
    duration_min_s: float,
    duration_max_s: float,
    dtype=jnp.float32,
) -> jax.Array:
    """
    Sample smooth random force impulses in world frame.

    Returns:
      force_seq: (batch_n, total_steps, 3), force only (N), torque is implicitly zero.
    """
    pulse = _sample_external_force_impulse_components(
        key,
        batch_n=batch_n,
        total_steps=total_steps,
        dt=dt,
        num_impulses=num_impulses,
        magnitude_min_n=magnitude_min_n,
        magnitude_max_n=magnitude_max_n,
        duration_min_s=duration_min_s,
        duration_max_s=duration_max_s,
        dtype=dtype,
    )
    return jnp.sum(pulse, axis=1).astype(dtype)  # (B,T,3)


def sample_external_force_local_positions(
    key: jax.Array,
    *,
    batch_n: int,
    num_impulses: int,
    position_min_local_m: jax.Array,
    position_max_local_m: jax.Array,
    dtype=jnp.float32,
) -> jax.Array:
    """Sample one local application point per impulse in body-local coordinates."""
    if num_impulses <= 0:
        return jnp.zeros((batch_n, 0, 3), dtype=dtype)
    pos_min = jnp.asarray(position_min_local_m, dtype=dtype).reshape((3,))
    pos_max = jnp.asarray(position_max_local_m, dtype=dtype).reshape((3,))
    pos_lo = jnp.minimum(pos_min, pos_max)
    pos_hi = jnp.maximum(pos_min, pos_max)
    return random.uniform(
        key,
        (batch_n, num_impulses, 3),
        minval=pos_lo,
        maxval=pos_hi,
        dtype=dtype,
    )


def rollout_one(sys:mjx.Model,
                q_ref: jnp.ndarray, qd_ref: jnp.ndarray,
                actuator_fn, arm_dof_idx: jnp.ndarray, batched_rollout_params:structs.RolloutParams,
                rng: jnp.ndarray,
                initial_actuator_carry:Optional[Any]=None,
                reset_interval:Optional[float]=None, adaptor_seq_length:int=1, base_rollout=None,
                external_force_ee: Optional[jnp.ndarray] = None,
                external_force_body_id: Optional[int] = None,
                external_force_impulse_terms: Optional[jnp.ndarray] = None,
                external_force_local_positions: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Simulate one trajectory over T_STEPS with PD control.
    Returns (Q, QD, U) with shapes:
      Q:  (T, sys.nv)     # we log generalized positions for all dofs (note: qpos may include quats for free joints;
                          # here we log qvel-sized q (angles for hinges), so use qvel-sized projection if needed)
      QD: (T, sys.nv)
      U:  (T, sys.nu)
    """
    n_batch = batched_rollout_params.kp.shape[0] if batched_rollout_params.kp is not None else batched_rollout_params.dof_damping.shape[0]
    state = mjx.make_data(sys)
    state = jax.tree.map(lambda x: jnp.broadcast_to(x, (n_batch,) + x.shape), state)
    reset_state = state
    total_steps = q_ref.shape[-2] if q_ref is not None else base_rollout['q'].shape[-2]
    window_len = max(int(adaptor_seq_length), 1)
    scan_start = window_len - 1 if base_rollout is not None else 0
    if scan_start >= total_steps:
        raise ValueError(f"scan_start ({scan_start}) must be smaller than rollout length ({total_steps})")
    T_STEPS = total_steps - scan_start  # number of steps to simulate after seeding the initial window
    use_external_force = external_force_body_id is not None and int(external_force_body_id) >= 0

    # Build explicit mapping between selected dof order (arm_dof_idx) and actuator order.
    arm_dof_idx_arr = jnp.asarray(arm_dof_idx, dtype=jnp.int32).reshape((-1,))
    ctrl_dim = int(arm_dof_idx_arr.shape[0])

    actuator_trnid = jnp.asarray(sys.actuator_trnid, dtype=jnp.int32)
    jnt_dofadr = jnp.asarray(sys.jnt_dofadr, dtype=jnp.int32)
    act_jnt_id = actuator_trnid[:, 0]
    act_jnt_id_clamped = jnp.clip(act_jnt_id, 0, int(sys.njnt) - 1)
    act_dof_id = jnt_dofadr[act_jnt_id_clamped]
    valid = (
        (act_jnt_id >= 0)
        & (act_jnt_id < int(sys.njnt))
        & (act_dof_id >= 0)
        & (act_dof_id < int(sys.nv))
    )
    fallback_dof_id = jnp.minimum(
        jnp.arange(int(sys.nu), dtype=jnp.int32),
        jnp.asarray(max(0, int(sys.nv) - 1), dtype=jnp.int32),
    )
    actuator_dof_abs_full = jnp.where(valid, act_dof_id, fallback_dof_id)
    actuator_dof_abs_ctrl = actuator_dof_abs_full[:ctrl_dim]

    # Map each actuator-associated absolute dof id to local index in arm_dof_idx.
    local_guess = jnp.clip(
        jnp.searchsorted(arm_dof_idx_arr, actuator_dof_abs_ctrl, side="left"),
        0,
        max(0, ctrl_dim - 1),
    )
    matched = arm_dof_idx_arr[local_guess] == actuator_dof_abs_ctrl
    fallback_local = jnp.minimum(
        jnp.arange(ctrl_dim, dtype=jnp.int32),
        jnp.asarray(max(0, ctrl_dim - 1), dtype=jnp.int32),
    )
    actuator_to_local_ctrl = jnp.where(matched, local_guess, fallback_local)

    def _to_actuator_order(x):
        if x is None:
            return None
        x = jnp.asarray(x)
        return x[..., actuator_to_local_ctrl]

    def _to_dof_order(x):
        if x is None:
            return None
        x = jnp.asarray(x)
        out = jnp.zeros_like(x)
        out = out.at[..., actuator_to_local_ctrl].set(x)
        return out

    def init_window(val, initial_seq=None, name: str = "window"):
        val = jnp.asarray(val)
        if initial_seq is not None:
            initial_seq = jnp.asarray(initial_seq)
            if initial_seq.shape[-2] < window_len:
                raise ValueError(f"base_rollout['{name}'] shorter than adaptor_seq_length={window_len}")
            return initial_seq[..., :window_len, :]
        return jnp.repeat(val[..., None, :], window_len, axis=-2)

    def push_window(window, new_val):
        if window is None:
            return None
        new_val = new_val[..., None, :]
        updated = jnp.concatenate([window[..., 1:, :], new_val], axis=-2)
        return updated
        
    # set initial q and qd
    if base_rollout is not None:
        state = state.replace(
            qpos=state.qpos.at[:,arm_dof_idx].set(base_rollout['q'][:,scan_start]),
            qvel=state.qvel.at[:,arm_dof_idx].set(base_rollout['qd'][:,scan_start]),
            time=state.time.at[:].set(base_rollout['times'][:,scan_start])
        )
    else:
        state = state.replace(
            qpos=state.qpos.at[:,arm_dof_idx].set(q_ref[:,0]),
            qvel=state.qvel.at[:,arm_dof_idx].set(qd_ref[:,0]),
        )
    # prepare reference sequences
    if base_rollout is not None:
        u_ref_seq = base_rollout['u'] if 'u' in base_rollout else None
        if 'q_ref' in base_rollout:
            q_ref_seq = base_rollout['q_ref']
            qd_ref_seq = base_rollout['qd_ref']
        else:
            q_ref_seq = None if u_ref_seq is not None else q_ref
            qd_ref_seq = None if u_ref_seq is not None else qd_ref
    else:
        u_ref_seq = None
        q_ref_seq = q_ref
        qd_ref_seq = qd_ref
    assert (q_ref_seq is not None) or (u_ref_seq is not None), "Provide either reference states or reference torques."
    u_ref_seq = _to_actuator_order(u_ref_seq)
    q_ref_seq = _to_actuator_order(q_ref_seq)
    qd_ref_seq = _to_actuator_order(qd_ref_seq)

    # initialize observation/reference windows
    base_q_window = _to_actuator_order(base_rollout['q']) if base_rollout is not None else None
    base_qd_window = _to_actuator_order(base_rollout['qd']) if base_rollout is not None else None
    base_u_window = (
        _to_actuator_order(base_rollout['u'])
        if (base_rollout is not None and 'u' in base_rollout)
        else None
    )
    q_window = init_window(_to_actuator_order(state.qpos[..., arm_dof_idx]), base_q_window, name="q")
    qd_window = init_window(_to_actuator_order(state.qvel[..., arm_dof_idx]), base_qd_window, name="qd")
    if u_ref_seq is not None:
        u_ref_window = init_window(u_ref_seq[:, 0], base_u_window, name="u")
        # calculate ctrl gt
        def _gt_torque(rollout_params:structs.RolloutParams, q, qd, qacc):
            return actuator_util.calculate_gt_torque(rollout_params.set_mjx_model(sys), q, qd, 
                                                         actuator_params=rollout_params.actuator_params, qacc=qacc)
        if adaptor_seq_length > 1:
            ctrl_window = jax.vmap(
                jax.vmap(_gt_torque, (None, 0, 0, 0))
                    )(batched_rollout_params, q_window, qd_window, base_rollout['qdd'][:,:adaptor_seq_length])  # (n_batch, T)
        else:
            ctrl_window = u_ref_window
    else:
        u_ref_window = None
        ctrl_window = jnp.zeros_like(q_window)
    u_ref_seq_scan = u_ref_seq[:, scan_start:] if u_ref_seq is not None else None
    q_ref_seq_scan = q_ref_seq[:, scan_start:] if q_ref_seq is not None else None
    qd_ref_seq_scan = qd_ref_seq[:, scan_start:] if qd_ref_seq is not None else None
    q_noise_std_ctl = (
        _to_actuator_order(batched_rollout_params.q_noise_std)
        if batched_rollout_params.q_noise_std is not None
        else None
    )
    dq_noise_std_ctl = (
        _to_actuator_order(batched_rollout_params.dq_noise_std)
        if batched_rollout_params.dq_noise_std is not None
        else None
    )
    if external_force_impulse_terms is not None:
        external_force_impulse_terms = jnp.asarray(external_force_impulse_terms, dtype=state.qpos.dtype)
        if (
            external_force_impulse_terms.ndim != 4
            or external_force_impulse_terms.shape[0] != n_batch
            or external_force_impulse_terms.shape[2] != total_steps
            or external_force_impulse_terms.shape[-1] != 3
        ):
            raise ValueError(
                "external_force_impulse_terms must have shape "
                f"(B={n_batch}, Nimp, T={total_steps}, 3); got {external_force_impulse_terms.shape}"
            )
        if external_force_local_positions is None:
            raise ValueError(
                "external_force_local_positions is required when external_force_impulse_terms is provided."
            )
        external_force_local_positions = jnp.asarray(
            external_force_local_positions, dtype=state.qpos.dtype
        )
        if external_force_local_positions.shape != (
            n_batch,
            external_force_impulse_terms.shape[1],
            3,
        ):
            raise ValueError(
                "external_force_local_positions must have shape "
                f"(B={n_batch}, Nimp={external_force_impulse_terms.shape[1]}, 3); "
                f"got {external_force_local_positions.shape}"
            )
        external_force_seq_scan = None
        external_force_impulse_terms_scan = external_force_impulse_terms[:, :, scan_start:]
        external_force_wrench_dim = 6
    else:
        if external_force_local_positions is not None:
            raise ValueError(
                "external_force_local_positions requires external_force_impulse_terms."
            )
        if external_force_ee is None:
            external_force_ee = jnp.zeros((n_batch, total_steps, 3), dtype=state.qpos.dtype)
        else:
            external_force_ee = jnp.asarray(external_force_ee, dtype=state.qpos.dtype)
            if (
                external_force_ee.ndim != 3
                or external_force_ee.shape[0] != n_batch
                or external_force_ee.shape[1] != total_steps
                or external_force_ee.shape[-1] not in (3, 6)
            ):
                raise ValueError(
                    f"external_force_ee must have shape (B={n_batch}, T={total_steps}, 3|6); "
                    f"got {external_force_ee.shape}"
                )
        external_force_seq_scan = external_force_ee[:, scan_start:]
        external_force_impulse_terms_scan = None
        external_force_wrench_dim = int(external_force_ee.shape[-1])
    actuator_carry0 = initial_actuator_carry if initial_actuator_carry is not None else jnp.zeros((n_batch, arm_dof_idx.shape[0]), dtype=state.ctrl.dtype)

    def mjx_step(sys, state, rollout_params:structs.RolloutParams) -> mjx.Data:
        qpos_for_act = state.qpos[..., actuator_dof_abs_full]
        qvel_for_act = state.qvel[..., actuator_dof_abs_full]
        tau_eff = actuator_util.actuator_model(state.ctrl, qpos_for_act, qvel_for_act, rollout_params.actuator_params)
        tau_eff = jnp.zeros(sys.nu, dtype=tau_eff.dtype).at[:tau_eff.shape[-1]].set(tau_eff)
        state = state.replace(ctrl=tau_eff)
        model_mjx = rollout_params.set_mjx_model(sys)
        new_state = mjx.step(model_mjx, state)  # for side effects
        return new_state

    def mjx_forward_kinematics(sys, state, rollout_params:structs.RolloutParams) -> mjx.Data:
        model_mjx = rollout_params.set_mjx_model(sys)
        return mjx.forward(
            model_mjx,
            state.replace(xfrc_applied=jnp.zeros_like(state.xfrc_applied)),
        )

    def _take_step(seq, t, axis: int = 1):
        """Safely slice time step t from a [B, T, ...] array inside scan."""
        if seq is None:
            return None
        sl = lax.dynamic_slice_in_dim(seq, t, 1, axis=axis)
        return jnp.squeeze(sl, axis=axis)

    def _set_body_wrench(xfrc_applied: jnp.ndarray, wrench: jnp.ndarray) -> jnp.ndarray:
        if wrench.shape[-1] == 3:
            return xfrc_applied.at[:, external_force_body_id, :3].set(
                wrench.astype(xfrc_applied.dtype)
            )
        if wrench.shape[-1] == 6:
            return xfrc_applied.at[:, external_force_body_id, :].set(
                wrench.astype(xfrc_applied.dtype)
            )
        raise ValueError(f"external force must have trailing size 3 or 6, got {wrench.shape}")

    def _compute_wrench_from_impulses(
        state_for_step: mjx.Data,
        force_terms_t: jnp.ndarray,
    ) -> jnp.ndarray:
        body_rot = jnp.asarray(
            state_for_step.xmat[:, external_force_body_id], dtype=state_for_step.qpos.dtype
        )
        if body_rot.ndim == 2:
            body_rot = body_rot.reshape((body_rot.shape[0], 3, 3))
        body_ipos_local = (
            jnp.asarray(
                batched_rollout_params.body_ipos[:, external_force_body_id, :],
                dtype=state_for_step.qpos.dtype,
            )
            if batched_rollout_params.body_ipos is not None
            else jnp.broadcast_to(
                jnp.asarray(sys.body_ipos[external_force_body_id], dtype=state_for_step.qpos.dtype),
                (n_batch, 3),
            )
        )
        lever_local = external_force_local_positions - body_ipos_local[:, None, :]
        lever_world = jnp.einsum("bij,bkj->bki", body_rot, lever_local)
        torque_terms_t = jnp.cross(lever_world, force_terms_t, axis=-1)
        force_t = jnp.sum(force_terms_t, axis=1)
        torque_t = jnp.sum(torque_terms_t, axis=1)
        return jnp.concatenate([force_t, torque_t], axis=-1)

    @jax.checkpoint
    def step_fn(carry, t):
        state, actuator_carry, rng, q_window, qd_window, u_ref_window, ctrl_window = carry
        global_t = t + scan_start
        if reset_interval is not None:
            assert base_rollout is not None, "base_rollout must be provided when using reset_interval"
            reset_mask = (global_t % reset_interval == 0)
            reset_state_ = reset_state.replace(
                qpos=state.qpos.at[:, arm_dof_idx].set(base_rollout['q'][:,global_t]),
                qvel=state.qvel.at[:, arm_dof_idx].set(base_rollout['qd'][:,global_t]),
                time=state.time.at[:].set(base_rollout['times'][:,global_t])
            )
            reset_state_ = jax.lax.stop_gradient(reset_state_)
            state = jax.tree.map(lambda a, b: jnp.where(reset_mask, b, a), state, reset_state_)
        else:
            reset_mask = jnp.array(False)
        q_obs_dof = state.qpos[..., arm_dof_idx]
        qd_obs_dof = state.qvel[..., arm_dof_idx]
        q_obs = _to_actuator_order(q_obs_dof)
        qd_obs = _to_actuator_order(qd_obs_dof)
        if q_noise_std_ctl is not None:
            rng, q_obs_key = jax.random.split(rng)
            rng, qd_obs_key = jax.random.split(rng)
            q_obs = q_obs + q_noise_std_ctl * random.normal(q_obs_key, q_obs.shape)
            qd_obs = qd_obs + dq_noise_std_ctl * random.normal(qd_obs_key, qd_obs.shape)
        q_window = push_window(q_window, q_obs)
        qd_window = push_window(qd_window, qd_obs)

        u_ref_window = push_window(u_ref_window, _take_step(u_ref_seq_scan, t)) if u_ref_seq_scan is not None else None
        
        q_ref_t = _take_step(q_ref_seq_scan, t) if q_ref_seq_scan is not None else None
        qd_ref_t = _take_step(qd_ref_seq_scan, t) if qd_ref_seq_scan is not None else None

        rng, rng_action = jax.random.split(rng)
        u_input_window = jnp.concat([ctrl_window[...,1:,:], u_ref_window[..., -1:, :]], axis=-2) if u_ref_window is not None else ctrl_window
        # u_input_window = u_ref_window
        ctrl, delta_tau, actuator_carry = actuator_fn(q_window, qd_window, q_ref_t, qd_ref_t, rng_action, actuator_carry, u_ref=u_input_window)
        ctrl_window = push_window(ctrl_window, ctrl)
        ctrl_full = jnp.zeros_like(state.ctrl).at[..., :ctrl.shape[-1]].set(ctrl)
        u_dof = _to_dof_order(ctrl)
        delta_tau_dof = _to_dof_order(delta_tau) if delta_tau is not None else None
        state_for_step = state.replace(ctrl=ctrl_full)
        if external_force_impulse_terms_scan is not None:
            force_terms_t = _take_step(external_force_impulse_terms_scan, t, axis=2)
            state_for_force = jax.vmap(mjx_forward_kinematics, in_axes=(None, 0, 0))(
                sys,
                state_for_step,
                batched_rollout_params,
            )
            force_t = _compute_wrench_from_impulses(state_for_force, force_terms_t)
        else:
            force_t = _take_step(external_force_seq_scan, t)
        xfrc_applied = jnp.zeros_like(state_for_step.xfrc_applied)
        if use_external_force:
            xfrc_applied = _set_body_wrench(xfrc_applied, force_t)
        else:
            force_t = jnp.zeros((n_batch, external_force_wrench_dim), dtype=state_for_step.qpos.dtype)
        state_for_step = state_for_step.replace(xfrc_applied=xfrc_applied)
        new_state: mjx.Data = jax.vmap(mjx_step, in_axes=(None, 0, 0))(sys, state_for_step, batched_rollout_params)
        return (
            (new_state, actuator_carry, rng, q_window, qd_window, u_ref_window, ctrl_window),
            (
                state.qpos[..., arm_dof_idx],
                state.qvel[..., arm_dof_idx],
                new_state.qacc[..., arm_dof_idx],
                u_dof,
                delta_tau_dof,
                state.time,
                force_t,
            ),
        )

    (new_state, _, _, _, _, _, _), (Qs, QDs, QDDs, Us, delta_taus, times, external_forces) = \
        lax.scan(step_fn, (state, actuator_carry0, rng, q_window, qd_window, u_ref_window, ctrl_window), jnp.arange(T_STEPS))
    
    # add last dims
    Qs = jnp.concat([Qs, new_state.qpos[None, ..., arm_dof_idx]], axis=0)
    QDs = jnp.concat([QDs, new_state.qvel[None, ..., arm_dof_idx]], axis=0)
    QDDs = jnp.concat([QDDs, jnp.zeros_like(new_state.qacc[None, ..., arm_dof_idx])], axis=0)
    Us = jnp.concat([Us, jnp.zeros_like(new_state.qacc[None, ..., arm_dof_idx])], axis=0)
    delta_taus = jnp.concat([delta_taus, jnp.zeros_like(delta_taus[:1])], axis=0) if delta_taus is not None else None
    times = jnp.concat([times, new_state.time[None]], axis=0)
    external_forces = jnp.concat([external_forces, jnp.zeros_like(external_forces[:1])], axis=0)
    return Qs, QDs, QDDs, Us, delta_taus, times, external_forces


# -----------------------
# Batched generator
# -----------------------
def generate_waypoints(
    k: jax.Array,
    num_wps: int,
    *,
    batch_n: int,
    dof: int,
    joint_range,
    pause_prob=0.4,
    initial_wp: Optional[jax.Array] = None,
    waypoint_max_delta_deg_profile: Optional[jax.Array] = None,
) -> jax.Array:
    """Sequential waypoint generator with bounded deltas and occasional pauses."""
    k_seq = random.split(k, num_wps)
    # first waypoint uses provided seed if given, otherwise sample uniformly in range
    if initial_wp is not None:
        w0 = jnp.asarray(initial_wp)
        if w0.ndim == 1:
            w0 = jnp.broadcast_to(w0, (batch_n, dof))
        w0 = jnp.clip(w0, joint_range[:, 0], joint_range[:, 1])
    else:
        w0 = random.uniform(k_seq[0], shape=(batch_n, dof), minval=joint_range[:, 0], maxval=joint_range[:, 1])
    if waypoint_max_delta_deg_profile is None:
        max_delta_profile_deg = jnp.array([100, 100, 100, 100, 150, 150, 150], dtype=jnp.float32)
    else:
        max_delta_profile_deg = jnp.asarray(waypoint_max_delta_deg_profile, dtype=jnp.float32).reshape(-1)
        if int(max_delta_profile_deg.shape[0]) <= 0:
            raise ValueError("waypoint_max_delta_deg_profile must contain at least one value.")
    max_delta_profile = jnp.deg2rad(max_delta_profile_deg) * 0.8
    if int(dof) <= int(max_delta_profile.shape[0]):
        max_delta = max_delta_profile[: int(dof)]
    else:
        tail = jnp.repeat(max_delta_profile[-1:], int(dof) - int(max_delta_profile.shape[0]), axis=0)
        max_delta = jnp.concatenate([max_delta_profile, tail], axis=0)

    def step(prev, i):
        key_delta, key_pause = random.split(k_seq[i])
        # pause_mask = random.bernoulli(key_delta, p=pause_prob, shape=prev.shape)
        pause_mask = random.bernoulli(key_delta, p=pause_prob, shape=prev.shape[:-1] + (1,)) # pause per sample, not per dof
        low = jnp.clip(prev - max_delta, joint_range[:, 0], joint_range[:, 1])
        high = jnp.clip(prev + max_delta, joint_range[:, 0], joint_range[:, 1])
        cand = random.uniform(key_pause, shape=prev.shape, minval=low, maxval=high)
        cand = jnp.where(pause_mask, prev, cand)
        return cand, cand

    _, wps_tail = lax.scan(step, w0, jnp.arange(1, num_wps))
    wps_tail = jnp.swapaxes(wps_tail, 0, 1)  # (batch, num_wps-1, dof)
    wps = jnp.concatenate([w0[:, None, :], wps_tail], axis=1)
    return wps  # (batch, num_wps, dof)


def rollout_generation(key, actuator_fn, model_mjx:mjx.Model, dof_idx_arm: np.ndarray, 
                       batched_rollout_params: structs.RolloutParams, num_waypoints:Optional[int]=None, 
                       base_rollout:Optional[jnp.ndarray]=None, duration:float=5.0, dt=0.001, 
                       reset_duration:Optional[float]=None, initial_actuator_carry:Optional[jnp.ndarray]=None,
                       adaptor_seq_length:int=1, pause_prob:float=0.15,
                       external_force_body_id: Optional[int] = None,
                       external_force_num_impulses: int = 0,
                       external_force_magnitude_min_n: float = 0.0,
                       external_force_magnitude_max_n: float = 0.0,
                       external_force_duration_min_s: float = 0.05,
                       external_force_duration_max_s: float = 0.20,
                       external_force_position_min_local_m: Optional[jnp.ndarray] = None,
                       external_force_position_max_local_m: Optional[jnp.ndarray] = None,
                       waypoint_max_delta_deg_profile: Optional[jnp.ndarray] = None) -> Dict[str, jnp.ndarray]:
    """
    """
    arm = jnp.array(dof_idx_arm)  # (7,)
    batch_n = batched_rollout_params.kp.shape[0] if batched_rollout_params.kp is not None else batched_rollout_params.dof_damping.shape[0]
    # dt = model_mjx.opt.timestep
    reset_interval = None if reset_duration is None else int(np.ceil(reset_duration / dt).astype(np.int32))
    joint_range = model_mjx.jnt_range[arm]
    assert dt == 0.001, "Only support timestep=0.001 for now."
    if external_force_body_id is not None and int(external_force_body_id) >= model_mjx.nbody:
        raise ValueError(
            f"external_force_body_id={external_force_body_id} out of range for nbody={model_mjx.nbody}"
        )
    if (external_force_position_min_local_m is None) != (external_force_position_max_local_m is None):
        raise ValueError(
            "external_force_position_min_local_m and external_force_position_max_local_m must be provided together."
        )
    has_external_force_position_box = external_force_position_min_local_m is not None
    if has_external_force_position_box:
        external_force_position_min_local_m = jnp.asarray(
            external_force_position_min_local_m, dtype=jnp.asarray(joint_range).dtype
        ).reshape((3,))
        external_force_position_max_local_m = jnp.asarray(
            external_force_position_max_local_m, dtype=jnp.asarray(joint_range).dtype
        ).reshape((3,))
        pos_min = jnp.minimum(
            external_force_position_min_local_m,
            external_force_position_max_local_m,
        )
        pos_max = jnp.maximum(
            external_force_position_min_local_m,
            external_force_position_max_local_m,
        )
        external_force_position_min_local_m = pos_min
        external_force_position_max_local_m = pos_max

    def one_traj(k):
        k1, k2, k_force, k_force_pos = random.split(k, 4)
        if base_rollout is not None:
            q_ref = base_rollout['q_ref'] if 'q_ref' in base_rollout else None
            qd_ref = base_rollout['qd_ref'] if 'qd_ref' in base_rollout else None
            total_steps = base_rollout['q'].shape[1]
        else:
            assert num_waypoints is not None, "Either waypoints or num_waypoints must be provided."
            T_STEPS = int(duration / dt)
            total_steps = T_STEPS
            wps = generate_waypoints(
                k1,
                num_waypoints,
                batch_n=batch_n,
                dof=arm.shape[-1],
                joint_range=joint_range,
                pause_prob=pause_prob,
                waypoint_max_delta_deg_profile=waypoint_max_delta_deg_profile,
            )
            q_ref, qd_ref = jax.vmap(build_traj_from_waypoints, (0, None, None))(wps, T_STEPS, duration)  # (T,7), (T,7)
        if has_external_force_position_box:
            external_force_impulse_terms = _sample_external_force_impulse_components(
                k_force,
                batch_n=batch_n,
                total_steps=total_steps,
                dt=dt,
                num_impulses=external_force_num_impulses,
                magnitude_min_n=external_force_magnitude_min_n,
                magnitude_max_n=external_force_magnitude_max_n,
                duration_min_s=external_force_duration_min_s,
                duration_max_s=external_force_duration_max_s,
                dtype=jnp.asarray(joint_range).dtype,
            )
            external_force_local_positions = sample_external_force_local_positions(
                k_force_pos,
                batch_n=batch_n,
                num_impulses=external_force_num_impulses,
                position_min_local_m=external_force_position_min_local_m,
                position_max_local_m=external_force_position_max_local_m,
                dtype=jnp.asarray(joint_range).dtype,
            )
            external_force_ee = None
        else:
            external_force_ee = sample_external_force_impulses(
                k_force,
                batch_n=batch_n,
                total_steps=total_steps,
                dt=dt,
                num_impulses=external_force_num_impulses,
                magnitude_min_n=external_force_magnitude_min_n,
                magnitude_max_n=external_force_magnitude_max_n,
                duration_min_s=external_force_duration_min_s,
                duration_max_s=external_force_duration_max_s,
                dtype=jnp.asarray(joint_range).dtype,
            )
            external_force_impulse_terms = None
            external_force_local_positions = None
        q, qd, qdd, u, delta_taus, times, external_force_ee = rollout_one(
            model_mjx,
            q_ref,
            qd_ref,
            actuator_fn,
            arm,
            batched_rollout_params,
            rng=k2,
            reset_interval=reset_interval,
            base_rollout=base_rollout,
            initial_actuator_carry=initial_actuator_carry,
            adaptor_seq_length=adaptor_seq_length,
            external_force_ee=external_force_ee,
            external_force_body_id=external_force_body_id,
            external_force_impulse_terms=external_force_impulse_terms,
            external_force_local_positions=external_force_local_positions,
        )
        q, qd, qdd, u, delta_taus, times, external_force_ee = jax.tree.map(
            lambda x: einops.rearrange(x, 't b ... -> b t ...'),
            (q, qd, qdd, u, delta_taus, times, external_force_ee),
        )
        q_ref = jnp.concat([q_ref, q_ref[:, -1:]], axis=1) if q_ref is not None else None
        qd_ref = jnp.concat([qd_ref, qd_ref[:, -1:]], axis=1) if qd_ref is not None else None
        return (q, qd, qdd, u, delta_taus, q_ref, qd_ref, times, external_force_ee)

    q, qd, qdd, u, delta_taus ,q_ref ,qd_ref ,times, external_force_ee = one_traj(key)
    return {
        "q": q,
        "qd": qd,
        "qdd": qdd,
        "u": u,
        "delta_taus": delta_taus,
        "q_ref": q_ref,
        "qd_ref": qd_ref,
        "times": times,
        "external_force_ee": external_force_ee,
    }
