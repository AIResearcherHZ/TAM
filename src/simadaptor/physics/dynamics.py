from typing import Optional, Tuple
import math

import mujoco
from mujoco import mjx

import jax
import jax.numpy as jnp

from mujoco.mjx._src import forward as mjx_forward
from mujoco.mjx._src import smooth
import numpy as np


DEFAULT_SIM_TIMESTEP = 0.001


def load_mjx_model_from_path(
    xml_path: str,
    remove_constraints: bool = False,
    timestep: Optional[float] = DEFAULT_SIM_TIMESTEP,
) -> mjx.Model:
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    if timestep is not None:
        timestep = float(timestep)
        if not math.isfinite(timestep) or timestep <= 0.0:
            raise ValueError(f"timestep must be positive and finite, got {timestep!r}.")
        mj_model.opt.timestep = timestep
    if remove_constraints:
        # mj_model.opt.disableflags = mj_model.opt.disableflags | mujoco.mjtDisableBit.mjDSBL_CONSTRAINT | mujoco.mjtDisableBit.mjDSBL_EQUALITY | mujoco.mjtDisableBit.mjDSBL_CONTACT
        mj_model.opt.disableflags = mj_model.opt.disableflags | mujoco.mjtDisableBit.mjDSBL_CONTACT
        mj_model.dof_frictionloss = np.zeros_like(mj_model.dof_frictionloss)  # remove friction
        mj_model.jnt_limited = np.zeros_like(mj_model.jnt_limited)  # remove joint limits
        mj_model.actuator_actlimited = np.zeros_like(mj_model.actuator_actlimited)
        mj_model.actuator_forcelimited = np.zeros_like(mj_model.actuator_forcelimited)
        mj_model.jnt_actfrclimited = np.zeros_like(mj_model.jnt_actfrclimited)
        
    mjx_model = mjx.put_model(mj_model)
    return mjx_model

def _remove_limits(model: mjx.Model) -> mjx.Model:
    return model.tree_replace({
        "jnt_range": jnp.zeros_like(model.jnt_range).at[..., 0].set(-jnp.inf).at[..., 1].set(jnp.inf),
        "actuator_ctrlrange": jnp.zeros_like(model.actuator_ctrlrange).at[..., 0].set(-jnp.inf).at[..., 1].set(jnp.inf),
        "actuator_actrange": jnp.zeros_like(model.actuator_actrange).at[..., 0].set(-jnp.inf).at[..., 1].set(jnp.inf),
        "jnt_actfrcrange": jnp.zeros_like(model.jnt_actfrcrange).at[..., 0].set(-jnp.inf).at[..., 1].set(jnp.inf),
        "actuator_forcerange": jnp.zeros_like(model.actuator_forcerange).at[..., 0].set(-jnp.inf).at[..., 1].set(jnp.inf),
        "actuator_gear": jnp.zeros_like(model.actuator_gear).at[..., 0].set(1.0),
    })


def _prepare_inverse_dynamics_model(mjx_model: mjx.Model) -> mjx.Model:
    return _remove_limits(
        mjx_model.replace(
            opt=mjx_model.opt.replace(
                iterations=1,
                ls_iterations=1,
                disableflags=mjx_model.opt.disableflags | mjx.DisableBit.CONTACT,
                timestep=0.001,
                integrator=mujoco.mjtIntegrator.mjINT_EULER,
            )
        )
    )


def affine_inverse_dynamics_terms(
    mjx_model: mjx.Model,
    q: jax.Array,
    qd: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """Return `(M, bias)` for a fixed state such that `tau = M @ qdd + bias`.

    The returned bias term follows the same convention as `mjx_inverse_dynamics_rne`,
    including the model's configured gravity-compensation and damping behavior.
    """
    mjx_model = _prepare_inverse_dynamics_model(mjx_model)

    nq = mjx_model.nq
    nv = mjx_model.nv
    assert q.shape == (nq,)
    assert qd.shape == (nv,)

    d = mjx.make_data(mjx_model)
    d = d.replace(qpos=q, qvel=qd)
    d = mjx.forward(mjx_model, d)

    mass_matrix = mjx.full_m(mjx_model, d)
    bias = d.qfrc_bias - d.qfrc_gravcomp + qd * mjx_model.dof_damping
    return mass_matrix, bias


def affine_forward_dynamics_terms(
    mjx_model: mjx.Model,
    q: jax.Array,
    qd: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Return forward-dynamics affine terms for fixed state.

    At fixed `q, qd`, MuJoCo forward dynamics is affine in control:
      `M(q) @ qdd = rhs_bias(q, qd) + control @ control_map(q, qd)`

    Returns:
      - `mass_matrix`: full inertia matrix used by the forward solve
      - `rhs_bias`: force-space bias with zero control already folded in
      - `control_map`: matrix mapping actuator controls to generalized forces
    """
    mjx_model = _remove_limits(mjx_model)

    nq = mjx_model.nq
    nv = mjx_model.nv
    nu = mjx_model.nu
    assert q.shape == (nq,)
    assert qd.shape == (nv,)

    ctrl0 = jnp.zeros((nu,), dtype=q.dtype)
    d = mjx.make_data(mjx_model).replace(qpos=q, qvel=qd, ctrl=ctrl0)
    d = mjx_forward.fwd_position(mjx_model, d)
    d = mjx_forward.fwd_velocity(mjx_model, d)

    mass_matrix = mjx.full_m(mjx_model, d)

    def _qfrc_actuator(ctrl: jax.Array) -> jax.Array:
        return mjx_forward.fwd_actuation(mjx_model, d.replace(ctrl=ctrl)).qfrc_actuator

    qfrc_actuator_0 = _qfrc_actuator(ctrl0)
    if int(nu) == 0:
        control_map = jnp.zeros((0, nv), dtype=q.dtype)
    else:
        ctrl_eye = jnp.eye(nu, dtype=q.dtype)
        qfrc_actuator_basis = jax.vmap(_qfrc_actuator)(ctrl_eye)
        control_map = qfrc_actuator_basis - qfrc_actuator_0[None, :]

    rhs_bias = d.qfrc_passive - d.qfrc_bias + qfrc_actuator_0
    return mass_matrix, rhs_bias, control_map


def get_mass_matrix_diag_at_qpos0(model: mjx.Model) -> jax.Array:
    """Return diag(M(q0)) where q0 is the model initial joint configuration."""
    qpos0 = jnp.asarray(model.qpos0, dtype=jnp.float32)
    qvel0 = jnp.zeros((model.nv,), dtype=qpos0.dtype)
    ctrl0 = jnp.zeros((model.nu,), dtype=qpos0.dtype)
    data0 = mjx.make_data(model).replace(qpos=qpos0, qvel=qvel0, ctrl=ctrl0)
    data0 = mjx.forward(model, data0)
    mass_matrix = mjx.full_m(model, data0)
    return jnp.clip(jnp.diag(mass_matrix), 1e-6, None)



def _catmull_rom_slopes(t_seq, q_seq):
    """Compute per-knot slopes for a Catmull-Rom-style cubic spline."""
    h = jnp.diff(t_seq)  # (T-1,)
    delta = jnp.diff(q_seq, axis=0) / h[:, None]  # (T-1, nq)
    m0 = delta[0]
    mN = delta[-1]
    w0 = h[:-1][:, None]
    w1 = h[1:][:, None]
    denom = w0 + w1
    denom = jnp.where(denom == 0.0, 1.0, denom)  # avoid div-by-zero if times identical
    mid = (w1 * delta[:-1] + w0 * delta[1:]) / denom
    return jnp.concatenate([m0[None], mid, mN[None]], axis=0)  # (T, nq)

def _cubic_interp_with_derivs(t_query, t_seq, q_seq):
    """Evaluate cubic Hermite spline at t_query (vector) and return q, qd, qdd analytically."""
    slopes = _catmull_rom_slopes(t_seq, q_seq)
    eps = jnp.finfo(t_seq.dtype).eps

    def _interp_one(tu):
        idx = jnp.clip(jnp.searchsorted(t_seq, tu, side="right") - 1, 0, t_seq.shape[0] - 2)
        t0 = t_seq[idx]
        t1 = t_seq[idx + 1]
        h = jnp.maximum(t1 - t0, eps)
        inv_h = 1.0 / h
        inv_h2 = inv_h * inv_h

        s = (tu - t0) * inv_h
        s2 = s * s
        s3 = s2 * s

        p0 = q_seq[idx]
        p1 = q_seq[idx + 1]
        m0 = slopes[idx]
        m1 = slopes[idx + 1]

        q_val = (
            (2 * s3 - 3 * s2 + 1) * p0
            + (-2 * s3 + 3 * s2) * p1
            + (s3 - 2 * s2 + s) * (h * m0)
            + (s3 - s2) * (h * m1)
        )

        qd_val = (
            (6 * s2 - 6 * s) * inv_h * p0
            + (-6 * s2 + 6 * s) * inv_h * p1
            + (3 * s2 - 4 * s + 1) * m0
            + (3 * s2 - 2 * s) * m1
        )

        qdd_val = (
            (12 * s - 6) * inv_h2 * p0
            + (-12 * s + 6) * inv_h2 * p1
            + (6 * s - 4) * inv_h * m0
            + (6 * s - 2) * inv_h * m1
        )

        return q_val, qd_val, qdd_val

    return jax.vmap(_interp_one)(t_query)


def q_traj_to_traj(
    q_traj: jax.Array,
    u_traj: Optional[jax.Array],
    times: jax.Array,
    *,
    dt: float,
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


def q_traj_to_accel(q_traj: jax.Array, dt: float) -> jax.Array:
    """
    q_traj: (..., T, nq)
    returns: (..., T, nq)

    Uses a 5-point finite-difference stencil along the time dimension to
    estimate joint accelerations. Requires at least 5 samples.
    """
    # if dt <= 0:
    #     raise ValueError("dt must be positive for finite differencing")

    T = q_traj.shape[-2]
    if T < 5:
        raise ValueError("Need at least 5 frames to use a 5-point stencil")

    inv_dt2 = 1.0 / (dt * dt)
    dtype = q_traj.dtype

    start_coeff = jnp.asarray([35.0, -104.0, 114.0, -56.0, 11.0], dtype=dtype) / 12.0
    near_start_coeff = jnp.asarray([11.0, -20.0, 6.0, 4.0, -1.0], dtype=dtype) / 12.0
    center_coeff = jnp.asarray([-1.0, 16.0, -30.0, 16.0, -1.0], dtype=dtype) / 12.0
    near_end_coeff = near_start_coeff[::-1]
    end_coeff = start_coeff[::-1]

    def weighted_sum(coeffs, values):
        return jnp.tensordot(coeffs, values, axes=[0, -2])

    qdd = jnp.zeros_like(q_traj)

    start_window = q_traj[..., :5, :]
    qdd = qdd.at[..., 0, :].set(weighted_sum(start_coeff, start_window) * inv_dt2)
    qdd = qdd.at[..., 1, :].set(weighted_sum(near_start_coeff, start_window) * inv_dt2)

    end_window = q_traj[..., -5:, :]
    qdd = qdd.at[..., -2, :].set(weighted_sum(near_end_coeff, end_window) * inv_dt2)
    qdd = qdd.at[..., -1, :].set(weighted_sum(end_coeff, end_window) * inv_dt2)

    central = (
        center_coeff[0] * q_traj[..., :-4, :]
        + center_coeff[1] * q_traj[..., 1:-3, :]
        + center_coeff[2] * q_traj[..., 2:-2, :]
        + center_coeff[3] * q_traj[..., 3:-1, :]
        + center_coeff[4] * q_traj[..., 4:, :]
    ) * inv_dt2
    qdd = qdd.at[..., 2:-2, :].set(central)

    return qdd

def mjx_inverse_dynamics_rne(
    mjx_model: mjx.Model,
    q: jax.Array,
    qd: jax.Array,
    qdd: jax.Array,
) -> jax.Array:
    """
    mjx_model should turn off
    dof_frictionloss, jnt_limited
    
    """
    nq = mjx_model.nq
    nv = mjx_model.nv
    assert q.shape == (nq,)
    assert qd.shape == (nv,)
    assert qdd.shape == (nv,)

    mass_matrix, bias = affine_inverse_dynamics_terms(mjx_model, q, qd)
    tau = jnp.einsum("...ij,...j", mass_matrix, qdd) + bias
    return tau

    # 3) Set desired acceleration (generalized)
    d = d.replace(qacc=qdd)

    # 4) RNE: with flg_acc=True to include inertial term
    d = smooth.rne(mjx_model, d, flg_acc=True)

    return d.qfrc_bias


def _compute_external_tau_equivalent_single(
    mjx_model: mjx.Model,
    q: jax.Array,
    qd: jax.Array,
    external_force_ee: jax.Array,
    *,
    external_force_body_id: jax.Array,
) -> jax.Array:
    mjx_model = mjx_model.replace(
        opt=mjx_model.opt.replace(
            iterations=1,
            ls_iterations=1,
            disableflags=mjx_model.opt.disableflags | mjx.DisableBit.CONTACT,
            timestep=0.001,
            integrator=mujoco.mjtIntegrator.mjINT_EULER,
        )
    )
    mjx_model = _remove_limits(mjx_model)

    nq = mjx_model.nq
    nv = mjx_model.nv
    assert q.shape == (nq,)
    assert qd.shape == (nv,)

    force_ee = jnp.asarray(external_force_ee, dtype=q.dtype)
    if force_ee.shape[-1] == 3:
        force_wrench = jnp.zeros((6,), dtype=q.dtype).at[:3].set(force_ee)
    elif force_ee.shape[-1] == 6:
        force_wrench = force_ee
    else:
        raise ValueError(f"external_force_ee must have shape (..., 3) or (..., 6); got {force_ee.shape}")

    body_id = jnp.asarray(external_force_body_id, dtype=jnp.int32)

    def _compute_with_force(_):
        data0 = mjx.make_data(mjx_model).replace(
            qpos=q,
            qvel=qd,
            ctrl=jnp.zeros((mjx_model.nu,), dtype=q.dtype),
        )

        xfrc_zero = jnp.zeros_like(data0.xfrc_applied)
        data_no_force = mjx.forward(mjx_model, data0.replace(xfrc_applied=xfrc_zero))

        xfrc_applied = xfrc_zero.at[body_id, :].set(force_wrench.astype(xfrc_zero.dtype))
        data_with_force = mjx.forward(mjx_model, data0.replace(xfrc_applied=xfrc_applied))

        delta_qacc = data_with_force.qacc_smooth - data_no_force.qacc_smooth
        mass_matrix = mjx.full_m(mjx_model, data_no_force)
        return jnp.einsum("ij,j->i", mass_matrix, delta_qacc)

    return jax.lax.cond(
        (body_id >= 0) & (body_id < int(mjx_model.nbody)),
        _compute_with_force,
        lambda _: jnp.zeros((nv,), dtype=q.dtype),
        operand=None,
    )


def compute_external_tau_equivalent(
    mjx_model: mjx.Model,
    q: jax.Array,
    qd: jax.Array,
    external_force_ee: jax.Array,
    *,
    external_force_body_id: int | jax.Array = -1,
) -> jax.Array:
    """Map an applied external body wrench to an equivalent joint torque.

    Supported shapes:
      - `q`, `qd`: `[D]`, `external_force_ee`: `[3|6]`, `external_force_body_id`: scalar
      - `q`, `qd`: `[B, D]`, `external_force_ee`: `[B, 3|6]`, `external_force_body_id`: scalar or `[B]`
    """
    q = jnp.asarray(q)
    qd = jnp.asarray(qd)
    force = jnp.asarray(external_force_ee, dtype=q.dtype)
    body_id = jnp.asarray(
        -1 if external_force_body_id is None else external_force_body_id,
        dtype=jnp.int32,
    )

    if q.ndim != qd.ndim:
        raise ValueError(f"q and qd rank mismatch: q={q.shape}, qd={qd.shape}.")

    if q.ndim == 1:
        if force.ndim != 1:
            raise ValueError(
                f"Expected external_force_ee rank 1 for a single state; got {force.shape}."
            )
        if body_id.ndim != 0:
            raise ValueError(
                f"Expected scalar external_force_body_id for a single state; got {body_id.shape}."
            )
        return _compute_external_tau_equivalent_single(
            mjx_model,
            q,
            qd,
            force,
            external_force_body_id=body_id,
        )

    if q.ndim == 2:
        batch_size = int(q.shape[0])
        if int(qd.shape[0]) != batch_size:
            raise ValueError(f"Batch mismatch: q={q.shape}, qd={qd.shape}.")
        if force.ndim != 2 or int(force.shape[0]) != batch_size:
            raise ValueError(
                f"Expected external_force_ee shape ({batch_size}, 3|6), got {force.shape}."
            )
        if body_id.ndim == 0:
            body_id = jnp.broadcast_to(body_id, (batch_size,))
        elif body_id.ndim == 1 and int(body_id.shape[0]) == batch_size:
            pass
        else:
            raise ValueError(
                f"Expected scalar or shape ({batch_size},) external_force_body_id for batched states, "
                f"got {body_id.shape}."
            )
        return jax.vmap(
            lambda q_i, qd_i, force_i, body_id_i: _compute_external_tau_equivalent_single(
                mjx_model,
                q_i,
                qd_i,
                force_i,
                external_force_body_id=body_id_i,
            )
        )(q, qd, force, body_id)

    raise ValueError(f"Unsupported q/qd rank for compute_external_tau_equivalent: q={q.shape}, qd={qd.shape}.")
    
    
if __name__ == '__main__':
    from functools import partial
    import numpy as np
    import simadaptor.physics.actuator as actuator_util
    
    # robot = xml_parser.load_mjcf_kinematics("assets/franka_panda/panda.xml")
    
    
    mj_model = mujoco.MjModel.from_xml_path("assets/franka_panda/panda.xml")
    mjx_model = mjx.put_model(mj_model)
    mjx_model = mjx_model.replace(jnt_limited=np.zeros_like(mjx_model.jnt_limited))

    q   = jnp.zeros(mjx_model.nq)
    qd  = jnp.zeros(mjx_model.nv)
    qdd = jnp.zeros(mjx_model.nv)  # for pure gravity compensation
    
    rng = jax.random.PRNGKey(0)
    rng, q_key, qd_key, qdd_key, tau_key = jax.random.split(rng, 5)
    q = jax.random.uniform(q_key, shape=(mjx_model.nv,), minval=mjx_model.jnt_range[...,0], maxval=mjx_model.jnt_range[...,1])
    qd = jax.random.normal(qd_key, shape=(mjx_model.nv,)) * 0.1
    qdd = jax.random.normal(qdd_key, shape=(mjx_model.nv,)) * 0.1
    tau_test = jax.random.normal(tau_key, shape=(mjx_model.nv,))
    
    mjx_model_wo_damping = mjx_model.replace(dof_damping=jnp.zeros_like(mjx_model.dof_damping))
    
    inv_dyn = partial(mjx_inverse_dynamics_rne, mjx_model)
    tau = inv_dyn(q, qd, qdd)
    res_d = mjx.step(mjx_model, mjx.make_data(mjx_model).replace(qpos=q, qvel=qd, ctrl=tau))
    
    print(qdd - res_d.qacc)  # should be close to zero
