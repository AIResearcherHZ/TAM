from collections.abc import Mapping

import jax.numpy as jnp
import jax
import jax.nn as jnn

import simadaptor.physics.dynamics as dynamics
from mujoco import mjx
from simadaptor.core.structs import ActuatorParams, RolloutParams

DEADZONE_SLOPE = 1e-2  # small slope inside the deadzone to keep the mapping invertible


def _directional_param(param: jnp.ndarray | None, qd: jnp.ndarray) -> jnp.ndarray | None:
    """Select per-direction parameters ([..., 2]) based on velocity sign."""
    if param is None:
        return None
    if param.shape[-1] == 2:
        neg = param[..., 0]
        pos = param[..., 1]
        return jnp.where(qd >= 0, pos, neg)
    return param


def _split_friction_params(friction_params: jnp.ndarray | None):
    """Return (amp, slope, shift) each shaped (..., ndof, 2) from stacked friction params."""
    if friction_params is None:
        return None, None, None
    amp = friction_params[..., 0:2]
    slope = friction_params[..., 2:4]
    shift = friction_params[..., 4:6]
    return amp, slope, shift


def sigmoidal_friction(qd: jnp.ndarray, amp: jnp.ndarray | None, slope: jnp.ndarray | None, shift: jnp.ndarray | None) -> jnp.ndarray:
    """Compute sigmoidal friction torque for each joint.

    Model per joint: tau = A * (sigma(k * (v + s)) - sigma(k * s)), where sigma is logistic.
    Parameters can be shaped (ndof,) or (ndof, 2) for (neg, pos) directional values.
    """
    if amp is None or slope is None or shift is None:
        return jnp.zeros_like(qd)

    amp_dir = _directional_param(amp, qd)
    slope_dir = _directional_param(slope, qd)
    shift_dir = _directional_param(shift, qd)

    return amp_dir * (jnn.sigmoid(slope_dir * (qd + shift_dir)) - jnn.sigmoid(slope_dir * shift_dir))


def invert_sigmoidal_friction(tau, amp, slope, shift, eps: float = 1e-6):
    """Invert sigmoidal friction model to recover velocity from torque.

    Model: tau = amp * (sigma(k * (v + s)) - sigma(k * s)), sigma(x) = 1 / (1 + exp(-x)).
    Broadcasts parameters to match tau's leading dimensions.
    """
    tau = jnp.asarray(tau)
    amp = jnp.asarray(amp)
    slope = jnp.asarray(slope)
    shift = jnp.asarray(shift)

    def _expand(param):
        return jnp.reshape(param, (1,) * (tau.ndim - 1) + (-1,))

    amp_b = _expand(amp)
    slope_b = _expand(slope)
    shift_b = _expand(shift)

    sigma_ks = jnn.sigmoid(slope_b * shift_b)
    y = tau / amp_b + sigma_ks
    y_clipped = jnp.clip(y, eps, 1.0 - eps)
    v = jnp.log(y_clipped / (1.0 - y_clipped)) / slope_b - shift_b
    return v


    
    
def actuator_model(tau, q, qd, actuator_params:ActuatorParams|RolloutParams=None):
    
    if isinstance(actuator_params, RolloutParams):
        actuator_params = actuator_params.actuator_params
    
    if actuator_params.torque_scale is not None:
        scale = actuator_params.torque_scale  # (ndof,) or (ndof,2)
        scale_neg = scale[..., 0]
        scale_pos = scale[..., 1]
        tau = tau * jnp.where(tau >= 0, scale_pos, scale_neg)
    
    # deadzone -> torque bias -> damping (directional) -> sigmoidal friction
    if actuator_params.deadzone is not None:
        dz = actuator_params.deadzone  # (ndof,) or (ndof,2)
        dz_neg = dz[..., 0] if dz.shape[-1]==2 else dz
        dz_pos = dz[..., 1] if dz.shape[-1]==2 else dz
        abs_tau = jnp.abs(tau)
        sign_tau = jnp.sign(tau)
        dz_dir = jnp.where(sign_tau >= 0, dz_pos, dz_neg)
        tau = jnp.where(
            abs_tau <= dz_dir,
            tau * DEADZONE_SLOPE,
            sign_tau * (abs_tau - dz_dir + DEADZONE_SLOPE * dz_dir),
        )

    if actuator_params.torque_bias is not None:
        bias = actuator_params.torque_bias  # (ndof,) or (ndof,2)
        bias_neg = bias[..., 0] if bias.shape[-1]==2 else bias
        bias_pos = bias[..., 1] if bias.shape[-1]==2 else bias
        tau = tau + jnp.where(qd >= 0, bias_pos, bias_neg)
    
    if actuator_params.damping is not None:
        damp = actuator_params.damping  # (ndof,) or (ndof,2)
        damp_neg = damp[..., 0] if damp.shape[-1]==2 else damp
        damp_pos = damp[..., 1] if damp.shape[-1]==2 else damp
        tau_damp = jnp.where(qd >= 0, damp_pos, damp_neg) * qd
        tau = tau - tau_damp

    # non-linear sigmoidal friction
    amp, slope, shift = _split_friction_params(actuator_params.friction_params)
    tau_fric = sigmoidal_friction(qd, amp, slope, shift)
    tau = tau - tau_fric

    return tau

def inv_actuator_model(tau_desired, q, qd, actuator_params:ActuatorParams=None):
    '''
    calculate the inverse of the actuator model
    '''

    tau_in = tau_desired
    if actuator_params is None:
        return tau_in
    # invert sigmoidal friction first (subtract in forward)
    amp, slope, shift = _split_friction_params(actuator_params.friction_params)
    tau_in = tau_in + sigmoidal_friction(qd, amp, slope, shift)
    if actuator_params.damping is not None:
        damp = actuator_params.damping
        damp_neg = damp[..., 0]
        damp_pos = damp[..., 1]
        tau_in = tau_in + jnp.where(qd >= 0, damp_pos, damp_neg) * qd
    if actuator_params.torque_bias is not None:
        bias = actuator_params.torque_bias
        bias_neg = bias[..., 0]
        bias_pos = bias[..., 1]
        tau_in = tau_in - jnp.where(qd >= 0, bias_pos, bias_neg)
    if actuator_params.deadzone is not None:
        dz = actuator_params.deadzone
        dz_neg = dz[..., 0]
        dz_pos = dz[..., 1]
        sign_tau = jnp.sign(tau_in)
        dz_dir = jnp.where(sign_tau >= 0, dz_pos, dz_neg)
        abs_tau = jnp.abs(tau_in)
        sign_tau = jnp.sign(tau_in)
        threshold = DEADZONE_SLOPE * dz_dir
        tau_in = jnp.where(
            abs_tau <= threshold,
            tau_in / DEADZONE_SLOPE,
            sign_tau * (abs_tau - threshold + dz_dir),
        )

    if actuator_params.torque_scale is not None:
        scale = actuator_params.torque_scale
        if scale.shape[-1] == 2:
            scale_neg = scale[..., 0]
            scale_pos = scale[..., 1]
            scale_dir = jnp.where(tau_in >= 0, scale_pos, scale_neg)
        else:
            scale_dir = scale
        tau_in = tau_in / scale_dir

    return tau_in


def _pad_prefix_to_size(x: jnp.ndarray, size: int) -> jnp.ndarray:
    common = min(int(x.shape[-1]), int(size))
    out = jnp.zeros(x.shape[:-1] + (int(size),), dtype=x.dtype)
    return out.at[..., :common].set(x[..., :common])


def _solve_forward_dynamics_from_affine_terms(
    mass_matrix: jnp.ndarray,
    rhs_force: jnp.ndarray,
) -> jnp.ndarray:
    rhs = rhs_force
    if rhs.ndim == 1:
        return jnp.linalg.solve(mass_matrix, rhs)
    rhs_t = jnp.swapaxes(rhs, -1, -2)
    qacc_t = jnp.linalg.solve(mass_matrix, rhs_t)
    return jnp.swapaxes(qacc_t, -1, -2)


def _external_tau_equivalent_or_zero(
    mjx_model: mjx.Model,
    q: jnp.ndarray,
    qd: jnp.ndarray,
    external_force_ee: jnp.ndarray | None,
    *,
    external_force_body_id: int | jax.Array = -1,
) -> jnp.ndarray:
    if external_force_ee is None:
        return jnp.zeros((int(mjx_model.nv),), dtype=q.dtype)
    return dynamics.compute_external_tau_equivalent(
        mjx_model,
        q,
        qd,
        external_force_ee,
        external_force_body_id=external_force_body_id,
    )


def calculate_gt_torque_shared_linear(
    mjx_model_perturbed: mjx.Model,
    q,
    qd,
    *,
    mjx_model_ideal: mjx.Model,
    tau,
    actuator_params: ActuatorParams | RolloutParams = None,
    external_force_ee: jnp.ndarray | None = None,
    external_force_body_id: int | jax.Array = -1,
) -> jnp.ndarray:
    """Fast GT-torque solve for one state and one-or-many desired torque samples.

    This reuses the ideal and perturbed affine dynamics terms for fixed `q, qd`:
      `tau_eff = M(q) @ qdd + bias(q, qd)`.

    Shapes:
      q: [nq_in]
      qd: [nv_in]
      tau: [nu_in] or [nsamp, nu_in]
      returns: [common_nu] or [nsamp, common_nu]
    """
    q = jnp.asarray(q)    # [nq_in]
    qd = jnp.asarray(qd)  # [nv_in]
    tau = jnp.asarray(tau)  # [nu_in] or [nsamp, nu_in]
    if q.ndim != 1 or qd.ndim != 1:
        raise ValueError(f"Expected q/qd rank-1 for shared_linear solve; got q={q.shape}, qd={qd.shape}.")
    if tau.ndim not in (1, 2):
        raise ValueError(f"Expected tau rank 1 or 2 for shared_linear solve; got tau={tau.shape}.")

    tau_is_single = tau.ndim == 1
    tau_samples = tau[None, :] if tau_is_single else tau  # [nsamp, nu_in]

    if isinstance(actuator_params, RolloutParams):
        mjx_model_perturbed = actuator_params.set_mjx_model(mjx_model_perturbed)
        actuator_params = actuator_params.actuator_params

    mjx_model_perturbed = dynamics._remove_limits(mjx_model_perturbed)
    mjx_model_ideal = dynamics._remove_limits(mjx_model_ideal)

    common_nq = min(int(q.shape[-1]), int(mjx_model_ideal.nq), int(mjx_model_perturbed.nq))
    common_nv = min(int(qd.shape[-1]), int(mjx_model_ideal.nv), int(mjx_model_perturbed.nv))
    common_nu = min(
        int(tau_samples.shape[-1]),
        int(mjx_model_ideal.nu),
        int(mjx_model_perturbed.nu),
        int(mjx_model_ideal.nv),
        int(mjx_model_perturbed.nv),
    )
    if common_nu < 1:
        empty = jnp.zeros(tau_samples.shape[:-1] + (0,), dtype=tau_samples.dtype)
        return empty[0] if tau_is_single else empty

    q_ideal = _pad_prefix_to_size(q, mjx_model_ideal.nq)    # [nq_ideal]
    qd_ideal = _pad_prefix_to_size(qd, mjx_model_ideal.nv)  # [nv_ideal]
    q_ideal = q_ideal.at[:common_nq].set(q[:common_nq])
    qd_ideal = qd_ideal.at[:common_nv].set(qd[:common_nv])
    ctrl_ideal = _pad_prefix_to_size(tau_samples, mjx_model_ideal.nu).at[..., :common_nu].set(
        tau_samples[..., :common_nu]
    )  # [nsamp, nu_ideal]

    mass_ideal, rhs_bias_ideal, control_map_ideal = dynamics.affine_forward_dynamics_terms(
        mjx_model_ideal,
        q_ideal,
        qd_ideal,
    )  # [nv_ideal, nv_ideal], [nv_ideal], [nu_ideal, nv_ideal]
    tau_eff_ext_ideal = _external_tau_equivalent_or_zero(
        mjx_model_ideal,
        q_ideal,
        qd_ideal,
        external_force_ee,
        external_force_body_id=external_force_body_id,
    )  # [nv_ideal]
    rhs_ideal = rhs_bias_ideal + jnp.einsum("...u,uv->...v", ctrl_ideal, control_map_ideal)  # [nsamp, nv_ideal]
    rhs_ideal = rhs_ideal + tau_eff_ext_ideal  # [nsamp, nv_ideal]
    qacc_ideal = _solve_forward_dynamics_from_affine_terms(mass_ideal, rhs_ideal)  # [nsamp, nv_ideal]

    q_ptb = _pad_prefix_to_size(q, mjx_model_perturbed.nq)    # [nq_ptb]
    qd_ptb = _pad_prefix_to_size(qd, mjx_model_perturbed.nv)  # [nv_ptb]
    q_ptb = q_ptb.at[:common_nq].set(q[:common_nq])
    qd_ptb = qd_ptb.at[:common_nv].set(qd[:common_nv])
    qacc_ptb = _pad_prefix_to_size(qacc_ideal[..., :common_nv], mjx_model_perturbed.nv)  # [nsamp, nv_ptb]
    tau_eff_ext_ptb = _external_tau_equivalent_or_zero(
        mjx_model_perturbed,
        q_ptb,
        qd_ptb,
        external_force_ee,
        external_force_body_id=external_force_body_id,
    )  # [nv_ptb]

    mass_ptb, bias_ptb = dynamics.affine_inverse_dynamics_terms(mjx_model_perturbed, q_ptb, qd_ptb)
    # mass_ptb: [nv_ptb, nv_ptb], bias_ptb: [nv_ptb]
    tau_eff_ptb = jnp.einsum("ij,...j->...i", mass_ptb, qacc_ptb) + bias_ptb - tau_eff_ext_ptb  # [nsamp, nv_ptb]
    tau_cmd = inv_actuator_model(
        tau_eff_ptb[..., :common_nu],
        q_ptb,
        qd_ptb[..., :common_nu],
        actuator_params,
    )  # [nsamp, common_nu]
    return tau_cmd[0] if tau_is_single else tau_cmd


def calculate_gt_torque(mjx_model_perturbed:mjx.Model, q, qd, *, 
                        mjx_model_ideal:mjx.Model|None=None, tau=None, actuator_params:ActuatorParams|RolloutParams=None, qacc:jnp.ndarray|None=None,
                        external_force_ee: jnp.ndarray | None = None, external_force_body_id: int | jax.Array = -1) -> jnp.ndarray:
    """
    Compute torques for the perturbed model that reproduce the desired acceleration.
    Option 1: provide mjx_model_ideal + desired torque (tau) to derive qacc internally.
    Option 2: provide qacc directly and omit mjx_model_ideal.
    """
    # ensure that both have effectively no limits so the inverse dynamics does not clip

    if isinstance(actuator_params, RolloutParams):
        mjx_model_perturbed = actuator_params.set_mjx_model(mjx_model_perturbed)
        actuator_params = actuator_params.actuator_params

    mjx_model_perturbed = dynamics._remove_limits(mjx_model_perturbed)

    if qacc is None:
        if mjx_model_ideal is None or tau is None:
            raise ValueError("Provide mjx_model_ideal and tau when qacc is not supplied.")
        mjx_model_ideal = dynamics._remove_limits(mjx_model_ideal)

        common_nq = min(q.shape[-1], mjx_model_ideal.nq, mjx_model_perturbed.nq)
        common_nv = min(qd.shape[-1], mjx_model_ideal.nv, mjx_model_perturbed.nv)
        common_nu = min(tau.shape[-1], mjx_model_ideal.nu, mjx_model_perturbed.nu)

        q_ideal = jnp.zeros(mjx_model_ideal.nq, dtype=jnp.float32).at[:common_nq].set(q[..., :common_nq])
        qd_ideal = jnp.zeros(mjx_model_ideal.nv, dtype=jnp.float32).at[:common_nv].set(qd[..., :common_nv])
        ctrl_ideal = jnp.zeros(mjx_model_ideal.nu, dtype=jnp.float32).at[:common_nu].set(tau[..., :common_nu])
        xfrc_ideal = jnp.zeros_like(mjx.make_data(mjx_model_ideal).xfrc_applied)
        if external_force_ee is not None:
            force_wrench = jnp.asarray(external_force_ee, dtype=q_ideal.dtype)
            if force_wrench.shape[-1] == 3:
                force_wrench = jnp.zeros((6,), dtype=q_ideal.dtype).at[:3].set(force_wrench)
            elif force_wrench.shape[-1] == 6:
                pass
            else:
                raise ValueError(
                    f"external_force_ee must have shape (..., 3) or (..., 6); got {force_wrench.shape}"
                )
            body_id = jnp.asarray(
                -1 if external_force_body_id is None else external_force_body_id,
                dtype=jnp.int32,
            )

            def _apply_force(xfrc: jnp.ndarray) -> jnp.ndarray:
                return xfrc.at[body_id, :].set(force_wrench.astype(xfrc.dtype))

            xfrc_ideal = jax.lax.cond(
                (body_id >= 0) & (body_id < int(mjx_model_ideal.nbody)),
                _apply_force,
                lambda xfrc: xfrc,
                xfrc_ideal,
            )

        data_ideal = mjx.forward(
            mjx_model_ideal,
            mjx.make_data(mjx_model_ideal).replace(
                qpos=q_ideal,
                qvel=qd_ideal,
                ctrl=ctrl_ideal,
                xfrc_applied=xfrc_ideal,
            ),
        )
        qacc_src = data_ideal.qacc_smooth
    else:
        common_nq = min(q.shape[-1], mjx_model_perturbed.nq)
        common_nv = min(qd.shape[-1], qacc.shape[-1], mjx_model_perturbed.nv)
        common_nu = min(tau.shape[-1], mjx_model_perturbed.nu) if tau is not None else mjx_model_perturbed.nu
        qacc_src = qacc

    q_ptb = jnp.zeros(mjx_model_perturbed.nq, dtype=jnp.float32).at[:common_nq].set(q[..., :common_nq])
    qd_ptb = jnp.zeros(mjx_model_perturbed.nv, dtype=jnp.float32).at[:common_nv].set(qd[..., :common_nv])
    qacc_ptb = jnp.zeros(mjx_model_perturbed.nv, dtype=jnp.float32).at[:common_nv].set(qacc_src[..., :common_nv])
    tau_eff_ext_ptb = _external_tau_equivalent_or_zero(
        mjx_model_perturbed,
        q_ptb,
        qd_ptb,
        external_force_ee,
        external_force_body_id=external_force_body_id,
    )

    tau_corrected = dynamics.mjx_inverse_dynamics_rne(mjx_model_perturbed, q_ptb, qd_ptb, qacc_ptb)
    tau_corrected = tau_corrected - tau_eff_ext_ptb
    tau_corrected = inv_actuator_model(tau_corrected[..., :common_nu], q_ptb, qd_ptb[..., :common_nu], actuator_params)

    return tau_corrected[..., :common_nu]

def mjx_step_with_actuator(mjx_model:mjx.Model, data:mjx.Data, robot_params:ActuatorParams|RolloutParams=None) -> mjx.Data:
    '''
    Step the mujoco model with actuator model applied to the control inputs.
    '''
    if isinstance(robot_params, RolloutParams):
        mjx_model = robot_params.set_mjx_model(mjx_model)
    # apply actuator model
    tau_eff = actuator_model(data.ctrl, data.qpos, data.qvel, robot_params)

    # step mujoco with effective torques
    data = data.replace(ctrl=tau_eff)
    data = mjx.step(mjx_model, data)

    return data


if __name__=='__main__':
    import mujoco
    import numpy as np    

    # Ideal model: 7-DOF arm. Perturbed model: arm + gripper (extra joints/actuator)..
    mjx_model_ideal = dynamics.load_mjx_model_from_path('assets/franka_panda/panda.xml', remove_constraints=True)
    
    # remove all constraints for simplicity
    mjx_model_ideal:mjx.Model = mjx_model_ideal.tree_replace({
        "dof_frictionloss": jnp.zeros_like(mjx_model_ideal.dof_frictionloss),
        "actuator_gear": jnp.zeros_like(mjx_model_ideal.actuator_gear).at[...,0].set(1.0),
        "jnt_limited": np.zeros_like(mjx_model_ideal.jnt_limited),
        "jnt_actfrclimited": np.zeros_like(mjx_model_ideal.jnt_actfrclimited),
        "actuator_ctrllimited": np.zeros_like(mjx_model_ideal.actuator_ctrllimited),
        "actuator_actlimited": np.zeros_like(mjx_model_ideal.actuator_actlimited),
        "actuator_forcelimited": np.zeros_like(mjx_model_ideal.actuator_forcelimited),
    })
    
    # Build a perturbed model by lightly modifying mass/COM/damping of the ideal model.
    key = jax.random.PRNGKey(0)
    key, mass_key, com_key, damp_key, friction_key, q_key, qd_key, tau_key = jax.random.split(key, 8)

    mjx_model_perturbed = mjx_model_ideal.tree_replace({
        "body_mass": mjx_model_ideal.body_mass * (1.0 + 0.02 * jax.random.normal(mass_key, mjx_model_ideal.body_mass.shape)),
        "body_ipos": mjx_model_ideal.body_ipos + 0.01 * jax.random.normal(com_key, mjx_model_ideal.body_ipos.shape),
        "dof_damping": jax.random.uniform(damp_key, mjx_model_ideal.dof_damping.shape, minval=0.0, maxval=5.0),
    })

    # Random state and desired torque on the shared 7 DoF prefix.
    q_min, q_max = jnp.array(mjx_model_ideal.jnt_range[:, 0]), jnp.array(mjx_model_ideal.jnt_range[:, 1])
    q = jax.random.uniform(q_key, shape=(mjx_model_ideal.nq,), minval=q_min, maxval=q_max)
    qd = jax.random.normal(qd_key, shape=(mjx_model_ideal.nv,)) * 0.05
    tau_desired = jax.random.normal(tau_key, shape=(mjx_model_ideal.nu,)) * 10.0

    # Sample actuator params (deadzone + bias) over the shared actuators.
    key, deadzone_key, bias_key, damp_key = jax.random.split(key, 4)
    actuator_params = ActuatorParams(
        deadzone=jax.random.uniform(deadzone_key, shape=(mjx_model_ideal.nu,2), minval=0.0, maxval=0.2),
        torque_bias=jax.random.uniform(bias_key, shape=(mjx_model_ideal.nu,2), minval=-0.5, maxval=0.5),
        damping=jax.random.uniform(damp_key, shape=(mjx_model_ideal.nu,2), minval=0.0, maxval=5.0),
    )
    # Forward ideal model.
    data_ideal = mjx.forward(mjx_model_ideal, mjx.make_data(mjx_model_ideal).replace(qpos=q, qvel=qd, ctrl=tau_desired))
    qacc_ideal = data_ideal.qacc_smooth
    
    # Calculate torque command for the perturbed model that should match the ideal qacc.
    tau_gt = calculate_gt_torque(mjx_model_perturbed, q, qd, actuator_params=actuator_params, qacc=qacc_ideal)

    # Forward perturbed model with padded state/control (extra gripper joints/actuator left at zero).
    q_ptb = jnp.zeros(mjx_model_perturbed.nq, dtype=jnp.float32).at[:mjx_model_ideal.nq].set(q)
    qd_ptb = jnp.zeros(mjx_model_perturbed.nv, dtype=jnp.float32).at[:mjx_model_ideal.nv].set(qd)
    ctrl_ptb = jnp.zeros(mjx_model_perturbed.nu, dtype=jnp.float32).at[:tau_gt.shape[-1]].set(tau_gt)
    data_ptb = mjx_step_with_actuator(mjx_model_perturbed, mjx.make_data(mjx_model_perturbed).replace(qpos=q_ptb, qvel=qd_ptb, ctrl=ctrl_ptb), actuator_params)

    qacc_ptb = data_ptb.qacc_smooth[:mjx_model_ideal.nv]
    diff = jnp.max(jnp.abs(qacc_ideal - qacc_ptb))
    print(f"Max qacc diff on shared joints: {diff:.6f}")
    assert diff < 1e-3, "Shared joint accelerations should match between ideal and perturbed prefixes"
