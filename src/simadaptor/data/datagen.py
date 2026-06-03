import jax
import jax.numpy as jnp
from functools import partial
from typing import Any, Optional, Tuple, Dict
from dataclasses import dataclass

from mujoco import mjx

import simadaptor.core.structs as structs
import simadaptor.physics.dynamics as dynamics
import simadaptor.physics.rollout as rollout

_DEFAULT_EE_PAYLOAD_MASS_DELTA_RANGE = (0.0, 1.5)
# Keep these as plain Python tuples so dataloader worker imports do not
# trigger JAX backend selection at module-import time.
_DEFAULT_EE_PAYLOAD_COM_OFFSET_MIN_LOCAL_M = (-0.075, -0.075, -0.075)
_DEFAULT_EE_PAYLOAD_COM_OFFSET_MAX_LOCAL_M = (0.075, 0.075, 0.075)
_DEFAULT_JOINT_MODEL_MAJOR_EE_SCALE = 0.02
_DEFAULT_JOINT_MODEL_MAJOR_GLOBAL_SCALE = 0.02


def _profile_vector(
    name: str,
    values: jnp.ndarray,
    expected_size: int,
    *,
    dtype=jnp.float32,
) -> jnp.ndarray:
    arr = jnp.asarray(values, dtype=dtype).reshape((-1,))
    if int(arr.shape[0]) != int(expected_size):
        raise ValueError(
            f"{name} length mismatch: got {int(arr.shape[0])}, expected {int(expected_size)}."
        )
    return arr


def _resolve_range_2(
    range_vals: Tuple[float, float],
) -> Tuple[float, float]:
    if range_vals is None:
        raise ValueError("Range value is required and cannot be None.")
    lo, hi = float(range_vals[0]), float(range_vals[1])
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi


def _resolve_vector3_box(
    min_name: str,
    min_values: Optional[jnp.ndarray],
    max_name: str,
    max_values: Optional[jnp.ndarray],
    *,
    default_min: jnp.ndarray,
    default_max: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    min_arr = (
        jnp.asarray(default_min, dtype=jnp.float32)
        if min_values is None
        else jnp.asarray(min_values, dtype=jnp.float32).reshape((3,))
    )
    max_arr = (
        jnp.asarray(default_max, dtype=jnp.float32)
        if max_values is None
        else jnp.asarray(max_values, dtype=jnp.float32).reshape((3,))
    )
    if tuple(min_arr.shape) != (3,):
        raise ValueError(f"{min_name} must be length 3, got shape {min_arr.shape}.")
    if tuple(max_arr.shape) != (3,):
        raise ValueError(f"{max_name} must be length 3, got shape {max_arr.shape}.")
    return jnp.minimum(min_arr, max_arr), jnp.maximum(min_arr, max_arr)


def _resolve_ee_body_id(model: mjx.Model) -> jnp.ndarray:
    if model.nsite > 0:
        return jnp.asarray(model.site_bodyid[-1], dtype=jnp.int32)
    ee_dof_id = jnp.maximum(jnp.minimum(model.nu, model.nv) - 1, 0)
    return jnp.asarray(model.dof_bodyid[ee_dof_id], dtype=jnp.int32)


def _point_mass_inertia_diag(mass: jnp.ndarray, offset: jnp.ndarray) -> jnp.ndarray:
    offset = jnp.asarray(offset, dtype=jnp.float32).reshape((3,))
    ox, oy, oz = offset
    diag = jnp.asarray(
        (oy * oy + oz * oz, ox * ox + oz * oz, ox * ox + oy * oy),
        dtype=offset.dtype,
    )
    return jnp.asarray(mass, dtype=offset.dtype) * diag


def _resolve_nonnegative_scale(
    name: str,
    value: Optional[float],
    default: float,
) -> float:
    resolved = default if value is None else float(value)
    if resolved < 0.0:
        raise ValueError(f"{name} must be non-negative, got {resolved}.")
    return resolved


def _compose_body_payload_randomization(
    *,
    model_body_mass: jnp.ndarray,
    model_body_inertia: jnp.ndarray,
    model_body_ipos: jnp.ndarray,
    raw_body_mass_scale: jnp.ndarray,
    raw_ipos_noise: jnp.ndarray,
    ee_body_id: jnp.ndarray,
    raw_ee_payload_mass_delta: jnp.ndarray,
    raw_ee_payload_offset_local: jnp.ndarray,
    is_joint_model_major: jnp.ndarray,
    joint_model_major_global_scale: float,
    joint_model_major_ee_scale: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compose physically coherent body mass, inertia, COM, and EE payload params."""
    body_mass_dtype = jnp.asarray(model_body_mass).dtype
    global_scale = jnp.asarray(joint_model_major_global_scale, dtype=body_mass_dtype)
    one = jnp.asarray(1.0, dtype=body_mass_dtype)
    body_mass_scale = jnp.where(
        is_joint_model_major,
        one + global_scale * (raw_body_mass_scale - one),
        raw_body_mass_scale,
    )
    body_mass_base = model_body_mass * body_mass_scale

    safe_mass = jnp.maximum(model_body_mass, 1e-8)
    inertia_scale = jnp.where(model_body_mass > 0, body_mass_base / safe_mass, 1.0)
    body_inertia_base = model_body_inertia * inertia_scale[:, None]

    ipos_scale = jnp.where(is_joint_model_major, global_scale, one)
    ipos_noise = raw_ipos_noise * ipos_scale
    mass_mask = (model_body_mass > 0).astype(jnp.asarray(model_body_ipos).dtype)[:, None]
    body_ipos_base = model_body_ipos + ipos_noise * mass_mask

    ee_scale = jnp.asarray(joint_model_major_ee_scale, dtype=body_mass_dtype)
    payload_scale = jnp.where(is_joint_model_major, ee_scale, one)
    ee_payload_mass_delta = raw_ee_payload_mass_delta * payload_scale
    ee_payload_offset_local = raw_ee_payload_offset_local * payload_scale

    ee_base_mass = body_mass_base[ee_body_id]
    ee_total_mass = jnp.maximum(ee_base_mass + ee_payload_mass_delta, 1e-2)
    ee_com_shift = jnp.where(
        ee_total_mass > 0,
        ee_payload_offset_local * (ee_payload_mass_delta / ee_total_mass),
        jnp.zeros_like(ee_payload_offset_local),
    )
    ee_payload_rel_to_total_com = ee_payload_offset_local - ee_com_shift
    ee_total_inertia = (
        body_inertia_base[ee_body_id]
        + _point_mass_inertia_diag(ee_base_mass, ee_com_shift)
        + _point_mass_inertia_diag(ee_payload_mass_delta, ee_payload_rel_to_total_com)
    )
    ee_total_inertia = jnp.maximum(
        ee_total_inertia, jnp.asarray(1e-8, dtype=ee_total_inertia.dtype)
    )

    ee_has_mass = jnp.asarray(model_body_mass[ee_body_id] > 0, dtype=jnp.asarray(model_body_ipos).dtype)
    body_mass = body_mass_base.at[ee_body_id].set(ee_total_mass)
    body_inertia = body_inertia_base.at[ee_body_id].set(ee_total_inertia)
    body_ipos = body_ipos_base.at[ee_body_id].set(
        model_body_ipos[ee_body_id] + ee_com_shift * ee_has_mass
    )
    return body_mass, body_inertia, body_ipos


def _actuator_mass_diag_from_model(model: mjx.Model, dof_mass_diag: jnp.ndarray) -> jnp.ndarray:
    """Map dof-space diag(M) at q0 to one approximate inertia value per actuator."""
    dof_mass_diag = _profile_vector("dof_mass_diag", dof_mass_diag, int(model.nv), dtype=jnp.float32)
    if int(model.nu) == 0:
        return jnp.zeros((0,), dtype=dof_mass_diag.dtype)

    actuator_trnid = jnp.asarray(model.actuator_trnid, dtype=jnp.int32)
    jnt_dofadr = jnp.asarray(model.jnt_dofadr, dtype=jnp.int32)

    act_jnt_id = actuator_trnid[:, 0]
    act_jnt_id_clamped = jnp.clip(act_jnt_id, 0, int(model.njnt) - 1)
    act_dof_id = jnt_dofadr[act_jnt_id_clamped]

    valid_jnt = (act_jnt_id >= 0) & (act_jnt_id < int(model.njnt))
    valid_dof = (act_dof_id >= 0) & (act_dof_id < int(model.nv))
    valid = valid_jnt & valid_dof

    act_dof_id_clamped = jnp.clip(act_dof_id, 0, int(model.nv) - 1)
    mapped_mass = dof_mass_diag[act_dof_id_clamped]

    # Fallback: positional map when transmission joint id is unavailable/non-standard.
    fallback_dof_id = jnp.minimum(jnp.arange(model.nu, dtype=jnp.int32), int(model.nv) - 1)
    fallback_mass = dof_mass_diag[fallback_dof_id]
    actuator_mass_diag = jnp.where(valid, mapped_mass, fallback_mass)
    return jnp.maximum(actuator_mass_diag, jnp.asarray(1e-6, dtype=actuator_mass_diag.dtype))


def sample_sigmoid_params_asym(key, tau_lim: jnp.ndarray):
    """Sample asymmetric sigmoidal friction parameters per joint.

    Returns A_pos, k_pos, s_pos, A_neg, k_neg, s_neg (each shape [n_joints]).
    """
    k1, k2, k3, k4 = jax.random.split(key, 4)

    base_A = jax.random.uniform(k1, tau_lim.shape + (2,), minval=0.005, maxval=3.0)
    A_pos, A_neg = base_A[...,0], base_A[...,1]

    # slope: sample velocity width in [0.02, 0.2] rad/s
    dv = jax.random.uniform(k3, tau_lim.shape, minval=0.02, maxval=0.2)
    k0 = 4.4 / dv
    delta_logk = jax.random.uniform(k4, tau_lim.shape, minval=-0.3, maxval=0.3)
    k_pos = k0 * 10.0 ** (delta_logk)
    k_neg = k0 * 10.0 ** (-delta_logk)

    # shift
    key_s1, key_s2 = jax.random.split(k4)
    s_pos = jax.random.uniform(key_s1, tau_lim.shape, minval=-0.02, maxval=0.02)
    eps_s = jax.random.uniform(key_s2, tau_lim.shape, minval=-0.01, maxval=0.01)
    s_neg = -s_pos + eps_s

    return A_pos, k_pos, s_pos, A_neg, k_neg, s_neg


# =========================
# Domain randomization
# =========================
def sample_random_params(
    rng,
    model: mjx.Model,
    evaluation_mode=False,
    armature_min_profile: jnp.ndarray = None,
    armature_max_profile: jnp.ndarray = None,
    base_kp_profile: jnp.ndarray = None,
    white_base_profile: jnp.ndarray = None,
    walk_base_profile: jnp.ndarray = None,
    white_scale_range: Tuple[float, float] = None,
    walk_scale_range: Tuple[float, float] = None,
    kp_scale_small_range: Tuple[float, float] = None,
    kp_scale_large_range: Tuple[float, float] = None,
    kd_scale_range: Optional[Tuple[float, float]] = None,
    kp_small_prob: float = None,
    actuator_mass_diag: jnp.ndarray = None,
    ee_payload_mass_delta_range: Optional[Tuple[float, float]] = None,
    ee_payload_com_offset_min_local_m: Optional[jnp.ndarray] = None,
    ee_payload_com_offset_max_local_m: Optional[jnp.ndarray] = None,
    joint_model_major_ee_scale: Optional[float] = None,
    joint_model_major_global_scale: Optional[float] = None,
):
    required_params = {
        "armature_min_profile": armature_min_profile,
        "armature_max_profile": armature_max_profile,
        "base_kp_profile": base_kp_profile,
        "white_base_profile": white_base_profile,
        "walk_base_profile": walk_base_profile,
        "white_scale_range": white_scale_range,
        "walk_scale_range": walk_scale_range,
        "kp_scale_small_range": kp_scale_small_range,
        "kp_scale_large_range": kp_scale_large_range,
        "kp_small_prob": kp_small_prob,
    }
    missing = [k for k, v in required_params.items() if v is None]
    if missing:
        raise ValueError(
            "sample_random_params requires explicit datagen profile values; "
            f"missing keys: {missing}"
        )
    kp_small_prob = float(kp_small_prob)
    if kp_small_prob < 0.0 or kp_small_prob > 1.0:
        raise ValueError(f"kp_small_prob must be in [0, 1], got {kp_small_prob}.")
    joint_model_major_ee_scale = _resolve_nonnegative_scale(
        "joint_model_major_ee_scale",
        joint_model_major_ee_scale,
        _DEFAULT_JOINT_MODEL_MAJOR_EE_SCALE,
    )
    joint_model_major_global_scale = _resolve_nonnegative_scale(
        "joint_model_major_global_scale",
        joint_model_major_global_scale,
        _DEFAULT_JOINT_MODEL_MAJOR_GLOBAL_SCALE,
    )
    ee_payload_mass_delta_range = (
        _DEFAULT_EE_PAYLOAD_MASS_DELTA_RANGE
        if ee_payload_mass_delta_range is None
        else ee_payload_mass_delta_range
    )
    ee_payload_mass_delta_lo, ee_payload_mass_delta_hi = _resolve_range_2(
        ee_payload_mass_delta_range
    )
    ee_payload_com_offset_min_local_m, ee_payload_com_offset_max_local_m = _resolve_vector3_box(
        "ee_payload_com_offset_min_local_m",
        ee_payload_com_offset_min_local_m,
        "ee_payload_com_offset_max_local_m",
        ee_payload_com_offset_max_local_m,
        default_min=_DEFAULT_EE_PAYLOAD_COM_OFFSET_MIN_LOCAL_M,
        default_max=_DEFAULT_EE_PAYLOAD_COM_OFFSET_MAX_LOCAL_M,
    )

    rng_armature, rng = jax.random.split(rng)
    armature_min = _profile_vector(
        "armature_min_profile", armature_min_profile, int(model.nv), dtype=jnp.float32
    )
    armature_max = _profile_vector(
        "armature_max_profile", armature_max_profile, int(model.nv), dtype=jnp.float32
    )
    random_armature = jax.random.uniform(
        rng_armature,
        (model.nv,),
        minval=armature_min,
        maxval=armature_max,
        dtype=jnp.float32,
    )

    rng_body_mass, rng = jax.random.split(rng)
    random_body_mass_scale = jax.random.uniform(rng_body_mass, (model.nbody,), minval=0.9, maxval=1.1)

    ee_body_id = _resolve_ee_body_id(model)

    # Random sample that makes major model difference from actuator models.
    rng_joint_major, rng = jax.random.split(rng)
    is_joint_model_major = jax.random.uniform(rng_joint_major) < 0.4

    # Randomize COM position (body_ipos) with a small perturbation for non-EE bodies.
    rng_ipos, rng = jax.random.split(rng)
    ipos_noise = jax.random.uniform(rng_ipos, (model.nbody, 3), minval=-0.01, maxval=0.01)

    # Add an EE-local payload whose COM offset drives the EE body's COM/inertia together.
    rng_ee_mass, rng = jax.random.split(rng)
    ee_payload_mass_delta = jax.random.uniform(
        rng_ee_mass,
        (),
        minval=ee_payload_mass_delta_lo,
        maxval=ee_payload_mass_delta_hi,
        dtype=jnp.float32,
    )

    rng_ee_offset, rng = jax.random.split(rng)
    ee_payload_offset_local = jax.random.uniform(
        rng_ee_offset,
        (3,),
        minval=ee_payload_com_offset_min_local_m,
        maxval=ee_payload_com_offset_max_local_m,
        dtype=jnp.float32,
    )
    random_body_mass, random_body_inertia, random_body_ipos = _compose_body_payload_randomization(
        model_body_mass=model.body_mass,
        model_body_inertia=model.body_inertia,
        model_body_ipos=model.body_ipos,
        raw_body_mass_scale=random_body_mass_scale,
        raw_ipos_noise=ipos_noise,
        ee_body_id=ee_body_id,
        raw_ee_payload_mass_delta=ee_payload_mass_delta,
        raw_ee_payload_offset_local=ee_payload_offset_local,
        is_joint_model_major=is_joint_model_major,
        joint_model_major_global_scale=joint_model_major_global_scale,
        joint_model_major_ee_scale=joint_model_major_ee_scale,
    )
    
    perterbed_mjx_params = {
        "dof_frictionloss": jnp.zeros_like(model.dof_frictionloss),
        "dof_armature": random_armature,
        "dof_damping": jnp.zeros_like(model.dof_damping),
        "body_mass": random_body_mass,
        "body_inertia": random_body_inertia,
        "body_ipos": random_body_ipos,
    }
    
    original_mjx_params = {
        "dof_frictionloss": jnp.zeros_like(model.dof_frictionloss),
        "dof_armature": model.dof_armature,
        "dof_damping": model.dof_damping,
        "body_mass": model.body_mass,
        "body_inertia": model.body_inertia,
        "body_ipos": model.body_ipos,
    }

    rng, rng_Kp1, rng_Kp2, rng_Kp3 = jax.random.split(rng,4)
    base_Kp = _profile_vector("base_kp_profile", base_kp_profile, int(model.nu), dtype=jnp.float32)
    kp_small_lo, kp_small_hi = _resolve_range_2(kp_scale_small_range)
    kp_large_lo, kp_large_hi = _resolve_range_2(kp_scale_large_range)
    random_Kp1 = jax.random.uniform(rng_Kp1, (model.nu,), minval=kp_small_lo, maxval=kp_small_hi)
    random_Kp2 = jax.random.uniform(rng_Kp2, (model.nu,), minval=kp_large_lo, maxval=kp_large_hi)
    random_Kp = jnp.where(jax.random.uniform(rng_Kp3, shape=(1,)) < kp_small_prob, random_Kp1, random_Kp2)
    random_Kp = base_Kp * random_Kp

    if actuator_mass_diag is None:
        dof_mass_diag = dynamics.get_mass_matrix_diag_at_qpos0(model)
        actuator_mass_diag = _actuator_mass_diag_from_model(model, dof_mass_diag)
    else:
        actuator_mass_diag = _profile_vector(
            "actuator_mass_diag", actuator_mass_diag, int(model.nu), dtype=jnp.float32
        )
    
    if kd_scale_range is None:
        kd_scale_lo, kd_scale_hi = 0.5, 1.5
    else:
        kd_scale_lo, kd_scale_hi = _resolve_range_2(kd_scale_range)

    rng, rng_Kd_scale = jax.random.split(rng)
    random_Kd_scale = jax.random.uniform(
        rng_Kd_scale,
        (model.nu,),
        minval=kd_scale_lo,
        maxval=kd_scale_hi,
    )
    random_Kd = random_Kd_scale * 2.0 * jnp.sqrt(
        jnp.maximum(random_Kp, 1e-6) * jnp.maximum(actuator_mass_diag, 1e-6)
    )
    
    # actuator params    
    rng_damping, rng = jax.random.split(rng)
    random_act_damping = jax.random.uniform(rng_damping, (model.nv, 2), minval=0.0, maxval=2.0, dtype=jnp.float32)

    rng_deadzone, rng = jax.random.split(rng)
    random_deadzone = jax.random.uniform(rng_deadzone, (model.nu,2), minval=0.0, maxval=1.0)
    
    rng_torque_bias1, rng_torque_bias2, rng = jax.random.split(rng, 3)
    random_torque_bias_pos = jax.random.uniform(rng_torque_bias1, (model.nu,1), minval=-1.0, maxval=1.0)
    random_torque_bias_neg = random_torque_bias_pos + jax.random.uniform(rng_torque_bias2, (model.nu,1), minval=-0.2, maxval=0.2)
    random_torque_bias = jnp.concat([random_torque_bias_pos, random_torque_bias_neg], axis=-1) # same bias for pos and neg direction

    rng_friction, rng = jax.random.split(rng)
    torque_lim = jnp.max(jnp.abs(model.actuator_forcerange), axis=-1)
    torque_lim = jnp.where(jnp.isfinite(torque_lim), torque_lim, 1.0)
    A_pos, k_pos, s_pos, A_neg, k_neg, s_neg = sample_sigmoid_params_asym(rng_friction, torque_lim)
    random_friction_params = jnp.stack([A_neg, A_pos, k_neg, k_pos, s_neg, s_pos], axis=-1)
    
    rng_noise_white, rng = jax.random.split(rng)
    rng_noise_walk, rng = jax.random.split(rng)
    rng_tau_noise_type, rng = jax.random.split(rng)
    white_base = _profile_vector(
        "white_base_profile",
        white_base_profile,
        int(model.nu),
        dtype=jnp.float32,
    )
    walk_base = _profile_vector(
        "walk_base_profile",
        walk_base_profile,
        int(model.nu),
        dtype=jnp.float32,
    )
    white_scale_lo, white_scale_hi = _resolve_range_2(white_scale_range)
    walk_scale_lo, walk_scale_hi = _resolve_range_2(walk_scale_range)
    white_scale = jax.random.uniform(rng_noise_white, (model.nu,), minval=white_scale_lo, maxval=white_scale_hi)
    walk_scale = jax.random.uniform(rng_noise_walk, (model.nu,), minval=walk_scale_lo, maxval=walk_scale_hi)
    torque_noise_std = jnp.stack([white_base * white_scale, walk_base * walk_scale], axis=0)
    random_tau_noise_type = jax.random.choice(rng_tau_noise_type, jnp.array([0, 1, 2]).astype(jnp.int32), p=jnp.array([0.1, 0.45, 0.45]))

    rng_scale1, rng = jax.random.split(rng)
    random_torque_scale = jax.random.uniform(rng_scale1, (model.nu, 2), minval=0.99, maxval=1.01)
    
    shared_actuator_params = {"kp": random_Kp, "kd": random_Kd, "torque_range": model.actuator_forcerange, "joint_range": model.jnt_range}
    hidden_actuator_params = {
        "deadzone": random_deadzone,
        "torque_bias": random_torque_bias,
        "damping": random_act_damping,
        "friction_params": random_friction_params,
        "torque_scale": random_torque_scale,
    }
    rng_q_noise, rng = jax.random.split(rng)
    rng_dq_noise, rng = jax.random.split(rng)
    q_noise_std = jax.random.uniform(rng_q_noise, (model.nv,), minval=1e-4, maxval=1e-3)
    dq_noise_std = jax.random.uniform(rng_dq_noise, (model.nv,), minval=1e-3, maxval=1e-2)

    original_params = {
        "torque_noise_type": random_tau_noise_type,
        "torque_noise_std": torque_noise_std,
        "q_noise_std": q_noise_std,
        "dq_noise_std": dq_noise_std,
        **original_mjx_params,
        **shared_actuator_params,
    }
    perturbed_params = {
        "torque_noise_type": random_tau_noise_type,
        "torque_noise_std": torque_noise_std,
        "q_noise_std": q_noise_std,
        "dq_noise_std": dq_noise_std,
        **hidden_actuator_params,
        **perterbed_mjx_params,
        **shared_actuator_params,
    }
    return original_params, perturbed_params, rng


def data_generation(
    rng,
    mjx_model,
    dof_idx_arm,
    batch_size,
    num_waypoints,
    duration,
    dt: float = 0.001,
    filter_key=False,
    evaluation_mode=False,
    pause_prob=0.30,
    external_force_body_id: Optional[int] = None,
    external_force_num_impulses: int = 0,
    external_force_magnitude_min_n: float = 0.0,
    external_force_magnitude_max_n: float = 0.0,
    external_force_duration_min_s: float = 0.05,
    external_force_duration_max_s: float = 0.20,
    external_force_position_min_local_m: Optional[jnp.ndarray] = None,
    external_force_position_max_local_m: Optional[jnp.ndarray] = None,
    external_force_apply_to_perturbed: bool = True,
    external_force_apply_to_original: bool = True,
    armature_min_profile: Optional[jnp.ndarray] = None,
    armature_max_profile: Optional[jnp.ndarray] = None,
    base_kp_profile: Optional[jnp.ndarray] = None,
    white_base_profile: Optional[jnp.ndarray] = None,
    walk_base_profile: Optional[jnp.ndarray] = None,
    white_scale_range: Optional[Tuple[float, float]] = None,
    walk_scale_range: Optional[Tuple[float, float]] = None,
    kp_scale_small_range: Optional[Tuple[float, float]] = None,
    kp_scale_large_range: Optional[Tuple[float, float]] = None,
    kd_scale_range: Optional[Tuple[float, float]] = None,
    kp_small_prob: Optional[float] = None,
    waypoint_max_delta_deg_profile: Optional[jnp.ndarray] = None,
    rollout_cmd_noise_std: Optional[float] = None,
    ee_payload_mass_delta_range: Optional[Tuple[float, float]] = None,
    ee_payload_com_offset_min_local_m: Optional[jnp.ndarray] = None,
    ee_payload_com_offset_max_local_m: Optional[jnp.ndarray] = None,
    joint_model_major_ee_scale: Optional[float] = None,
    joint_model_major_global_scale: Optional[float] = None,
) -> Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray], structs.RolloutParams, structs.RolloutParams]:
    del rollout_cmd_noise_std
    required_params = {
        "armature_min_profile": armature_min_profile,
        "armature_max_profile": armature_max_profile,
        "base_kp_profile": base_kp_profile,
        "white_base_profile": white_base_profile,
        "walk_base_profile": walk_base_profile,
        "white_scale_range": white_scale_range,
        "walk_scale_range": walk_scale_range,
        "kp_scale_small_range": kp_scale_small_range,
        "kp_scale_large_range": kp_scale_large_range,
        "kp_small_prob": kp_small_prob,
    }
    missing = [k for k, v in required_params.items() if v is None]
    if missing:
        raise ValueError(
            "data_generation requires explicit datagen profile values; "
            f"missing keys: {missing}"
        )

    dof_mass_diag_q0 = dynamics.get_mass_matrix_diag_at_qpos0(mjx_model)
    actuator_mass_diag_q0 = _actuator_mass_diag_from_model(mjx_model, dof_mass_diag_q0)

    rng, perterbe_rng = jax.random.split(rng)
    original_params, perturbed_params, _ = jax.vmap(
        partial(
            sample_random_params,
            model=mjx_model,
            evaluation_mode=evaluation_mode,
            armature_min_profile=armature_min_profile,
            armature_max_profile=armature_max_profile,
            base_kp_profile=base_kp_profile,
            white_base_profile=white_base_profile,
            walk_base_profile=walk_base_profile,
            white_scale_range=white_scale_range,
            walk_scale_range=walk_scale_range,
            kp_scale_small_range=kp_scale_small_range,
            kp_scale_large_range=kp_scale_large_range,
            kd_scale_range=kd_scale_range,
            kp_small_prob=kp_small_prob,
            actuator_mass_diag=actuator_mass_diag_q0,
            ee_payload_mass_delta_range=ee_payload_mass_delta_range,
            ee_payload_com_offset_min_local_m=ee_payload_com_offset_min_local_m,
            ee_payload_com_offset_max_local_m=ee_payload_com_offset_max_local_m,
            joint_model_major_ee_scale=joint_model_major_ee_scale,
            joint_model_major_global_scale=joint_model_major_global_scale,
        )
    )(jax.random.split(perterbe_rng, batch_size))

    original_rollout_params = structs.RolloutParams(**original_params)
    perturbed_rollout_params = structs.RolloutParams(**perturbed_params)

    # (2) History inputs from perturbed model rollout
    rollout_key, rng = jax.random.split(rng)
    rollout_inputs = rollout.rollout_generation(
        rollout_key,
        perturbed_rollout_params.controller_params.get_actuator_fn(
            control_type='qref',
            add_noise=not evaluation_mode,
            ideal_mjx_model=mjx_model,
        ),
        mjx_model,
        dof_idx_arm,
        perturbed_rollout_params,
        num_waypoints=num_waypoints,
        duration=duration,
        dt=dt,
        pause_prob=pause_prob,
        external_force_body_id=external_force_body_id,
        external_force_num_impulses=(external_force_num_impulses if external_force_apply_to_perturbed else 0),
        external_force_magnitude_min_n=external_force_magnitude_min_n,
        external_force_magnitude_max_n=external_force_magnitude_max_n,
        external_force_duration_min_s=external_force_duration_min_s,
        external_force_duration_max_s=external_force_duration_max_s,
        external_force_position_min_local_m=external_force_position_min_local_m,
        external_force_position_max_local_m=external_force_position_max_local_m,
        waypoint_max_delta_deg_profile=waypoint_max_delta_deg_profile,
    )
    rollout_inputs = jax.lax.stop_gradient(rollout_inputs)
    

    # (3) Target rollouts on original model (batched)
    test_rollout_key, rng = jax.random.split(rng)
    test_batch_rollout_generation = partial(
        rollout.rollout_generation,
        actuator_fn=original_rollout_params.controller_params.get_actuator_fn(
            control_type='qref',
            add_noise=not evaluation_mode,
            ideal_mjx_model=mjx_model,
        ),
        model_mjx=mjx_model,
        dof_idx_arm=dof_idx_arm,
        batched_rollout_params=original_rollout_params,
        num_waypoints=num_waypoints,
        duration=duration,
        dt=dt,
        pause_prob=pause_prob,
        external_force_body_id=external_force_body_id,
        external_force_num_impulses=(external_force_num_impulses if external_force_apply_to_original else 0),
        external_force_magnitude_min_n=external_force_magnitude_min_n,
        external_force_magnitude_max_n=external_force_magnitude_max_n,
        external_force_duration_min_s=external_force_duration_min_s,
        external_force_duration_max_s=external_force_duration_max_s,
        external_force_position_min_local_m=external_force_position_min_local_m,
        external_force_position_max_local_m=external_force_position_max_local_m,
        waypoint_max_delta_deg_profile=waypoint_max_delta_deg_profile,
    )
    test_rollout = test_batch_rollout_generation(test_rollout_key)
    test_rollout = jax.lax.stop_gradient(test_rollout)
    
    if filter_key:
        perturbed_rollout_keys = ["q", "qd", "u", "times", "external_force_ee"]
        original_rollout_keys = ["q", "qd", "qdd", "q_ref", "qd_ref", "times", "external_force_ee"]
        rollout_inputs = {k: rollout_inputs[k] for k in perturbed_rollout_keys}
        test_rollout = {k: test_rollout[k] for k in original_rollout_keys}

    return rollout_inputs, test_rollout, perturbed_rollout_params, original_rollout_params
