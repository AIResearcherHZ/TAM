import inspect
from pathlib import Path
import sys
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.repo_paths import REPO_ROOT as ROOT
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from mujoco import mjx as _mjx  # noqa: F401
    import simadaptor.core.structs as structs
    import simadaptor.data.datagen as datagen
    from simadaptor.data.datagen_profiles import derive_robot_key, load_datagen_profile
    from simadaptor.eval.gt_tau_cmd_validation import compute_gt_tau_cmd
    import simadaptor.physics.dynamics as dynamics

    MJX_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment dependent
    structs = None
    datagen = None
    derive_robot_key = None
    load_datagen_profile = None
    compute_gt_tau_cmd = None
    dynamics = None
    MJX_IMPORT_ERROR = exc


XML_PATH = ROOT / "assets" / "franka_panda" / "panda_pandagripper.xml"
PROFILE_TABLE_PATH = ROOT / "assets" / "datagen_profiles.json"


def _safe_joint_ranges(model) -> tuple[jnp.ndarray, jnp.ndarray]:
    q_min_raw = jnp.asarray(model.jnt_range[:, 0], dtype=jnp.float32)
    q_max_raw = jnp.asarray(model.jnt_range[:, 1], dtype=jnp.float32)
    nq = int(model.nq)
    if q_min_raw.shape[0] < nq:
        pad = nq - int(q_min_raw.shape[0])
        q_min_raw = jnp.pad(q_min_raw, (0, pad), constant_values=-jnp.pi)
        q_max_raw = jnp.pad(q_max_raw, (0, pad), constant_values=jnp.pi)
    else:
        q_min_raw = q_min_raw[:nq]
        q_max_raw = q_max_raw[:nq]
    q_min = jnp.where(jnp.isfinite(q_min_raw), q_min_raw, -jnp.pi)
    q_max = jnp.where(jnp.isfinite(q_max_raw), q_max_raw, jnp.pi)
    return q_min, q_max


def _torque_bounds(torque_range, model, tau_abs_default: float = 30.0) -> tuple[jnp.ndarray, jnp.ndarray]:
    if torque_range is None or torque_range.ndim < 2 or torque_range.shape[-1] != 2:
        torque_range = jnp.asarray(model.actuator_forcerange, dtype=jnp.float32)
    else:
        torque_range = jnp.asarray(torque_range, dtype=jnp.float32)
    n = min(int(model.nu), int(torque_range.shape[-2]))
    lo = jnp.minimum(torque_range[:n, 0], torque_range[:n, 1])
    hi = jnp.maximum(torque_range[:n, 0], torque_range[:n, 1])
    lo = jnp.where(jnp.isfinite(lo), lo, -float(tau_abs_default))
    hi = jnp.where(jnp.isfinite(hi), hi, float(tau_abs_default))
    return lo, hi


def _build_ideal_model(ideal_model_has_gravity: bool):
    model = dynamics.load_mjx_model_from_path(str(XML_PATH), remove_constraints=True)
    if ideal_model_has_gravity:
        return model.replace(body_gravcomp=jnp.zeros_like(model.body_gravcomp))
    return model.replace(body_gravcomp=jnp.ones_like(model.body_gravcomp).at[..., 0].set(0))


def _profile_kwargs():
    robot_key = derive_robot_key(XML_PATH)
    _, datagen_profile_kwargs = load_datagen_profile(
        table_path=PROFILE_TABLE_PATH,
        robot_key=robot_key,
        profile_key=None,
    )
    sample_keys = set(inspect.signature(datagen.sample_random_params).parameters.keys())
    return {
        key: value
        for key, value in datagen_profile_kwargs.items()
        if key in sample_keys
    }


def _sample_rollout_params(key: jax.Array, ideal_model, *, ideal_model_has_gravity: bool):
    _, perturbed_params_dict, _ = datagen.sample_random_params(
        key,
        ideal_model,
        evaluation_mode=True,
        ideal_model_has_gravity=ideal_model_has_gravity,
        **_profile_kwargs(),
    )
    return structs.RolloutParams(**perturbed_params_dict)


def _sample_state_and_tau(
    key: jax.Array,
    ideal_model,
    rollout_params,
    *,
    num_samples: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    key_q, key_qd, key_tau = jax.random.split(key, 3)
    q_min, q_max = _safe_joint_ranges(ideal_model)
    q = jax.random.uniform(key_q, shape=(int(ideal_model.nq),), minval=q_min, maxval=q_max)
    qd = jax.random.normal(key_qd, shape=(int(ideal_model.nv),), dtype=q.dtype) * 0.25
    lo, hi = _torque_bounds(rollout_params.torque_range, ideal_model)
    tau = jax.random.uniform(
        key_tau,
        shape=(int(num_samples), int(ideal_model.nu)),
        minval=lo,
        maxval=hi,
        dtype=lo.dtype,
    )
    if int(num_samples) == 1:
        tau = tau[0]
    return q, qd, tau


def _sample_external_force(key: jax.Array, *, batch_size: Optional[int] = None) -> jax.Array:
    shape = (3,) if batch_size is None else (int(batch_size), 3)
    return jax.random.normal(key, shape=shape, dtype=jnp.float32) * 5.0


def _sample_external_wrench(key: jax.Array, *, batch_size: Optional[int] = None) -> jax.Array:
    shape = (6,) if batch_size is None else (int(batch_size), 6)
    return jax.random.normal(key, shape=shape, dtype=jnp.float32) * 5.0


def _stack_rollout_params(*params):
    return jax.tree_util.tree_map(
        lambda *xs: None if xs[0] is None else jnp.stack(xs, axis=0),
        *params,
    )


@pytest.mark.skipif(MJX_IMPORT_ERROR is not None, reason=f"MJX unavailable: {MJX_IMPORT_ERROR}")
@pytest.mark.parametrize("ideal_model_has_gravity", [True, False])
def test_compute_gt_tau_cmd_single_sample_matches_reference(ideal_model_has_gravity: bool):
    key = jax.random.PRNGKey(0 if ideal_model_has_gravity else 1)
    model = _build_ideal_model(ideal_model_has_gravity)
    key_params, key_sample = jax.random.split(key)
    rollout_params = _sample_rollout_params(key_params, model, ideal_model_has_gravity=ideal_model_has_gravity)
    q, qd, tau = _sample_state_and_tau(key_sample, model, rollout_params, num_samples=1)

    tau_ref = compute_gt_tau_cmd(model, rollout_params, q, qd, tau, method="reference")
    tau_fast = compute_gt_tau_cmd(model, rollout_params, q, qd, tau, method="shared_linear")

    assert tau_fast.shape == tau.shape
    np.testing.assert_allclose(np.asarray(tau_fast), np.asarray(tau_ref), rtol=2e-4, atol=2e-4)


@pytest.mark.skipif(MJX_IMPORT_ERROR is not None, reason=f"MJX unavailable: {MJX_IMPORT_ERROR}")
@pytest.mark.parametrize("ideal_model_has_gravity", [True, False])
def test_compute_gt_tau_cmd_multi_sample_matches_reference(ideal_model_has_gravity: bool):
    key = jax.random.PRNGKey(10 if ideal_model_has_gravity else 11)
    model = _build_ideal_model(ideal_model_has_gravity)
    key_params, key_sample = jax.random.split(key)
    rollout_params = _sample_rollout_params(key_params, model, ideal_model_has_gravity=ideal_model_has_gravity)
    q, qd, tau = _sample_state_and_tau(key_sample, model, rollout_params, num_samples=8)

    tau_ref = compute_gt_tau_cmd(model, rollout_params, q, qd, tau, method="reference")
    tau_fast = compute_gt_tau_cmd(model, rollout_params, q, qd, tau, method="shared_linear")

    assert tau_fast.shape == tau.shape
    np.testing.assert_allclose(np.asarray(tau_fast), np.asarray(tau_ref), rtol=2e-4, atol=2e-4)


@pytest.mark.skipif(MJX_IMPORT_ERROR is not None, reason=f"MJX unavailable: {MJX_IMPORT_ERROR}")
@pytest.mark.parametrize("ideal_model_has_gravity", [True, False])
def test_compute_gt_tau_cmd_batched_multi_sample_matches_reference(ideal_model_has_gravity: bool):
    model = _build_ideal_model(ideal_model_has_gravity)
    keys = jax.random.split(jax.random.PRNGKey(100 if ideal_model_has_gravity else 101), 4)
    rollout_params_a = _sample_rollout_params(keys[0], model, ideal_model_has_gravity=ideal_model_has_gravity)
    rollout_params_b = _sample_rollout_params(keys[1], model, ideal_model_has_gravity=ideal_model_has_gravity)
    q_a, qd_a, tau_a = _sample_state_and_tau(keys[2], model, rollout_params_a, num_samples=6)
    q_b, qd_b, tau_b = _sample_state_and_tau(keys[3], model, rollout_params_b, num_samples=6)

    rollout_params_batch = _stack_rollout_params(rollout_params_a, rollout_params_b)
    q_batch = jnp.stack([q_a, q_b], axis=0)
    qd_batch = jnp.stack([qd_a, qd_b], axis=0)
    tau_batch = jnp.stack([tau_a, tau_b], axis=0)

    tau_ref = compute_gt_tau_cmd(model, rollout_params_batch, q_batch, qd_batch, tau_batch, method="reference")
    tau_fast = compute_gt_tau_cmd(model, rollout_params_batch, q_batch, qd_batch, tau_batch, method="shared_linear")

    assert tau_fast.shape == tau_batch.shape
    np.testing.assert_allclose(np.asarray(tau_fast), np.asarray(tau_ref), rtol=3e-4, atol=3e-4)


@pytest.mark.skipif(MJX_IMPORT_ERROR is not None, reason=f"MJX unavailable: {MJX_IMPORT_ERROR}")
@pytest.mark.parametrize("ideal_model_has_gravity", [True, False])
def test_compute_gt_tau_cmd_forceaware_single_sample_matches_reference(ideal_model_has_gravity: bool):
    key = jax.random.PRNGKey(20 if ideal_model_has_gravity else 21)
    model = _build_ideal_model(ideal_model_has_gravity)
    key_params, key_sample, key_force = jax.random.split(key, 3)
    rollout_params = _sample_rollout_params(key_params, model, ideal_model_has_gravity=ideal_model_has_gravity)
    q, qd, tau = _sample_state_and_tau(key_sample, model, rollout_params, num_samples=1)
    force = _sample_external_force(key_force)
    body_id = int(model.nbody - 1)

    tau_ref = compute_gt_tau_cmd(
        model,
        rollout_params,
        q,
        qd,
        tau,
        external_force_ee=force,
        external_force_body_id=body_id,
        method="reference",
    )
    tau_fast = compute_gt_tau_cmd(
        model,
        rollout_params,
        q,
        qd,
        tau,
        external_force_ee=force,
        external_force_body_id=body_id,
        method="shared_linear",
    )

    assert tau_fast.shape == tau.shape
    np.testing.assert_allclose(np.asarray(tau_fast), np.asarray(tau_ref), rtol=3e-4, atol=3e-4)


@pytest.mark.skipif(MJX_IMPORT_ERROR is not None, reason=f"MJX unavailable: {MJX_IMPORT_ERROR}")
@pytest.mark.parametrize("ideal_model_has_gravity", [True, False])
def test_compute_gt_tau_cmd_forceaware_multi_sample_matches_reference(ideal_model_has_gravity: bool):
    key = jax.random.PRNGKey(30 if ideal_model_has_gravity else 31)
    model = _build_ideal_model(ideal_model_has_gravity)
    key_params, key_sample, key_force = jax.random.split(key, 3)
    rollout_params = _sample_rollout_params(key_params, model, ideal_model_has_gravity=ideal_model_has_gravity)
    q, qd, tau = _sample_state_and_tau(key_sample, model, rollout_params, num_samples=8)
    force = _sample_external_force(key_force)
    body_id = int(model.nbody - 1)

    tau_ref = compute_gt_tau_cmd(
        model,
        rollout_params,
        q,
        qd,
        tau,
        external_force_ee=force,
        external_force_body_id=body_id,
        method="reference",
    )
    tau_fast = compute_gt_tau_cmd(
        model,
        rollout_params,
        q,
        qd,
        tau,
        external_force_ee=force,
        external_force_body_id=body_id,
        method="shared_linear",
    )

    assert tau_fast.shape == tau.shape
    np.testing.assert_allclose(np.asarray(tau_fast), np.asarray(tau_ref), rtol=3e-4, atol=3e-4)


@pytest.mark.skipif(MJX_IMPORT_ERROR is not None, reason=f"MJX unavailable: {MJX_IMPORT_ERROR}")
@pytest.mark.parametrize("ideal_model_has_gravity", [True, False])
def test_compute_gt_tau_cmd_wrenchaware_single_sample_matches_reference(ideal_model_has_gravity: bool):
    key = jax.random.PRNGKey(35 if ideal_model_has_gravity else 36)
    model = _build_ideal_model(ideal_model_has_gravity)
    key_params, key_sample, key_force = jax.random.split(key, 3)
    rollout_params = _sample_rollout_params(key_params, model, ideal_model_has_gravity=ideal_model_has_gravity)
    q, qd, tau = _sample_state_and_tau(key_sample, model, rollout_params, num_samples=1)
    wrench = _sample_external_wrench(key_force)
    body_id = int(model.nbody - 1)

    tau_ref = compute_gt_tau_cmd(
        model,
        rollout_params,
        q,
        qd,
        tau,
        external_force_ee=wrench,
        external_force_body_id=body_id,
        method="reference",
    )
    tau_fast = compute_gt_tau_cmd(
        model,
        rollout_params,
        q,
        qd,
        tau,
        external_force_ee=wrench,
        external_force_body_id=body_id,
        method="shared_linear",
    )

    assert tau_fast.shape == tau.shape
    np.testing.assert_allclose(np.asarray(tau_fast), np.asarray(tau_ref), rtol=2e-4, atol=2e-4)


@pytest.mark.skipif(MJX_IMPORT_ERROR is not None, reason=f"MJX unavailable: {MJX_IMPORT_ERROR}")
@pytest.mark.parametrize("ideal_model_has_gravity", [True, False])
def test_compute_gt_tau_cmd_forceaware_batched_multi_sample_matches_reference(ideal_model_has_gravity: bool):
    model = _build_ideal_model(ideal_model_has_gravity)
    keys = jax.random.split(jax.random.PRNGKey(40 if ideal_model_has_gravity else 41), 5)
    rollout_params_a = _sample_rollout_params(keys[0], model, ideal_model_has_gravity=ideal_model_has_gravity)
    rollout_params_b = _sample_rollout_params(keys[1], model, ideal_model_has_gravity=ideal_model_has_gravity)
    q_a, qd_a, tau_a = _sample_state_and_tau(keys[2], model, rollout_params_a, num_samples=6)
    q_b, qd_b, tau_b = _sample_state_and_tau(keys[3], model, rollout_params_b, num_samples=6)
    force_batch = _sample_external_force(keys[4], batch_size=2)
    body_id = int(model.nbody - 1)

    rollout_params_batch = _stack_rollout_params(rollout_params_a, rollout_params_b)
    q_batch = jnp.stack([q_a, q_b], axis=0)
    qd_batch = jnp.stack([qd_a, qd_b], axis=0)
    tau_batch = jnp.stack([tau_a, tau_b], axis=0)

    tau_ref = compute_gt_tau_cmd(
        model,
        rollout_params_batch,
        q_batch,
        qd_batch,
        tau_batch,
        external_force_ee=force_batch,
        external_force_body_id=body_id,
        method="reference",
    )
    tau_fast = compute_gt_tau_cmd(
        model,
        rollout_params_batch,
        q_batch,
        qd_batch,
        tau_batch,
        external_force_ee=force_batch,
        external_force_body_id=body_id,
        method="shared_linear",
    )

    assert tau_fast.shape == tau_batch.shape
    np.testing.assert_allclose(np.asarray(tau_fast), np.asarray(tau_ref), rtol=4e-4, atol=4e-4)


@pytest.mark.skipif(MJX_IMPORT_ERROR is not None, reason=f"MJX unavailable: {MJX_IMPORT_ERROR}")
@pytest.mark.parametrize("ideal_model_has_gravity", [True, False])
def test_compute_gt_tau_cmd_forceaware_batched_body_ids_match_reference(ideal_model_has_gravity: bool):
    model = _build_ideal_model(ideal_model_has_gravity)
    keys = jax.random.split(jax.random.PRNGKey(44 if ideal_model_has_gravity else 45), 5)
    rollout_params_a = _sample_rollout_params(keys[0], model, ideal_model_has_gravity=ideal_model_has_gravity)
    rollout_params_b = _sample_rollout_params(keys[1], model, ideal_model_has_gravity=ideal_model_has_gravity)
    q_a, qd_a, tau_a = _sample_state_and_tau(keys[2], model, rollout_params_a, num_samples=5)
    q_b, qd_b, tau_b = _sample_state_and_tau(keys[3], model, rollout_params_b, num_samples=5)
    force_batch = _sample_external_force(keys[4], batch_size=2)
    body_ids = jnp.asarray(
        [int(model.nbody - 1), int(max(0, model.nbody - 2))],
        dtype=jnp.int32,
    )

    rollout_params_batch = _stack_rollout_params(rollout_params_a, rollout_params_b)
    q_batch = jnp.stack([q_a, q_b], axis=0)
    qd_batch = jnp.stack([qd_a, qd_b], axis=0)
    tau_batch = jnp.stack([tau_a, tau_b], axis=0)

    tau_ref = compute_gt_tau_cmd(
        model,
        rollout_params_batch,
        q_batch,
        qd_batch,
        tau_batch,
        external_force_ee=force_batch,
        external_force_body_id=body_ids,
        method="reference",
    )
    tau_fast = compute_gt_tau_cmd(
        model,
        rollout_params_batch,
        q_batch,
        qd_batch,
        tau_batch,
        external_force_ee=force_batch,
        external_force_body_id=body_ids,
        method="shared_linear",
    )

    assert tau_fast.shape == tau_batch.shape
    np.testing.assert_allclose(np.asarray(tau_fast), np.asarray(tau_ref), rtol=4e-4, atol=4e-4)


@pytest.mark.skipif(MJX_IMPORT_ERROR is not None, reason=f"MJX unavailable: {MJX_IMPORT_ERROR}")
def test_compute_gt_tau_cmd_preserves_tau_ref_shape():
    model = _build_ideal_model(True)
    keys = jax.random.split(jax.random.PRNGKey(200), 4)
    rollout_params_a = _sample_rollout_params(keys[0], model, ideal_model_has_gravity=True)
    rollout_params_b = _sample_rollout_params(keys[1], model, ideal_model_has_gravity=True)
    q_a, qd_a, tau_single = _sample_state_and_tau(keys[2], model, rollout_params_a, num_samples=1)
    q_b, qd_b, tau_multi = _sample_state_and_tau(keys[3], model, rollout_params_b, num_samples=4)

    tau_batch_single = jnp.stack([tau_single, tau_single], axis=0)
    tau_batch_multi = jnp.stack([tau_multi, tau_multi], axis=0)
    q_batch = jnp.stack([q_a, q_a], axis=0)
    qd_batch = jnp.stack([qd_a, qd_a], axis=0)
    rollout_params_batch = _stack_rollout_params(rollout_params_a, rollout_params_a)

    assert compute_gt_tau_cmd(model, rollout_params_a, q_a, qd_a, tau_single).shape == tau_single.shape
    assert compute_gt_tau_cmd(model, rollout_params_b, q_b, qd_b, tau_multi).shape == tau_multi.shape
    assert compute_gt_tau_cmd(model, rollout_params_batch, q_batch, qd_batch, tau_batch_single).shape == tau_batch_single.shape
    assert compute_gt_tau_cmd(model, rollout_params_batch, q_batch, qd_batch, tau_batch_multi).shape == tau_batch_multi.shape


@pytest.mark.skipif(MJX_IMPORT_ERROR is not None, reason=f"MJX unavailable: {MJX_IMPORT_ERROR}")
def test_compute_gt_tau_cmd_zero_force_matches_default():
    model = _build_ideal_model(True)
    key = jax.random.PRNGKey(250)
    key_params, key_sample = jax.random.split(key)
    rollout_params = _sample_rollout_params(key_params, model, ideal_model_has_gravity=True)
    q, qd, tau = _sample_state_and_tau(key_sample, model, rollout_params, num_samples=4)
    body_id = int(model.nbody - 1)
    zero_force = jnp.zeros((3,), dtype=jnp.float32)

    tau_default = compute_gt_tau_cmd(model, rollout_params, q, qd, tau, method="shared_linear")
    tau_force = compute_gt_tau_cmd(
        model,
        rollout_params,
        q,
        qd,
        tau,
        external_force_ee=zero_force,
        external_force_body_id=body_id,
        method="shared_linear",
    )

    np.testing.assert_allclose(np.asarray(tau_force), np.asarray(tau_default), rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(MJX_IMPORT_ERROR is not None, reason=f"MJX unavailable: {MJX_IMPORT_ERROR}")
def test_compute_gt_tau_cmd_rejects_invalid_rank_combinations():
    model = _build_ideal_model(True)
    key = jax.random.PRNGKey(300)
    key_params, key_sample = jax.random.split(key)
    rollout_params = _sample_rollout_params(key_params, model, ideal_model_has_gravity=True)
    q, qd, tau_single = _sample_state_and_tau(key_sample, model, rollout_params, num_samples=1)
    q_batch = jnp.stack([q, q], axis=0)
    qd_batch = jnp.stack([qd, qd], axis=0)

    with pytest.raises(ValueError, match="Expected tau_ref rank 2 or 3 for batched states"):
        compute_gt_tau_cmd(model, rollout_params, q_batch, qd_batch, tau_single)

    with pytest.raises(ValueError, match="Expected tau_ref rank 1 or 2 for a single state"):
        compute_gt_tau_cmd(model, rollout_params, q, qd, tau_single[None, None, :])

    with pytest.raises(ValueError, match="Expected external_force_ee rank 1 for a single state"):
        compute_gt_tau_cmd(model, rollout_params, q, qd, tau_single, external_force_ee=tau_single[None, :])

    with pytest.raises(ValueError, match="Expected external_force_ee rank 2 for batched states"):
        compute_gt_tau_cmd(model, rollout_params, q_batch, qd_batch, q_batch, external_force_ee=tau_single)

    with pytest.raises(ValueError, match="external_force_ee must have trailing size 3 or 6"):
        compute_gt_tau_cmd(
            model,
            rollout_params,
            q,
            qd,
            tau_single,
            external_force_ee=jnp.zeros((4,), dtype=jnp.float32),
        )

    with pytest.raises(ValueError, match="Expected scalar or shape \\(2,\\) external_force_body_id"):
        compute_gt_tau_cmd(
            model,
            _stack_rollout_params(rollout_params, rollout_params),
            q_batch,
            qd_batch,
            q_batch,
            external_force_ee=jnp.zeros((2, 3), dtype=jnp.float32),
            external_force_body_id=jnp.zeros((2, 1), dtype=jnp.int32),
        )
