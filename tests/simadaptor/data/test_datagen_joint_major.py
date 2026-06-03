import sys

import jax.numpy as jnp
import numpy as np

from tests.repo_paths import REPO_ROOT as ROOT

SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from simadaptor.data import datagen  # noqa: E402


def _point_mass_inertia_np(mass: float, offset: np.ndarray) -> np.ndarray:
    ox, oy, oz = np.asarray(offset, dtype=np.float32).reshape(3)
    return float(mass) * np.asarray(
        [oy * oy + oz * oz, ox * ox + oz * oz, ox * ox + oy * oy],
        dtype=np.float32,
    )


def test_joint_major_body_payload_scaling_is_physically_coherent():
    model_body_mass = jnp.asarray([0.0, 2.0, 4.0, 1.0], dtype=jnp.float32)
    model_body_inertia = jnp.asarray(
        [
            [1.0, 1.0, 1.0],
            [0.20, 0.30, 0.40],
            [0.50, 0.60, 0.70],
            [0.08, 0.09, 0.10],
        ],
        dtype=jnp.float32,
    )
    model_body_ipos = jnp.asarray(
        [
            [0.00, 0.00, 0.00],
            [0.10, 0.20, 0.30],
            [0.40, 0.50, 0.60],
            [0.70, 0.80, 0.90],
        ],
        dtype=jnp.float32,
    )
    raw_body_mass_scale = jnp.asarray([1.10, 0.90, 1.10, 0.95], dtype=jnp.float32)
    raw_ipos_noise = jnp.asarray(
        [
            [0.010, -0.010, 0.005],
            [-0.010, 0.008, -0.006],
            [0.004, -0.002, 0.010],
            [0.003, 0.005, -0.009],
        ],
        dtype=jnp.float32,
    )
    raw_payload_mass_delta = jnp.asarray(1.5, dtype=jnp.float32)
    raw_payload_offset = jnp.asarray([0.075, -0.050, 0.025], dtype=jnp.float32)
    ee_body_id = 2
    global_scale = 0.02
    ee_scale = 0.02

    body_mass, body_inertia, body_ipos = datagen._compose_body_payload_randomization(
        model_body_mass=model_body_mass,
        model_body_inertia=model_body_inertia,
        model_body_ipos=model_body_ipos,
        raw_body_mass_scale=raw_body_mass_scale,
        raw_ipos_noise=raw_ipos_noise,
        ee_body_id=jnp.asarray(ee_body_id, dtype=jnp.int32),
        raw_ee_payload_mass_delta=raw_payload_mass_delta,
        raw_ee_payload_offset_local=raw_payload_offset,
        is_joint_model_major=jnp.asarray(True),
        joint_model_major_global_scale=global_scale,
        joint_model_major_ee_scale=ee_scale,
    )

    body_mass_np = np.asarray(body_mass)
    body_inertia_np = np.asarray(body_inertia)
    body_ipos_np = np.asarray(body_ipos)
    model_mass_np = np.asarray(model_body_mass)
    model_inertia_np = np.asarray(model_body_inertia)
    model_ipos_np = np.asarray(model_body_ipos)
    raw_mass_scale_np = np.asarray(raw_body_mass_scale)
    raw_ipos_noise_np = np.asarray(raw_ipos_noise)

    effective_mass_scale = 1.0 + global_scale * (raw_mass_scale_np - 1.0)
    assert np.max(np.abs(effective_mass_scale - 1.0)) <= 0.002 + 1e-7

    non_ee_positive = (model_mass_np > 0.0) & (np.arange(model_mass_np.shape[0]) != ee_body_id)
    np.testing.assert_allclose(
        body_mass_np[non_ee_positive],
        model_mass_np[non_ee_positive] * effective_mass_scale[non_ee_positive],
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        body_inertia_np[non_ee_positive],
        model_inertia_np[non_ee_positive] * effective_mass_scale[non_ee_positive, None],
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        body_ipos_np[non_ee_positive],
        model_ipos_np[non_ee_positive] + raw_ipos_noise_np[non_ee_positive] * global_scale,
        rtol=1e-6,
        atol=1e-7,
    )
    assert np.max(np.abs(body_ipos_np[non_ee_positive] - model_ipos_np[non_ee_positive])) <= 0.0002 + 1e-7

    ee_base_mass = model_mass_np[ee_body_id] * effective_mass_scale[ee_body_id]
    ee_base_inertia = model_inertia_np[ee_body_id] * effective_mass_scale[ee_body_id]
    payload_mass = float(raw_payload_mass_delta) * ee_scale
    payload_offset = np.asarray(raw_payload_offset, dtype=np.float32) * ee_scale
    np.testing.assert_allclose(payload_mass, 0.03, rtol=1e-6, atol=1e-9)
    assert np.max(np.abs(payload_offset)) <= 0.0015 + 1e-9

    ee_total_mass = max(ee_base_mass + payload_mass, 1e-2)
    ee_com_shift = payload_offset * (payload_mass / ee_total_mass)
    payload_rel_to_total_com = payload_offset - ee_com_shift
    expected_ee_inertia = (
        ee_base_inertia
        + _point_mass_inertia_np(ee_base_mass, ee_com_shift)
        + _point_mass_inertia_np(payload_mass, payload_rel_to_total_com)
    )

    np.testing.assert_allclose(body_mass_np[ee_body_id], ee_total_mass, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(
        body_ipos_np[ee_body_id],
        model_ipos_np[ee_body_id] + ee_com_shift,
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        body_inertia_np[ee_body_id],
        expected_ee_inertia,
        rtol=1e-6,
        atol=1e-7,
    )
    assert np.all(body_mass_np[model_mass_np > 0.0] > 0.0)
    assert np.all(body_inertia_np[model_mass_np > 0.0] > 0.0)
