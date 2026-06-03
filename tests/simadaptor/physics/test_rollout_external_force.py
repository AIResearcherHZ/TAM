from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.repo_paths import REPO_ROOT as ROOT
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    import mujoco
    from mujoco import mjx as _mjx  # noqa: F401

    import simadaptor.core.structs as structs
    import simadaptor.physics.dynamics as dynamics
    import simadaptor.physics.rollout as rollout

    MJX_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment dependent
    mujoco = None
    structs = None
    dynamics = None
    rollout = None
    MJX_IMPORT_ERROR = exc


XML_PATH = ROOT / "assets" / "franka_panda" / "panda_pandagripper.xml"


def _make_rollout_params(model, *, batch_size: int = 1):
    return structs.RolloutParams(
        kp=jnp.zeros((batch_size, int(model.nu)), dtype=jnp.float32),
        kd=jnp.zeros((batch_size, int(model.nu)), dtype=jnp.float32),
        dof_armature=jnp.broadcast_to(jnp.asarray(model.dof_armature), (batch_size, int(model.nv))),
        dof_damping=jnp.broadcast_to(jnp.asarray(model.dof_damping), (batch_size, int(model.nv))),
        body_mass=jnp.broadcast_to(jnp.asarray(model.body_mass), (batch_size, int(model.nbody))),
        body_inertia=jnp.broadcast_to(jnp.asarray(model.body_inertia), (batch_size, int(model.nbody), 3)),
        body_ipos=jnp.broadcast_to(jnp.asarray(model.body_ipos), (batch_size, int(model.nbody), 3)),
    )


def _zero_actuator_fn(q_window, qd_window, q_ref_t, qd_ref_t, rng_action, actuator_carry, u_ref=None):
    del qd_window, q_ref_t, qd_ref_t, rng_action, u_ref
    batch_size = int(q_window.shape[0])
    dof = int(q_window.shape[-1])
    ctrl = jnp.zeros((batch_size, dof), dtype=q_window.dtype)
    delta = jnp.zeros_like(ctrl)
    carry_out = actuator_carry if actuator_carry is not None else jnp.zeros_like(ctrl)
    return ctrl, delta, carry_out


@pytest.mark.skipif(MJX_IMPORT_ERROR is not None, reason=f"MJX unavailable: {MJX_IMPORT_ERROR}")
def test_rollout_generation_external_force_stays_3d_without_position_box():
    model = dynamics.load_mjx_model_from_path(str(XML_PATH), remove_constraints=True)
    cpu_model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    body_id = int(mujoco.mj_name2id(cpu_model, mujoco.mjtObj.mjOBJ_BODY, "hand"))

    out = rollout.rollout_generation(
        jax.random.PRNGKey(0),
        _zero_actuator_fn,
        model,
        np.arange(int(model.nu), dtype=np.int32),
        _make_rollout_params(model),
        num_waypoints=2,
        duration=0.01,
        pause_prob=0.0,
        external_force_body_id=body_id,
        external_force_num_impulses=1,
        external_force_magnitude_min_n=20.0,
        external_force_magnitude_max_n=20.0,
        external_force_duration_min_s=0.01,
        external_force_duration_max_s=0.01,
    )

    assert out["external_force_ee"].shape[0] == 1
    assert out["external_force_ee"].shape[-1] == 3


@pytest.mark.skipif(MJX_IMPORT_ERROR is not None, reason=f"MJX unavailable: {MJX_IMPORT_ERROR}")
def test_rollout_generation_external_force_becomes_6d_with_position_box():
    model = dynamics.load_mjx_model_from_path(str(XML_PATH), remove_constraints=True)
    cpu_model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    body_id = int(mujoco.mj_name2id(cpu_model, mujoco.mjtObj.mjOBJ_BODY, "hand"))

    out = rollout.rollout_generation(
        jax.random.PRNGKey(1),
        _zero_actuator_fn,
        model,
        np.arange(int(model.nu), dtype=np.int32),
        _make_rollout_params(model),
        num_waypoints=2,
        duration=0.01,
        pause_prob=0.0,
        external_force_body_id=body_id,
        external_force_num_impulses=1,
        external_force_magnitude_min_n=20.0,
        external_force_magnitude_max_n=20.0,
        external_force_duration_min_s=0.01,
        external_force_duration_max_s=0.01,
        external_force_position_min_local_m=jnp.asarray([0.05, 0.04, 0.03], dtype=jnp.float32),
        external_force_position_max_local_m=jnp.asarray([0.05, 0.04, 0.03], dtype=jnp.float32),
    )

    force_wrench = np.asarray(out["external_force_ee"])
    assert force_wrench.shape[0] == 1
    assert force_wrench.shape[-1] == 6
    assert np.max(np.abs(force_wrench[..., 3:])) > 0.0
