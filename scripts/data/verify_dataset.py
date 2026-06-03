import glob
import os
import numpy as np
import mujoco
from mujoco import mjx
import jax.numpy as jnp
import argparse
from typing import Any, Dict, Optional, List
import jax
from tqdm import tqdm
import zarr

from simadaptor.data.dataloader import PandaRolloutShardDataset
import simadaptor.core.structs as structs
import simadaptor.physics.actuator as actuator_util
import simadaptor.physics.dynamics as dynamics
from pathlib import Path
import json

def _to_jnp(x: Any) -> jnp.ndarray:
    return jnp.asarray(np.asarray(x))


def _open_zarr_group(path: str, mode: str = "r"):
    if path.endswith(".zip"):
        store = zarr.storage.ZipStore(path, mode=mode)
        group = zarr.open_group(store=store, mode=mode)
        return group, store.close
    return zarr.open_group(path, mode=mode), (lambda: None)


def _first_split_path(base_path: str, split: str) -> Optional[str]:
    zips = glob.glob(os.path.join(base_path, split, "*.zarr.zip"))
    dirs = glob.glob(os.path.join(base_path, split, "*.zarr"))
    paths = sorted(zips + dirs)
    return paths[0] if paths else None


def _is_robot_dataset_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "data_generation_config.json").exists()
        and (path / "perturbed").exists()
    )


def _discover_robot_dataset_dirs(dataset_root: Path) -> List[Path]:
    if _is_robot_dataset_dir(dataset_root):
        return [dataset_root]
    robot_dirs = []
    if not dataset_root.exists():
        return robot_dirs
    for child in sorted(dataset_root.iterdir()):
        if _is_robot_dataset_dir(child):
            robot_dirs.append(child)
    return robot_dirs


def _resolve_dataset_dir(base_path: str, robot_key: Optional[str]) -> Path:
    base = Path(base_path).expanduser()
    robot_dirs = _discover_robot_dataset_dirs(base)
    if not robot_dirs:
        raise FileNotFoundError(
            f"No robot dataset directory found at {base}. "
            "Expected either a robot dataset dir with 'perturbed/' + data_generation_config.json "
            "or a root containing such subdirectories."
        )
    if len(robot_dirs) == 1 and robot_key is None:
        return robot_dirs[0]

    if robot_key is None:
        keys = ", ".join(d.name for d in robot_dirs)
        raise ValueError(
            f"Multiple robot datasets found under {base}. "
            f"Please specify --robot-key. Available: {keys}"
        )

    exact = base / robot_key
    if _is_robot_dataset_dir(exact):
        return exact

    for d in robot_dirs:
        manifest_path = d / "robot_model" / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            if str(manifest.get("robot_key", "")) == robot_key:
                return d
        except Exception:
            pass

    keys = ", ".join(d.name for d in robot_dirs)
    raise ValueError(f"Unknown --robot-key '{robot_key}'. Available: {keys}")


def _resolve_robot_xml(dataset_dir: Path, xml_override: Optional[str]) -> Path:
    if xml_override:
        p = Path(xml_override).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"--xml not found: {p}")
        return p

    robot_dir = dataset_dir / "robot_model"
    manifest_path = robot_dir / "manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            rel_xml = manifest.get("robot_xml")
            if rel_xml:
                cand = dataset_dir / rel_xml
                if cand.exists():
                    return cand
        except Exception:
            pass

    robot_xml = robot_dir / "robot.xml"
    if robot_xml.exists():
        return robot_xml

    cfg_path = dataset_dir / "data_generation_config.json"
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            ds_cfg = json.load(f)
        cfg_xml = ds_cfg.get("xml_path")
        if cfg_xml:
            p = Path(cfg_xml).expanduser()
            for cand in (p, Path.cwd() / p, dataset_dir / p):
                if cand.exists():
                    return cand

    raise FileNotFoundError(
        f"Could not resolve robot XML from dataset dir {dataset_dir}. "
        "Provide --xml explicitly."
    )


def _rollout_field_exists(path: str, field: str) -> bool:
    g, closer = _open_zarr_group(path, mode="r")
    try:
        return ("rollout" in g) and (field in g["rollout"])
    finally:
        closer()


def _read_root_attrs(path: str) -> Dict[str, Any]:
    g, closer = _open_zarr_group(path, mode="r")
    try:
        return dict(g.attrs)
    finally:
        closer()


def _resolve_external_force_body(
    xml_path: str,
    root_attrs: Dict[str, Any],
    ds_cfg: Dict[str, Any],
) -> tuple[int, str]:
    body_id_attr = root_attrs.get("external_force_body_id", None)
    if body_id_attr is not None:
        try:
            body_id = int(body_id_attr)
            if body_id >= 0:
                body_name = root_attrs.get("external_force_body_name", f"body_{body_id}")
                return body_id, str(body_name)
        except Exception:
            pass

    mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
    body_name = ds_cfg.get("external_force_body_name")

    if body_name:
        body_id = int(mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, str(body_name)))
        if body_id >= 0:
            return body_id, str(body_name)

    raise ValueError(
        f"Failed to resolve external-force body from attrs/cfg. "
        f"attrs_body_id={body_id_attr!r}, cfg_body_name={body_name!r}"
    )


def _resolve_ee_payload_body(
    xml_path: str,
    root_attrs: Dict[str, Any],
    ds_cfg: Dict[str, Any],
) -> tuple[int, str]:
    body_id_attr = root_attrs.get("ee_payload_body_id", ds_cfg.get("ee_payload_body_id"))
    if body_id_attr is not None:
        try:
            body_id = int(body_id_attr)
            if body_id >= 0:
                body_name = root_attrs.get(
                    "ee_payload_body_name",
                    ds_cfg.get("ee_payload_body_name", f"body_{body_id}"),
                )
                return body_id, str(body_name)
        except Exception:
            pass

    mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
    body_name = root_attrs.get("ee_payload_body_name", ds_cfg.get("ee_payload_body_name"))
    if body_name:
        body_id = int(mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, str(body_name)))
        if body_id >= 0:
            return body_id, str(body_name)

    if int(mj_model.nsite) > 0:
        body_id = int(np.asarray(mj_model.site_bodyid, dtype=np.int32)[-1])
    else:
        dof_bodyid = np.asarray(mj_model.dof_bodyid, dtype=np.int32)
        if dof_bodyid.size == 0:
            raise ValueError(
                f"Failed to resolve EE payload body from attrs/cfg/xml for xml_path={xml_path!r}."
            )
        ee_dof_id = max(min(int(mj_model.nu), int(mj_model.nv)) - 1, 0)
        body_id = int(dof_bodyid[min(ee_dof_id, int(dof_bodyid.size) - 1)])
    body_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
    return body_id, str(body_name)


def _maybe_vector3(values: Any) -> Optional[jnp.ndarray]:
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size != 3:
        raise ValueError(f"Expected length-3 vector, got {values!r}.")
    return jnp.asarray(arr, dtype=jnp.float32)


def _verify_ee_param_ranges(
    mjx_model: mjx.Model,
    params: Dict[str, Any],
    *,
    ee_body_id: int,
    ee_body_name: str,
    ee_payload_offset_min_local_m: jnp.ndarray,
    ee_payload_offset_max_local_m: jnp.ndarray,
) -> Dict[str, Any]:
    body_ipos = _to_jnp(params["body_ipos"])
    if body_ipos.ndim != 3:
        raise ValueError(f"Expected params['body_ipos'] rank 3, got {body_ipos.shape}")
    if ee_body_id >= int(body_ipos.shape[1]):
        raise ValueError(
            f"EE body id {ee_body_id} is out of range for params['body_ipos'] shape {body_ipos.shape}"
        )

    ee_abs_bound = jnp.maximum(
        jnp.abs(ee_payload_offset_min_local_m),
        jnp.abs(ee_payload_offset_max_local_m),
    )
    ee_nominal_ipos = jnp.asarray(mjx_model.body_ipos[ee_body_id], dtype=body_ipos.dtype)
    ee_delta = body_ipos[:, ee_body_id, :] - ee_nominal_ipos[None, :]
    tol = jnp.asarray(1.0e-6, dtype=body_ipos.dtype)
    violating = jnp.any(jnp.abs(ee_delta) > (ee_abs_bound + tol), axis=-1)
    if jnp.any(violating):
        violating_idx = np.asarray(jnp.flatnonzero(violating))[:16].astype(int).tolist()
        raise ValueError(
            f"EE body_ipos delta for body '{ee_body_name}' (id={ee_body_id}) exceeded the payload COM-offset bound "
            f"{np.asarray(ee_abs_bound).astype(float).tolist()}. "
            f"violating_indices={violating_idx}"
        )

    body_mass = _to_jnp(params["body_mass"])
    if body_mass.ndim != 2:
        raise ValueError(f"Expected params['body_mass'] rank 2, got {body_mass.shape}")
    if jnp.any(body_mass[:, ee_body_id] <= 0.0):
        raise ValueError(
            f"EE body '{ee_body_name}' (id={ee_body_id}) has non-positive mass in params['body_mass']."
        )

    body_inertia = _to_jnp(params["body_inertia"])
    if body_inertia.ndim != 3:
        raise ValueError(f"Expected params['body_inertia'] rank 3, got {body_inertia.shape}")
    if jnp.any(body_inertia[:, ee_body_id, :] <= 0.0):
        raise ValueError(
            f"EE body '{ee_body_name}' (id={ee_body_id}) has non-positive inertia in params['body_inertia']."
        )

    return {
        "ee_delta_min": np.asarray(jnp.min(ee_delta, axis=0), dtype=np.float32),
        "ee_delta_max": np.asarray(jnp.max(ee_delta, axis=0), dtype=np.float32),
        "ee_mass_min": float(jnp.min(body_mass[:, ee_body_id])),
        "ee_mass_max": float(jnp.max(body_mass[:, ee_body_id])),
        "ee_inertia_min": np.asarray(jnp.min(body_inertia[:, ee_body_id, :], axis=0), dtype=np.float32),
        "ee_inertia_max": np.asarray(jnp.max(body_inertia[:, ee_body_id, :], axis=0), dtype=np.float32),
    }


def verify_rollout(
    mjx_model: mjx.Model,
    rollout: Dict[str, Any],
    params: Dict[str, Any],
    external_force_body_id: int | jax.Array = -1,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Check that stepping the model with (q, qd, u) matches stored next state/qdd.
    Also simulate an open-loop rollout for a fixed horizon and compare against the logged trajectory."""
    q = _to_jnp(rollout["q"])
    qd = _to_jnp(rollout["qd"])
    tau_cmd_log = _to_jnp(rollout["u"])
    times = _to_jnp(rollout["times"]) if "times" in rollout else None
    qdd = _to_jnp(rollout["qdd"]) if "qdd" in rollout else None
    external_force_ee = _to_jnp(rollout["external_force_ee"]) if "external_force_ee" in rollout else None
    external_force_body_ids = rollout.get("external_force_body_id", None)
    if external_force_body_ids is None:
        external_force_body_ids = jnp.full((int(q.shape[0]),), int(external_force_body_id), dtype=jnp.int32)
    else:
        external_force_body_ids = _to_jnp(external_force_body_ids).astype(jnp.int32)
        if external_force_body_ids.ndim == 2 and external_force_body_ids.shape[-1] == 1:
            external_force_body_ids = external_force_body_ids[:, 0]
        if external_force_body_ids.ndim != 1 or int(external_force_body_ids.shape[0]) != int(q.shape[0]):
            raise ValueError(
                f"external_force_body_id must have shape ({int(q.shape[0])},), got {external_force_body_ids.shape}"
            )
    if jnp.any(external_force_body_ids >= int(mjx_model.nbody)):
        raise ValueError(
            f"external_force_body_id contains out-of-range values for nbody={mjx_model.nbody}: "
            f"{external_force_body_ids}"
        )

    steps = q.shape[1] - 1
    if max_steps is not None:
        steps = int(min(steps, max_steps))
        q = q[:, : steps + 1]
        qd = qd[:, : steps + 1]
        tau_cmd_log = tau_cmd_log[:, : steps + 1]
        if times is not None:
            times = times[:, : steps + 1]
        if qdd is not None:
            qdd = qdd[:, : steps + 1]
        if external_force_ee is not None:
            external_force_ee = external_force_ee[:, : steps + 1]

    if external_force_ee is not None and external_force_ee.shape[:2] != q.shape[:2]:
        raise ValueError(
            f"external_force_ee shape mismatch: expected batch/time {q.shape[:2]}, got {external_force_ee.shape[:2]}"
        )
    if external_force_ee is not None and external_force_ee.shape[-1] not in (3, 6):
        raise ValueError(
            f"external_force_ee must have trailing size 3 or 6, got {external_force_ee.shape}"
        )
    per_step_max_q = jnp.zeros((steps,))
    per_step_max_qd = jnp.zeros((steps,))
    per_step_max_qdd = jnp.zeros((steps,))
    per_step_max_time = jnp.zeros((steps,)) if times is not None else None
    rollout_max_q = jnp.zeros((steps,))
    rollout_max_qd = jnp.zeros((steps,))
    rollout_max_qdd = jnp.zeros((steps,))
    rollout_max_time = jnp.zeros((steps,)) if times is not None else None
    tau_cmd_recovery_max = jnp.zeros((steps,))
    tau_cmd_base_ideal_forceaware_max = jnp.zeros((steps,))
    

    def _set_body_wrench(xfrc: jax.Array, force_t: jax.Array, body_id_t: jax.Array) -> jax.Array:
        force_t = jnp.asarray(force_t, dtype=xfrc.dtype)
        body_id_t = jnp.asarray(body_id_t, dtype=jnp.int32)
        if force_t.shape[-1] not in (3, 6):
            raise ValueError(f"external_force_ee must have trailing size 3 or 6, got {force_t.shape}")

        def _apply(xfrc_in: jax.Array) -> jax.Array:
            if force_t.shape[-1] == 3:
                return xfrc_in.at[body_id_t, :3].set(force_t)
            return xfrc_in.at[body_id_t, :].set(force_t)

        return jax.lax.cond(
            (body_id_t >= 0) & (body_id_t < int(xfrc.shape[-2])),
            _apply,
            lambda xfrc_in: xfrc_in,
            xfrc,
        )

    def one_step(q_t, qd_t, tau_cmd_t, t_t, force_t, body_id_t, rollout_params:structs.RolloutParams):
        model_b = rollout_params.set_mjx_model(mjx_model)
        data_template = mjx.make_data(model_b)
        tau_eff_t = actuator_util.actuator_model(tau_cmd_t, q_t, qd_t, rollout_params.actuator_params)
        xfrc = jnp.zeros_like(data_template.xfrc_applied)
        xfrc = _set_body_wrench(xfrc, force_t, body_id_t)
        data0 = data_template.replace(qpos=q_t, qvel=qd_t, ctrl=tau_eff_t, time=t_t, xfrc_applied=xfrc)
        data1 = mjx.step(model_b, data0)
        return data1.qpos, data1.qvel, data1.qacc, data1.time
    one_step_jit = jax.jit(jax.vmap(one_step, in_axes=(0, 0, 0, 0, 0, None, None)))

    def rollout_once(q0, qd0, tau_cmd_seq, force_seq, body_id_t, rollout_params:structs.RolloutParams):
        """Simulate a full rollout using the logged controls."""
        model_b = rollout_params.set_mjx_model(mjx_model)
        data = mjx.make_data(model_b).replace(qpos=q0, qvel=qd0, ctrl=jnp.zeros_like(tau_cmd_seq[0]))

        def _step(carry, inputs):
            tau_cmd_t, force_t = inputs
            data_t = carry
            tau_eff_t = actuator_util.actuator_model(tau_cmd_t, data_t.qpos, data_t.qvel, rollout_params.actuator_params)
            xfrc = jnp.zeros_like(data_t.xfrc_applied)
            xfrc = _set_body_wrench(xfrc, force_t, body_id_t)
            data_t = data_t.replace(ctrl=tau_eff_t, xfrc_applied=xfrc)
            data_t = mjx.step(model_b, data_t)
            return data_t, (data_t.qpos, data_t.qvel, data_t.qacc, data_t.time)

        data, (q_roll, qd_roll, qdd_roll, time_roll) = jax.lax.scan(
            _step,
            data,
            (tau_cmd_seq[:steps], force_seq[:steps]),
        )
        return q_roll, qd_roll, qdd_roll, time_roll
    rollout_once_jit = jax.jit(rollout_once)

    def calc_tau_cmd_recovered(q_t, qd_t, qacc_t, rollout_params:structs.RolloutParams):
        model_b = rollout_params.set_mjx_model(mjx_model)
        tau_eff_cmd = dynamics.mjx_inverse_dynamics_rne(model_b, q_t, qd_t, qacc_t)
        return actuator_util.inv_actuator_model(tau_eff_cmd, q_t, qd_t, rollout_params.actuator_params)
    calc_tau_cmd_recovered_jit = jax.jit(jax.vmap(calc_tau_cmd_recovered, in_axes=(0, 0, 0, None)))

    def calc_tau_cmd_base_ideal_forceaware(q_t, qd_t, qacc_t, force_t, body_id_t):
        tau_eff_id_ideal = dynamics.mjx_inverse_dynamics_rne(mjx_model, q_t, qd_t, qacc_t)
        tau_eff_ext_ideal = dynamics.compute_external_tau_equivalent(
            mjx_model,
            q_t,
            qd_t,
            force_t,
            external_force_body_id=body_id_t,
        )
        tau_cmd_base = tau_eff_id_ideal - tau_eff_ext_ideal
        return tau_cmd_base
    calc_tau_cmd_base_ideal_forceaware_jit = jax.jit(
        jax.vmap(calc_tau_cmd_base_ideal_forceaware, in_axes=(0, 0, 0, 0, None))
    )

    params = structs.RolloutParams(**params)
    # Iterate over trajectories inside the shard.
    for b in tqdm(range(q.shape[0])):
        rp = params[b]
        body_id_b = external_force_body_ids[b]

        time_in = times[b, :steps] if times is not None else jnp.zeros((steps,), dtype=q.dtype)
        force_in = (
            external_force_ee[b, :steps]
            if external_force_ee is not None
            else jnp.zeros((steps, 3), dtype=q.dtype)
        )
        q_pred, qd_pred, qdd_pred, time_pred = one_step_jit(
            q[b, :steps], qd[b, :steps], tau_cmd_log[b, :steps], time_in, force_in, body_id_b, rp
        )

        per_step_max_q = jnp.maximum(per_step_max_q, jnp.max(jnp.abs(q_pred - q[b, 1 : steps + 1]), axis=-1))
        per_step_max_qd = jnp.maximum(per_step_max_qd, jnp.max(jnp.abs(qd_pred - qd[b, 1 : steps + 1]), axis=-1))

        if qdd is not None:
            qdd_err = jnp.max(jnp.abs(qdd_pred - qdd[b, :steps]), axis=-1)
            per_step_max_qdd = jnp.maximum(per_step_max_qdd, qdd_err)

        if times is not None:
            time_err = jnp.abs(time_pred - times[b, 1 : steps + 1])
            per_step_max_time = jnp.maximum(per_step_max_time, time_err)
        
        # Full rollout comparison
        q_roll, qd_roll, qdd_roll, time_roll = rollout_once_jit(
            q[b, 0], qd[b, 0], tau_cmd_log[b, :steps], force_in, body_id_b, rp
        )
        rollout_max_q = jnp.maximum(rollout_max_q, jnp.max(jnp.abs(q_roll - q[b, 1 : steps + 1]), axis=-1))
        rollout_max_qd = jnp.maximum(rollout_max_qd, jnp.max(jnp.abs(qd_roll - qd[b, 1 : steps + 1]), axis=-1))
        if times is not None:
            rollout_time_err = jnp.abs(time_roll - times[b, 1 : steps + 1])
            rollout_max_time = jnp.maximum(rollout_max_time, rollout_time_err)

        # Command torque recovery under perturbed dynamics.
        # Prefer the model's own per-step qacc from stepping with u (avoids finite-diff noise).
        qacc_for_gt = qdd_pred
        tau_cmd_recovered = calc_tau_cmd_recovered_jit(q[b, :steps], qd[b, :steps], qacc_for_gt, rp)
        tau_cmd_err = jnp.max(jnp.abs(tau_cmd_recovered - tau_cmd_log[b, :steps]), axis=-1)
        tau_cmd_recovery_max = jnp.maximum(tau_cmd_recovery_max, tau_cmd_err)

        # Optional diagnostic: force-aware ideal reference against logged commanded torque.
        tau_cmd_base_ideal_forceaware = calc_tau_cmd_base_ideal_forceaware_jit(
            q[b, :steps], qd[b, :steps], qacc_for_gt, force_in, body_id_b
        )
        tau_cmd_base_ideal_err = jnp.max(
            jnp.abs(tau_cmd_base_ideal_forceaware - tau_cmd_log[b, :steps]), axis=-1
        )
        tau_cmd_base_ideal_forceaware_max = jnp.maximum(
            tau_cmd_base_ideal_forceaware_max,
            tau_cmd_base_ideal_err,
        )
        
        rollout_qdd_err = jnp.max(jnp.abs(qdd_roll - qacc_for_gt), axis=-1)
        rollout_max_qdd = jnp.maximum(rollout_max_qdd, rollout_qdd_err)

        print(
            f"q error {per_step_max_q.max():.3e}, qd error {per_step_max_qd.max():.3e}, "
            f"tau_cmd error {tau_cmd_recovery_max.max():.3e} \r",
            end="",
        )
    print("\n")
    return {
        "per_step_max_q": per_step_max_q,
        "per_step_max_qd": per_step_max_qd,
        "per_step_max_qdd": per_step_max_qdd,
        "per_step_max_time": per_step_max_time,
        "rollout_max_q": rollout_max_q,
        "rollout_max_qd": rollout_max_qd,
        "rollout_max_qdd": rollout_max_qdd,
        "rollout_max_time": rollout_max_time,
        "tau_recovery_max": tau_cmd_recovery_max,
        "tau_cmd_base_ideal_forceaware_max": tau_cmd_base_ideal_forceaware_max,
        "steps_checked": steps,
    }


def main():
    parser = argparse.ArgumentParser(description="Verify dataset alignment by re-simulating one-step transitions.")
    parser.add_argument(
        "--base-path",
        type=str,
        default="dataset",
        help="Dataset root or robot dataset directory.",
    )
    parser.add_argument(
        "--robot-key",
        type=str,
        default=None,
        help="Robot key/subdirectory name when --base-path is a multi-robot root.",
    )
    parser.add_argument("--split", type=str, default="perturbed", choices=["original", "perturbed"])
    parser.add_argument(
        "--xml",
        type=str,
        default=None,
        help="Optional explicit robot XML path. If omitted, resolve from dataset metadata.",
    )
    parser.add_argument("--num-shards", type=int, default=32, help="Number of shards to check.")
    parser.add_argument("--start-index", type=int, default=0, help="Index of the first shard to check.")
    parser.add_argument("--max-steps", type=int, default=100, help="Max time steps per trajectory to verify.")
    args = parser.parse_args()

    dataset_dir = _resolve_dataset_dir(args.base_path, args.robot_key)
    print(f"Resolved dataset directory: {dataset_dir}")

    fields = ["q", "qd", "u", "times"]
    if args.split == "original":
        fields.append("qdd")
    first_path = _first_split_path(str(dataset_dir), args.split)
    if first_path is None:
        raise FileNotFoundError(f"No shards found under {dataset_dir}/{args.split}")
    has_external_force = _rollout_field_exists(first_path, "external_force_ee")
    if has_external_force:
        fields.append("external_force_ee")

    ds = PandaRolloutShardDataset(base_path=str(dataset_dir), split=args.split, fields=fields)
    print(f"Loaded dataset split='{args.split}' with {len(ds)} shards.")
    print(f"External-force field present: {has_external_force}")

    xml_path = _resolve_robot_xml(dataset_dir, args.xml)
    print(f"Using robot XML: {xml_path}")
    mjx_model = dynamics.load_mjx_model_from_path(str(xml_path), True)

    # Public TAM uses the real-gravity ideal-model contract.
    cfg_path = dataset_dir / "data_generation_config.json"
    with open(cfg_path, "r") as f:
        ds_cfg = json.load(f)
    root_attrs = _read_root_attrs(first_path)
    if not bool(ds_cfg.get("ideal_model_has_gravity", True)):
        raise ValueError(
            f"Dataset config {cfg_path} uses an unsupported floating ideal-model contract. "
            "Regenerate it for public TAM."
        )
    print(f"Dataset config: public TAM real-gravity ideal model (file: {cfg_path})")
    ee_payload_offset_min = _maybe_vector3(
        root_attrs.get(
            "ee_payload_com_offset_min_local_m",
            ds_cfg.get("ee_payload_com_offset_min_local_m"),
        )
    )
    ee_payload_offset_max = _maybe_vector3(
        root_attrs.get(
            "ee_payload_com_offset_max_local_m",
            ds_cfg.get("ee_payload_com_offset_max_local_m"),
        )
    )
    ee_body_id = None
    ee_body_name = None
    if ee_payload_offset_min is not None and ee_payload_offset_max is not None:
        ee_body_id, ee_body_name = _resolve_ee_payload_body(str(xml_path), root_attrs, ds_cfg)
        print(
            "EE COM validation: "
            f"body='{ee_body_name}' (id={ee_body_id}), "
            f"payload_com_offset_box={np.asarray(ee_payload_offset_min).astype(float).tolist()}->"
            f"{np.asarray(ee_payload_offset_max).astype(float).tolist()}"
        )
    mjx_model = mjx_model.replace(body_gravcomp=jnp.zeros_like(mjx_model.body_gravcomp))
    print("Verification model: public TAM real-gravity ideal model (body_gravcomp=0).")

    external_force_body_id = -1
    if has_external_force:
        root_attrs = _read_root_attrs(first_path)
        external_force_body_id, external_force_body_name = _resolve_external_force_body(
            str(xml_path),
            root_attrs,
            ds_cfg,
        )
        print(
            f"Replaying external force with shard body ids when present; "
            f"fallback body_id={external_force_body_id}, body_name='{external_force_body_name}'"
        )

    end_idx = min(len(ds), args.start_index + args.num_shards)
    shards = []
    for shard_idx in tqdm(range(args.start_index, end_idx)):
        shard = jax.tree.map(lambda x: _to_jnp(x), ds[shard_idx])
        shards.append(shard)
    shards = jax.tree.map(lambda *x: jnp.concat(x, axis=0), *shards)

    ee_param_stats = None
    if (
        ee_body_id is not None
        and ee_body_name is not None
        and ee_payload_offset_min is not None
        and ee_payload_offset_max is not None
    ):
        ee_param_stats = _verify_ee_param_ranges(
            mjx_model,
            shards["params"],
            ee_body_id=ee_body_id,
            ee_body_name=ee_body_name,
            ee_payload_offset_min_local_m=ee_payload_offset_min,
            ee_payload_offset_max_local_m=ee_payload_offset_max,
        )
    
    stats = verify_rollout(
        mjx_model,
        shards["rollout"],
        shards["params"],
        external_force_body_id=external_force_body_id,
        max_steps=args.max_steps,
    )

    shard_max_q = float(jnp.max(stats["per_step_max_q"]))
    shard_max_qd = float(jnp.max(stats["per_step_max_qd"]))
    shard_max_qdd = float(jnp.max(stats["per_step_max_qdd"])) if stats["per_step_max_qdd"] is not None else None
    shard_max_time = float(jnp.max(stats["per_step_max_time"])) if stats["per_step_max_time"] is not None else None
    rollout_max_q = float(jnp.max(stats["rollout_max_q"]))
    rollout_max_qd = float(jnp.max(stats["rollout_max_qd"]))
    rollout_max_qdd = float(jnp.max(stats["rollout_max_qdd"])) if stats["rollout_max_qdd"] is not None else None
    rollout_max_time = float(jnp.max(stats["rollout_max_time"])) if stats["rollout_max_time"] is not None else None
    tau_recovery_max = float(jnp.max(stats["tau_recovery_max"]))
    tau_cmd_base_ideal_forceaware_max = float(jnp.max(stats["tau_cmd_base_ideal_forceaware_max"]))

    def _fmt(val: Optional[float]) -> str:
        return "-" if val is None else f"{val:.3e}"

    print(
        f"[shard {shard_idx}] "
        f"max|q|={_fmt(shard_max_q)}, "
        f"max|qd|={_fmt(shard_max_qd)}, "
        f"max|qdd|={_fmt(shard_max_qdd)}, "
        f"max|time drift|={_fmt(shard_max_time)}, "
        f"rollout max|q|={_fmt(rollout_max_q)}, "
        f"rollout max|qd|={_fmt(rollout_max_qd)}, "
        f"rollout max|qdd|={_fmt(rollout_max_qdd)}, "
        f"rollout max|time drift|={_fmt(rollout_max_time)}, "
        f"tau recovery max|u-gt|={_fmt(tau_recovery_max)}, "
        f"tau base(ideal-forceaware) max|u-base|={_fmt(tau_cmd_base_ideal_forceaware_max)}, "
        f"steps={stats['steps_checked']}"
    )
    if ee_param_stats is not None:
        print(
            "EE param stats: "
            f"delta_min={ee_param_stats['ee_delta_min'].astype(float).tolist()}, "
            f"delta_max={ee_param_stats['ee_delta_max'].astype(float).tolist()}, "
            f"mass_kg=[{ee_param_stats['ee_mass_min']:.4f}, {ee_param_stats['ee_mass_max']:.4f}], "
            f"inertia_min={ee_param_stats['ee_inertia_min'].astype(float).tolist()}, "
            f"inertia_max={ee_param_stats['ee_inertia_max'].astype(float).tolist()}"
        )

if __name__ == "__main__":
    main()
