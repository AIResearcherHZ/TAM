# training_simadaptor.py
from typing import Dict, Any, Optional, Iterable, List, Tuple
import os
import time
import glob
import datetime
import uuid
from dataclasses import asdict, is_dataclass

import numpy as np
from mujoco import mjx
import mujoco
import jax
import jax.numpy as jnp
from functools import partial
import tqdm
import zarr
from simadaptor.cli import parse_tyro_config
import simadaptor.data.datagen as datagen
from simadaptor.data.datagen_profiles import derive_robot_key, load_datagen_profile
import simadaptor.config as configs
from pathlib import Path
import shutil
import json

# CUR_TIME_STR = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


# -------------------------
# Zarr v3 helpers (save/load)
# -------------------------
def _to_np(x):
    # Materialize JAX arrays to numpy for storage; leave others unchanged.
    if isinstance(x, jnp.ndarray):
        return np.asarray(x)
    return x


def _guess_chunks(arr: np.ndarray) -> Tuple[int, ...]:
    # if arr.ndim == 3:
    #     B, T, D = arr.shape
    #     return (min(B, 8), min(T, 256), D)
    # if arr.ndim == 2:
    #     B, D = arr.shape
    #     return (min(B, 8), min(D, 256))
    return arr.shape


# v3: use zarr.codecs (not numcodecs directly) and `compressors=`
_ZSTD_BLOSC = zarr.codecs.BloscCodec(
    cname=zarr.codecs.BloscCname.zstd, clevel=8, shuffle=zarr.codecs.BloscShuffle.shuffle
)


def _create_and_write_array(g: zarr.Group, name: str, data: np.ndarray):
    """Create array with v3 API and write data."""
    # enforce fp32 to save space unless you need fp64
    if data.dtype == np.float64:
        data = data.astype(np.float32, copy=False)

    a = g.create_array(
        name=name,
        shape=data.shape,
        dtype=data.dtype,
        chunks=_guess_chunks(data),
        compressors=_ZSTD_BLOSC,
    )
    a[:] = data
    return a


_VALID_Q_ABS_LIMIT = 100.0
_VALID_QD_ABS_LIMIT = 100.0
_VALID_U_ABS_LIMIT = 1.0e8
_SAVE_MAX_BATCH_PER_FILE = 16
_PARAM_POSITIVE_EPS = 1.0e-8


def _print_actuator_order_diagnostics(xml_path: str) -> None:
    """Print actuator index ordering diagnostics for the input XML model."""
    mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
    nu = int(mj_model.nu)
    nv = int(mj_model.nv)
    if nu <= 0:
        print("[startup] actuator-order check: model has no actuators (nu=0).")
        return

    actuator_trnid = np.asarray(mj_model.actuator_trnid, dtype=np.int32)
    jnt_dofadr = np.asarray(mj_model.jnt_dofadr, dtype=np.int32)
    act_jnt_id = actuator_trnid[:, 0]
    act_jnt_id_clamped = np.clip(act_jnt_id, 0, int(mj_model.njnt) - 1)
    act_dof_id = jnt_dofadr[act_jnt_id_clamped]
    fallback_dof_id = np.minimum(np.arange(nu, dtype=np.int32), max(0, nv - 1))
    valid = (
        (act_jnt_id >= 0)
        & (act_jnt_id < int(mj_model.njnt))
        & (act_dof_id >= 0)
        & (act_dof_id < nv)
    )
    actuator_dof_idx = np.where(valid, act_dof_id, fallback_dof_id).astype(np.int32)
    expected = np.minimum(np.arange(nu, dtype=np.int32), max(0, nv - 1))
    mismatch = int(np.count_nonzero(actuator_dof_idx != expected))
    print(
        "[startup] actuator-order check: "
        f"nu={nu} nv={nv} mismatch={mismatch}/{nu}"
    )
    print(f"[startup] actuator->dof indices: {actuator_dof_idx.tolist()}")

    mapping_preview = []
    for i in range(nu):
        act_name = mj_model.actuator(i).name or f"actuator_{i}"
        j_id = int(act_jnt_id[i])
        if 0 <= j_id < int(mj_model.njnt):
            j_name = mj_model.joint(j_id).name or f"joint_{j_id}"
        else:
            j_name = f"joint_invalid_{j_id}"
        mapping_preview.append(f"{i}:{act_name}->{j_name}(dof={int(actuator_dof_idx[i])})")
    print("[startup] actuator mapping detail: " + ", ".join(mapping_preview))

def _infer_batch_size_from_rollout(rollout: Dict[str, Any]) -> int:
    for v in rollout.values():
        if v is None:
            continue
        arr = jnp.asarray(v)
        if arr.ndim >= 1:
            return int(arr.shape[0])
    raise ValueError("Could not infer batch size from rollout.")


def _per_traj_finite_mask(rollout: Dict[str, Any], batch_size: int) -> jnp.ndarray:
    mask = jnp.ones((batch_size,), dtype=bool)
    for v in rollout.values():
        if v is None:
            continue
        arr = jnp.asarray(v)
        if arr.ndim < 1 or int(arr.shape[0]) != batch_size:
            continue
        reduce_axes = tuple(range(1, arr.ndim))
        if reduce_axes:
            finite = jnp.all(jnp.isfinite(arr), axis=reduce_axes)
        else:
            finite = jnp.isfinite(arr)
        mask = mask & finite
    return mask


def _per_traj_abs_limit_mask(
    rollout: Dict[str, Any],
    batch_size: int,
    field: str,
    abs_limit: Optional[float],
) -> jnp.ndarray:
    if abs_limit is None or abs_limit <= 0.0:
        return jnp.ones((batch_size,), dtype=bool)
    if field not in rollout or rollout[field] is None:
        return jnp.ones((batch_size,), dtype=bool)

    arr = jnp.asarray(rollout[field])
    if arr.ndim < 1 or int(arr.shape[0]) != batch_size:
        return jnp.ones((batch_size,), dtype=bool)
    reduce_axes = tuple(range(1, arr.ndim))
    if reduce_axes:
        max_abs = jnp.max(jnp.abs(arr), axis=reduce_axes)
    else:
        max_abs = jnp.abs(arr)
    return max_abs <= float(abs_limit)


def _compute_valid_traj_mask(
    perturbed_rollout: Dict[str, Any],
    original_rollout: Optional[Dict[str, Any]],
    *,
    q_abs_limit: float = _VALID_Q_ABS_LIMIT,
    qd_abs_limit: float = _VALID_QD_ABS_LIMIT,
    u_abs_limit: float = _VALID_U_ABS_LIMIT,
) -> jnp.ndarray:
    batch_size = _infer_batch_size_from_rollout(perturbed_rollout)
    valid_mask = _per_traj_finite_mask(perturbed_rollout, batch_size)
    valid_mask = valid_mask & _per_traj_abs_limit_mask(perturbed_rollout, batch_size, "q", q_abs_limit)
    valid_mask = valid_mask & _per_traj_abs_limit_mask(perturbed_rollout, batch_size, "qd", qd_abs_limit)
    valid_mask = valid_mask & _per_traj_abs_limit_mask(perturbed_rollout, batch_size, "u", u_abs_limit)

    if original_rollout is not None:
        valid_mask = valid_mask & _per_traj_finite_mask(original_rollout, batch_size)
        valid_mask = valid_mask & _per_traj_abs_limit_mask(original_rollout, batch_size, "q", q_abs_limit)
        valid_mask = valid_mask & _per_traj_abs_limit_mask(original_rollout, batch_size, "qd", qd_abs_limit)

    return valid_mask


def _rollout_params_to_dict(params: Any) -> Dict[str, Any]:
    if is_dataclass(params):
        return asdict(params)
    if isinstance(params, dict):
        return params
    return {k: getattr(params, k) for k in dir(params) if not k.startswith("_")}


def _infer_batch_size_from_params(params: Any) -> int:
    params_dict = _rollout_params_to_dict(params)
    for v in params_dict.values():
        if v is None:
            continue
        arr = jnp.asarray(v)
        if arr.ndim >= 1:
            return int(arr.shape[0])
    raise ValueError("Could not infer batch size from rollout params.")


def _compute_valid_param_mask(
    params: Any,
    *,
    mjx_model: mjx.Model,
    ee_body_id: Optional[int] = None,
    ee_payload_com_offset_min_local_m: Optional[jnp.ndarray] = None,
    ee_payload_com_offset_max_local_m: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    params_dict = _rollout_params_to_dict(params)
    batch_size = _infer_batch_size_from_params(params_dict)
    valid_mask = _per_traj_finite_mask(params_dict, batch_size)

    positive_body_mask = jnp.asarray(np.asarray(mjx_model.body_mass) > 0, dtype=bool)

    body_mass = params_dict.get("body_mass", None)
    if body_mass is not None:
        body_mass = jnp.asarray(body_mass)
        if body_mass.ndim >= 2 and int(body_mass.shape[0]) == batch_size:
            valid_mask = valid_mask & jnp.all(
                body_mass[:, positive_body_mask] > _PARAM_POSITIVE_EPS,
                axis=-1,
            )

    body_inertia = params_dict.get("body_inertia", None)
    if body_inertia is not None:
        body_inertia = jnp.asarray(body_inertia)
        if body_inertia.ndim >= 3 and int(body_inertia.shape[0]) == batch_size:
            valid_mask = valid_mask & jnp.all(
                body_inertia[:, positive_body_mask, :] > _PARAM_POSITIVE_EPS,
                axis=(-2, -1),
            )

    body_ipos = params_dict.get("body_ipos", None)
    if (
        body_ipos is not None
        and ee_body_id is not None
        and ee_payload_com_offset_min_local_m is not None
        and ee_payload_com_offset_max_local_m is not None
    ):
        body_ipos = jnp.asarray(body_ipos)
        if (
            body_ipos.ndim >= 3
            and int(body_ipos.shape[0]) == batch_size
            and int(body_ipos.shape[1]) > int(ee_body_id)
        ):
            ee_delta = body_ipos[:, int(ee_body_id), :] - jnp.asarray(
                mjx_model.body_ipos[int(ee_body_id)], dtype=body_ipos.dtype
            )
            ee_abs_bound = jnp.maximum(
                jnp.abs(
                    jnp.asarray(ee_payload_com_offset_min_local_m, dtype=body_ipos.dtype).reshape((3,))
                ),
                jnp.abs(
                    jnp.asarray(ee_payload_com_offset_max_local_m, dtype=body_ipos.dtype).reshape((3,))
                ),
            )
            tol = jnp.asarray(1.0e-6, dtype=body_ipos.dtype)
            valid_mask = valid_mask & jnp.all(jnp.abs(ee_delta) <= (ee_abs_bound + tol), axis=-1)

    return valid_mask


def _summarize_ee_com_meta(
    params: Any,
    *,
    mjx_model: mjx.Model,
    ee_body_id: int,
    ee_body_name: str,
) -> Dict[str, Any]:
    params_dict = _rollout_params_to_dict(params)
    body_ipos = params_dict.get("body_ipos", None)
    if body_ipos is None:
        return {}

    body_ipos_np = np.asarray(body_ipos, dtype=np.float32)
    if body_ipos_np.ndim < 3 or ee_body_id >= int(body_ipos_np.shape[1]):
        return {}

    ee_nominal_ipos = np.asarray(mjx_model.body_ipos[ee_body_id], dtype=np.float32).reshape((1, 3))
    ee_delta = body_ipos_np[:, ee_body_id, :] - ee_nominal_ipos

    meta: Dict[str, Any] = {
        "ee_payload_body_id": int(ee_body_id),
        "ee_payload_body_name": str(ee_body_name),
        "ee_body_ipos_delta_frame": "body_local",
        "ee_body_ipos_delta_min_observed_local_m": ee_delta.min(axis=0).astype(float).tolist(),
        "ee_body_ipos_delta_max_observed_local_m": ee_delta.max(axis=0).astype(float).tolist(),
    }

    body_inertia = params_dict.get("body_inertia", None)
    if body_inertia is not None:
        body_inertia_np = np.asarray(body_inertia, dtype=np.float32)
        if body_inertia_np.ndim >= 3 and ee_body_id < int(body_inertia_np.shape[1]):
            meta["ee_body_inertia_min_observed"] = (
                body_inertia_np[:, ee_body_id, :].min(axis=0).astype(float).tolist()
            )
            meta["ee_body_inertia_max_observed"] = (
                body_inertia_np[:, ee_body_id, :].max(axis=0).astype(float).tolist()
            )

    body_mass = params_dict.get("body_mass", None)
    if body_mass is not None:
        body_mass_np = np.asarray(body_mass, dtype=np.float32)
        if body_mass_np.ndim >= 2 and ee_body_id < int(body_mass_np.shape[1]):
            meta["ee_body_mass_min_observed_kg"] = float(body_mass_np[:, ee_body_id].min())
            meta["ee_body_mass_max_observed_kg"] = float(body_mass_np[:, ee_body_id].max())

    return meta


def _filter_batched_tree(tree: Any, keep_indices: np.ndarray, batch_size: int) -> Any:
    def _maybe_take(x):
        if x is None:
            return None
        if isinstance(x, (jnp.ndarray, np.ndarray)) and x.ndim >= 1 and int(x.shape[0]) == batch_size:
            return x[keep_indices]
        return x

    return jax.tree_util.tree_map(_maybe_take, tree)


def _copy_robot_assets(
    xml_path: str,
    out_base: str,
    *,
    robot_key: str,
):
    xml_path = Path(xml_path)
    out_base = Path(out_base)
    robot_dir = out_base / "robot_model"
    robot_dir.mkdir(parents=True, exist_ok=True)
    dst_xml = robot_dir / "robot.xml"
    if not dst_xml.exists():
        shutil.copy2(xml_path, dst_xml)
    src_assets = xml_path.parent / "assets"
    if src_assets.exists():
        dst_assets = robot_dir / "assets"
        if not dst_assets.exists():
            shutil.copytree(src_assets, dst_assets, dirs_exist_ok=True)
    # write manifest
    manifest = {
        "robot_key": robot_key,
        "robot_xml": str(dst_xml.relative_to(out_base)),
        "source_xml": str(xml_path),
    }
    with open(robot_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return dst_xml


def _resolve_external_force_body(
    xml_path: str,
    body_name: Optional[str],
) -> Tuple[int, str]:
    """
    Resolve external-force target body id from body name.
    """
    mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
    if body_name:
        body_id = int(mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name))
        if body_id >= 0:
            return body_id, body_name

    raise ValueError(
        f"Failed to resolve external-force target. "
        f"body_name={body_name!r}, xml={xml_path!r}"
    )


def _resolve_end_effector_body(xml_path: str) -> Tuple[int, str]:
    mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
    if int(mj_model.nsite) > 0:
        body_id = int(np.asarray(mj_model.site_bodyid, dtype=np.int32)[-1])
    else:
        dof_bodyid = np.asarray(mj_model.dof_bodyid, dtype=np.int32)
        if dof_bodyid.size == 0:
            raise ValueError(f"Model at {xml_path!r} has no sites or dofs to resolve an end effector.")
        ee_dof_id = max(min(int(mj_model.nu), int(mj_model.nv)) - 1, 0)
        body_id = int(dof_bodyid[min(ee_dof_id, int(dof_bodyid.size) - 1)])
    body_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
    return body_id, body_name


def _resolve_external_force_targets(
    *,
    cfg_body_name: Optional[str],
    profile_force_targets: Optional[List[Dict[str, Any]]],
    legacy_position_min_local_m: Optional[jnp.ndarray],
    legacy_position_max_local_m: Optional[jnp.ndarray],
) -> List[Dict[str, Any]]:
    if profile_force_targets:
        return [
            {
                "body_name": str(target["body_name"]),
                "position_min_local_m": (
                    None
                    if target.get("position_min_local_m", None) is None
                    else jnp.asarray(target["position_min_local_m"], dtype=jnp.float32).reshape((3,))
                ),
                "position_max_local_m": (
                    None
                    if target.get("position_max_local_m", None) is None
                    else jnp.asarray(target["position_max_local_m"], dtype=jnp.float32).reshape((3,))
                ),
            }
            for target in profile_force_targets
        ]

    if cfg_body_name is None or str(cfg_body_name).strip() == "":
        return []

    return [
        {
            "body_name": str(cfg_body_name),
            "position_min_local_m": (
                None
                if legacy_position_min_local_m is None
                else jnp.asarray(legacy_position_min_local_m, dtype=jnp.float32).reshape((3,))
            ),
            "position_max_local_m": (
                None
                if legacy_position_max_local_m is None
                else jnp.asarray(legacy_position_max_local_m, dtype=jnp.float32).reshape((3,))
            ),
        }
    ]


def _external_force_target_cache_key(
    force_enabled: bool,
    target: Optional[Dict[str, Any]],
) -> Tuple[Any, ...]:
    if not force_enabled or target is None:
        return (False,)
    pos_min = target.get("position_min_local_m", None)
    pos_max = target.get("position_max_local_m", None)
    if pos_min is None or pos_max is None:
        return (True, str(target["body_name"]), None, None)
    return (
        True,
        str(target["body_name"]),
        tuple(np.asarray(pos_min, dtype=np.float32).reshape(3).astype(float).tolist()),
        tuple(np.asarray(pos_max, dtype=np.float32).reshape(3).astype(float).tolist()),
    )

def _canonicalize_config_value(value: Any) -> Any:
    """Convert config values into JSON-stable Python types for compare/write."""
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _canonicalize_config_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonicalize_config_value(v) for v in value]
    if isinstance(value, (np.ndarray, jnp.ndarray)):
        return _canonicalize_config_value(np.asarray(value).tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _build_effective_dataset_config(
    cfg: configs.DataConfig,
    *,
    robot_key: str,
    profile_key: str,
    datagen_profile_fields: List[str],
    datagen_profile_kwargs: Dict[str, Any],
    configured_force_targets: List[Dict[str, Any]],
    external_force_magnitude_min_n: float,
    external_force_magnitude_max_n: float,
    ee_payload_mass_delta_range: Any,
    ee_payload_com_offset_min_local_m: Any,
    ee_payload_com_offset_max_local_m: Any,
    joint_model_major_ee_scale: float,
    joint_model_major_global_scale: float,
) -> Dict[str, Any]:
    cfg_dict = _canonicalize_config_value(asdict(cfg) if is_dataclass(cfg) else dict(cfg))
    cfg_dict.pop("ideal_model_has_gravity", None)
    cfg_dict["robot_key"] = str(robot_key)
    cfg_dict["datagen_profile_key"] = str(profile_key)
    cfg_dict["datagen_profile_fields"] = list(datagen_profile_fields)
    cfg_dict["external_force_magnitude_min_n"] = float(external_force_magnitude_min_n)
    cfg_dict["external_force_magnitude_max_n"] = float(external_force_magnitude_max_n)
    cfg_dict["external_force_apply_to_original"] = bool(
        cfg.external_force_apply_to_original and bool(getattr(cfg, "save_original_split", True))
    )
    cfg_dict["ee_payload_mass_delta_range"] = _canonicalize_config_value(
        ee_payload_mass_delta_range
    )
    cfg_dict["ee_payload_com_offset_min_local_m"] = _canonicalize_config_value(
        ee_payload_com_offset_min_local_m
    )
    cfg_dict["ee_payload_com_offset_max_local_m"] = _canonicalize_config_value(
        ee_payload_com_offset_max_local_m
    )
    cfg_dict["joint_model_major_ee_scale"] = float(joint_model_major_ee_scale)
    cfg_dict["joint_model_major_global_scale"] = float(joint_model_major_global_scale)
    cfg_dict["resolved_external_force_targets"] = _canonicalize_config_value(
        configured_force_targets
    )
    for key, value in datagen_profile_kwargs.items():
        cfg_dict[str(key)] = _canonicalize_config_value(value)
    return cfg_dict


def _write_or_check_dataset_config(
    cfg_or_dict: configs.DataConfig | Dict[str, Any],
    dataset_dir: str | Path,
) -> Path:
    """
    Persist the dataset generation config into `dataset_dir`.
    If the config file already exists, require an exact match to prevent mixing datasets.
    """
    base = Path(dataset_dir)
    base.mkdir(parents=True, exist_ok=True)
    cfg_path = base / "data_generation_config.json"

    cfg_dict = _canonicalize_config_value(cfg_or_dict)
    if not bool(cfg_dict.pop("ideal_model_has_gravity", True)):
        raise ValueError("Public TAM data generation supports only the real-gravity ideal model.")

    def _atomic_write_config() -> None:
        tmp_path = cfg_path.with_name(
            f"{cfg_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            with open(tmp_path, "w") as f:
                json.dump(cfg_dict, f, indent=2, sort_keys=True)
                f.write("\n")
            os.replace(tmp_path, cfg_path)
        finally:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass

    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            existing = json.load(f)
        legacy_gravity = bool(existing.pop("ideal_model_has_gravity", True))
        if not legacy_gravity:
            raise ValueError(
                "Existing dataset config uses the floating ideal-model contract. "
                "Regenerate the dataset folder for public TAM."
            )
        existing = _canonicalize_config_value(existing)
        diffs = []
        for k in sorted(existing.keys()):
            if k not in cfg_dict or existing[k] != cfg_dict[k]:
                diffs.append((k, existing.get(k), cfg_dict.get(k)))
        if diffs:
            diff_str = "\n".join(f"- {k}: existing={old!r} new={new!r}" for k, old, new in diffs)
            raise ValueError(
                f"Dataset config mismatch for {cfg_path}.\n"
                f"Refusing to write into {base} with different settings:\n{diff_str}"
            )
        if existing != cfg_dict:
            print(
                f"Updating dataset config schema at {cfg_path} "
                "to the current canonical effective format."
            )
            _atomic_write_config()
        return cfg_path

    _atomic_write_config()
    return cfg_path

def save_rollout_zarr_v3(
    path: str,
    rollout: Dict[str, Any],
    params: Any,
    meta: Optional[Dict[str, Any]] = None,
    overwrite: bool = True,
    max_batch_per_file: int = 16,
) -> None:
    """
    Save rollout/params to Zarr v3.
    If batch size B > max_batch_per_file, split across multiple files with suffix:
      <path>.zarr                (if B <= max_batch_per_file)
      <path_without_suffix>_part0000.zarr, _part0001.zarr, ...
    Layout inside each file:
      /rollout/<key> : arrays
      /params/<key>  : arrays
      group attrs    : meta + sharding info
    """
    # --- figure out batch size B from any rollout array ---
    if not rollout:
        raise ValueError("rollout dict is empty; cannot infer batch size.")
    # materialize rollout to numpy to avoid jax arrays in storage
    rollout = jax.tree_util.tree_map(_to_np, rollout)
    # choose a reference key with an array
    ref_key = next(k for k, v in rollout.items() if v is not None)
    ref_arr = _to_np(rollout[ref_key])
    if not hasattr(ref_arr, "shape") or ref_arr.ndim == 0:
        raise ValueError(f"rollout['{ref_key}'] is not an array with batch dim.")
    B = ref_arr.shape[0]

    # normalize path pieces for sharded filenames
    def _strip_zarr_suffix(p: str) -> str:
        for suf in (".zarr.zip", ".zip", ".zarr"):
            if p.endswith(suf):
                return p[: -len(suf)]
        return p

    base_noext = _strip_zarr_suffix(path)

    # shard ranges
    if B <= max_batch_per_file:
        shard_specs = [(0, B, None)]  # (start, end, suffix)
    else:
        shard_specs = []
        part = 0
        for s in range(0, B, max_batch_per_file):
            e = min(B, s + max_batch_per_file)
            shard_specs.append((s, e, f"_part{part:04d}"))
            part += 1

    # prepare params dict
    if is_dataclass(params):
        p_dict = asdict(params)
    elif isinstance(params, dict):
        p_dict = params
    else:
        p_dict = {k: getattr(params, k) for k in dir(params) if not k.startswith("_")}
    # materialize params values
    p_dict = jax.tree_util.tree_map(_to_np, p_dict)

    # helper: slice value along batch axis if it matches B
    def slice_maybe(v, s, e):
        v = _to_np(v)
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == B:
            return v[s:e]
        return v  # scalar or not per-batch; keep as-is

    # write each shard
    num_parts = len(shard_specs)
    for part_idx, (s, e, suffix) in enumerate(shard_specs):
        out_path = f"{base_noext}{suffix}.zarr.zip"

        # open zarr group (overwrite each shard)
        mode = "w" if overwrite else "w-"
        store = zarr.storage.ZipStore(out_path, mode=mode)
        root = zarr.open_group(store=store, mode=mode)

        # create subgroups (v3; no exist_ok kwarg)
        g_rollout = root.create_group("rollout")
        g_params  = root.create_group("params")

        # rollout arrays (slice on [s:e] along batch axis 0)
        for k, v in rollout.items():
            if v is None:
                continue
            arr = slice_maybe(v, s, e)
            _create_and_write_array(g_rollout, k, arr)

        # params arrays (slice if batch-aligned, else copy)
        for k, v in p_dict.items():
            if v is None:
                continue
            arr = slice_maybe(v, s, e)
            if isinstance(arr, (int, float, np.generic)):
                arr = np.asarray(arr)
            if isinstance(arr, np.ndarray):
                if arr.dtype == np.float64:
                    arr = arr.astype(np.float32, copy=False)
                _create_and_write_array(g_params, k, arr)

        # attrs with sharding info
        attrs = dict(meta or {})
        attrs.setdefault("saved_at", out_path)
        attrs.update(
            dict(
                batch_total=int(B),
                shard_start=int(s),
                shard_end=int(e),
                shard_size=int(e - s),
                num_parts=int(num_parts),
                part_index=int(part_idx),
                max_batch_per_file=int(max_batch_per_file),
            )
        )
        for mk, mv in attrs.items():
            root.attrs[mk] = mv

        store.close()


# =========================
# Main
# =========================
def main(cfg: configs.DataConfig):
    dataset_root = Path(cfg.dataset_base_path)
    dataset_root.mkdir(parents=True, exist_ok=True)
    robot_key = derive_robot_key(cfg.xml_path)
    profile_key, datagen_profile_kwargs = load_datagen_profile(
        table_path=cfg.datagen_profile_table_path,
        robot_key=robot_key,
        profile_key=cfg.datagen_profile_key,
    )
    datagen_profile_fields = sorted(datagen_profile_kwargs.keys())
    datagen_profile_kwargs = dict(datagen_profile_kwargs)
    raw_datagen_profile_kwargs = dict(datagen_profile_kwargs)
    profile_force_mag_min = datagen_profile_kwargs.pop("external_force_magnitude_min_n", None)
    profile_force_mag_max = datagen_profile_kwargs.pop("external_force_magnitude_max_n", None)
    profile_force_pos_min = datagen_profile_kwargs.pop("external_force_position_min_local_m", None)
    profile_force_pos_max = datagen_profile_kwargs.pop("external_force_position_max_local_m", None)
    profile_force_targets = datagen_profile_kwargs.pop("external_force_targets", None)
    datagen_profile_kwargs.pop("external_force_body_names", None)
    datagen_profile_kwargs.pop("external_force_position_boxes_local_m_by_body_name", None)
    external_force_magnitude_min_n = float(
        cfg.external_force_magnitude_min_n
        if profile_force_mag_min is None
        else profile_force_mag_min
    )
    external_force_magnitude_max_n = float(
        cfg.external_force_magnitude_max_n
        if profile_force_mag_max is None
        else profile_force_mag_max
    )
    if not np.isfinite(external_force_magnitude_min_n) or not np.isfinite(external_force_magnitude_max_n):
        raise ValueError(
            "External-force magnitude bounds must be finite. "
            f"Got min={external_force_magnitude_min_n}, max={external_force_magnitude_max_n}."
        )
    if external_force_magnitude_min_n < 0.0 or external_force_magnitude_max_n < 0.0:
        raise ValueError(
            "External-force magnitude bounds must be non-negative. "
            f"Got min={external_force_magnitude_min_n}, max={external_force_magnitude_max_n}."
        )
    if external_force_magnitude_max_n < external_force_magnitude_min_n:
        raise ValueError(
            "External-force magnitude max must be >= min. "
            f"Got min={external_force_magnitude_min_n}, max={external_force_magnitude_max_n}."
        )
    ee_payload_mass_delta_range = datagen_profile_kwargs.get(
        "ee_payload_mass_delta_range", cfg.ee_payload_mass_delta_range
    )
    ee_payload_com_offset_min_local_m = datagen_profile_kwargs.get(
        "ee_payload_com_offset_min_local_m", cfg.ee_payload_com_offset_min_local_m
    )
    ee_payload_com_offset_max_local_m = datagen_profile_kwargs.get(
        "ee_payload_com_offset_max_local_m", cfg.ee_payload_com_offset_max_local_m
    )
    joint_model_major_ee_scale = float(
        datagen_profile_kwargs.get("joint_model_major_ee_scale", cfg.joint_model_major_ee_scale)
    )
    joint_model_major_global_scale = float(
        datagen_profile_kwargs.get(
            "joint_model_major_global_scale",
            cfg.joint_model_major_global_scale,
        )
    )
    ee_body_id, ee_body_name = _resolve_end_effector_body(cfg.xml_path)
    save_original_split = bool(getattr(cfg, "save_original_split", True))
    configured_force_targets = _resolve_external_force_targets(
        cfg_body_name=cfg.external_force_body_name,
        profile_force_targets=profile_force_targets,
        legacy_position_min_local_m=profile_force_pos_min,
        legacy_position_max_local_m=profile_force_pos_max,
    )
    robot_dataset_base = dataset_root / robot_key
    robot_dataset_base.mkdir(parents=True, exist_ok=True)
    effective_dataset_cfg = _build_effective_dataset_config(
        cfg,
        robot_key=robot_key,
        profile_key=profile_key,
        datagen_profile_fields=datagen_profile_fields,
        datagen_profile_kwargs=raw_datagen_profile_kwargs,
        configured_force_targets=configured_force_targets,
        external_force_magnitude_min_n=external_force_magnitude_min_n,
        external_force_magnitude_max_n=external_force_magnitude_max_n,
        ee_payload_mass_delta_range=ee_payload_mass_delta_range,
        ee_payload_com_offset_min_local_m=ee_payload_com_offset_min_local_m,
        ee_payload_com_offset_max_local_m=ee_payload_com_offset_max_local_m,
        joint_model_major_ee_scale=joint_model_major_ee_scale,
        joint_model_major_global_scale=joint_model_major_global_scale,
    )
    _write_or_check_dataset_config(effective_dataset_cfg, robot_dataset_base)
    print(f"Dataset root: {dataset_root}")
    print(f"Robot dataset dir: {robot_dataset_base} (robot_key={robot_key})")
    print(
        f"Datagen profile: key='{profile_key}', "
        f"fields={datagen_profile_fields}"
    )
    if profile_force_mag_min is not None or profile_force_mag_max is not None:
        print(
            "External-force magnitude override from datagen profile: "
            f"min={external_force_magnitude_min_n}, max={external_force_magnitude_max_n}"
        )
    if profile_force_pos_min is not None and profile_force_pos_max is not None:
        pos_min_np = np.asarray(profile_force_pos_min, dtype=np.float32).reshape(3)
        pos_max_np = np.asarray(profile_force_pos_max, dtype=np.float32).reshape(3)
        print(
            "External-force position override from datagen profile: "
            f"min_local_m={pos_min_np.tolist()}, max_local_m={pos_max_np.tolist()}"
        )
    print(
        "End-effector COM randomization: "
        f"body='{ee_body_name}' (id={ee_body_id}), "
        f"payload_mass_delta_range={tuple(float(x) for x in ee_payload_mass_delta_range)}, "
        f"payload_com_offset_box={np.asarray(ee_payload_com_offset_min_local_m, dtype=np.float32).reshape(3).tolist()}->"
        f"{np.asarray(ee_payload_com_offset_max_local_m, dtype=np.float32).reshape(3).tolist()}, "
        f"joint_model_major_ee_scale={joint_model_major_ee_scale}, "
        f"joint_model_major_global_scale={joint_model_major_global_scale}"
    )
    _print_actuator_order_diagnostics(cfg.xml_path)
    CUR_TIME_STR = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---- Setup model/env ----
    import simadaptor.physics.dynamics as dynamics
    sim_timestep = float(cfg.sim_timestep)
    if not np.isfinite(sim_timestep) or sim_timestep <= 0.0:
        raise ValueError(f"cfg.sim_timestep must be positive and finite, got {sim_timestep!r}.")
    mjx_model = dynamics.load_mjx_model_from_path(
        cfg.xml_path,
        remove_constraints=True,
        timestep=sim_timestep,
    )  # warmup cache
    mjx_model = mjx_model.replace(jnt_limited=np.zeros_like(mjx_model.jnt_limited)) # remove constraints
    model_dt = float(np.asarray(mjx_model.opt.timestep).reshape(()))
    if not np.isclose(model_dt, sim_timestep, rtol=0.0, atol=1e-8):
        raise ValueError(f"MJX model timestep mismatch: model={model_dt}, cfg={sim_timestep}.")
    print(f"Datagen sim timestep: {sim_timestep:g}s (model={model_dt:g}s)")
    print("Using public TAM real-gravity ideal model (body_gravcomp=0).")
    mjx_model = mjx_model.replace(body_gravcomp=jnp.zeros_like(mjx_model.body_gravcomp))
    dof_idx_arm = jnp.arange(mjx_model.nv)
    asset_xml_path = _copy_robot_assets(
        cfg.xml_path,
        robot_dataset_base,
        robot_key=robot_key,
    )
    asset_rel_path = str(asset_xml_path.relative_to(robot_dataset_base))

    force_enabled = (
        int(cfg.external_force_num_impulses) > 0
        and (
            bool(cfg.external_force_apply_to_perturbed)
            or (save_original_split and bool(cfg.external_force_apply_to_original))
        )
    )
    if force_enabled:
        if not configured_force_targets:
            raise ValueError(
                "External force is enabled, but no valid external-force target bodies were resolved. "
                "Set cfg.external_force_body_name or provide datagen profile "
                "'external_force_body_names' with per-body boxes."
            )
        target_preview = []
        for target in configured_force_targets:
            _resolve_external_force_body(cfg.xml_path, target["body_name"])
            pos_min = target.get("position_min_local_m", None)
            pos_max = target.get("position_max_local_m", None)
            if pos_min is not None and pos_max is not None:
                target_preview.append(
                    f"{target['body_name']}:{np.asarray(pos_min, dtype=np.float32).reshape(3).tolist()}->"
                    f"{np.asarray(pos_max, dtype=np.float32).reshape(3).tolist()}"
                )
            else:
                target_preview.append(f"{target['body_name']}:origin")
        print(
            "External force enabled: "
            f"targets={target_preview}, "
            f"num_impulses={cfg.external_force_num_impulses}, "
            f"magnitude_n=[{external_force_magnitude_min_n}, {external_force_magnitude_max_n}], "
            "sampling=uniform_per_shard"
        )
    else:
        print("External force disabled (num_impulses==0 or both split toggles are off).")

    # ---- RNG ----
    rng = jax.random.PRNGKey(np.random.randint(0, 1 << 31))

    base_force_meta = {
        "robot_key": robot_key,
        "external_force_field": "external_force_ee",
        "external_force_world_frame": True,
        "external_force_num_impulses": int(cfg.external_force_num_impulses),
        "external_force_magnitude_min_n": float(external_force_magnitude_min_n),
        "external_force_magnitude_max_n": float(external_force_magnitude_max_n),
        "external_force_duration_min_s": float(cfg.external_force_duration_min_s),
        "external_force_duration_max_s": float(cfg.external_force_duration_max_s),
        "external_force_apply_to_perturbed": bool(cfg.external_force_apply_to_perturbed),
        "external_force_apply_to_original": bool(cfg.external_force_apply_to_original and save_original_split),
        "save_original_split": save_original_split,
        "sim_timestep": float(sim_timestep),
        "ee_payload_body_id": int(ee_body_id),
        "ee_payload_body_name": ee_body_name,
        "ee_payload_mass_delta_range": [float(x) for x in ee_payload_mass_delta_range],
        "ee_payload_com_offset_min_local_m": np.asarray(
            ee_payload_com_offset_min_local_m, dtype=np.float32
        ).reshape(3).astype(float).tolist(),
        "ee_payload_com_offset_max_local_m": np.asarray(
            ee_payload_com_offset_max_local_m, dtype=np.float32
        ).reshape(3).astype(float).tolist(),
        "joint_model_major_ee_scale": float(joint_model_major_ee_scale),
        "joint_model_major_global_scale": float(joint_model_major_global_scale),
    }
    data_generation_jit_cache: Dict[Tuple[Any, ...], Any] = {}
    base_generation_kwargs = dict(
        mjx_model=mjx_model,
        dof_idx_arm=dof_idx_arm,
        batch_size=cfg.history_batch,
        num_waypoints=cfg.num_waypoints_history,
        duration=cfg.history_duration,
        dt=sim_timestep,
        filter_key=True,
        pause_prob=cfg.pause_prob,
        external_force_num_impulses=cfg.external_force_num_impulses,
        external_force_magnitude_min_n=external_force_magnitude_min_n,
        external_force_magnitude_max_n=external_force_magnitude_max_n,
        external_force_duration_min_s=cfg.external_force_duration_min_s,
        external_force_duration_max_s=cfg.external_force_duration_max_s,
        external_force_apply_to_perturbed=cfg.external_force_apply_to_perturbed,
        external_force_apply_to_original=(cfg.external_force_apply_to_original and save_original_split),
        ee_payload_mass_delta_range=ee_payload_mass_delta_range,
        ee_payload_com_offset_min_local_m=jnp.asarray(
            ee_payload_com_offset_min_local_m, dtype=jnp.float32
        ).reshape((3,)),
        ee_payload_com_offset_max_local_m=jnp.asarray(
            ee_payload_com_offset_max_local_m, dtype=jnp.float32
        ).reshape((3,)),
        joint_model_major_ee_scale=joint_model_major_ee_scale,
        joint_model_major_global_scale=joint_model_major_global_scale,
        **datagen_profile_kwargs,
    )

    t0 = time.time()
    for step in tqdm.tqdm(range(1, cfg.num_steps + 1)):
        rng, data_generation_key, target_select_key = jax.random.split(rng, 3)
        sampled_force_target = None
        force_body_id = -1
        force_body_name = ""
        if force_enabled:
            if len(configured_force_targets) == 1:
                sampled_force_target = configured_force_targets[0]
            else:
                target_idx = int(
                    jax.random.randint(
                        target_select_key,
                        shape=(),
                        minval=0,
                        maxval=len(configured_force_targets),
                    )
                )
                sampled_force_target = configured_force_targets[target_idx]
            force_body_id, force_body_name = _resolve_external_force_body(
                cfg.xml_path,
                sampled_force_target["body_name"],
            )
        target_cache_key = _external_force_target_cache_key(force_enabled, sampled_force_target)
        data_generation_jit = data_generation_jit_cache.get(target_cache_key, None)
        if data_generation_jit is None:
            generation_kwargs = dict(base_generation_kwargs)
            generation_kwargs["external_force_body_id"] = (force_body_id if force_enabled else None)
            if sampled_force_target is not None:
                generation_kwargs["external_force_position_min_local_m"] = sampled_force_target.get(
                    "position_min_local_m", None
                )
                generation_kwargs["external_force_position_max_local_m"] = sampled_force_target.get(
                    "position_max_local_m", None
                )
            else:
                generation_kwargs["external_force_position_min_local_m"] = None
                generation_kwargs["external_force_position_max_local_m"] = None
            data_generation_jit = jax.jit(partial(datagen.data_generation, **generation_kwargs))
            data_generation_jit_cache[target_cache_key] = data_generation_jit

        start_time = time.time()
        rollout_inputs, test_rollout, perturbed_rollout_params, original_rollout_params = data_generation_jit(
            data_generation_key
        )

        # materialize before writing
        jax.block_until_ready(rollout_inputs)
        if save_original_split:
            jax.block_until_ready(test_rollout)

        if getattr(perturbed_rollout_params, "torque_scale", None) is None:
            raise ValueError("perturbed_rollout_params.torque_scale is missing; regenerate with updated datagen.")

        # Keep only valid trajectories in the sampled batch.
        valid_rollout_mask = _compute_valid_traj_mask(
            rollout_inputs,
            (test_rollout if save_original_split else None),
            q_abs_limit=_VALID_Q_ABS_LIMIT,
            qd_abs_limit=_VALID_QD_ABS_LIMIT,
            u_abs_limit=_VALID_U_ABS_LIMIT,
        )
        valid_param_mask = _compute_valid_param_mask(
            perturbed_rollout_params,
            mjx_model=mjx_model,
            ee_body_id=ee_body_id,
            ee_payload_com_offset_min_local_m=jnp.asarray(
                ee_payload_com_offset_min_local_m, dtype=jnp.float32
            ),
            ee_payload_com_offset_max_local_m=jnp.asarray(
                ee_payload_com_offset_max_local_m, dtype=jnp.float32
            ),
        )
        if save_original_split:
            valid_param_mask = valid_param_mask & _compute_valid_param_mask(
                original_rollout_params,
                mjx_model=mjx_model,
            )
        valid_mask = valid_rollout_mask & valid_param_mask
        valid_rollout_mask_np = np.asarray(valid_rollout_mask, dtype=bool)
        valid_param_mask_np = np.asarray(valid_param_mask, dtype=bool)
        valid_mask_np = np.asarray(valid_mask, dtype=bool)
        batch_total = int(valid_mask_np.shape[0])
        valid_indices = np.flatnonzero(valid_mask_np)
        invalid_indices = np.flatnonzero(~valid_mask_np)
        num_valid_raw = int(valid_indices.shape[0])
        num_invalid = int(invalid_indices.shape[0])
        num_invalid_rollout = int(np.count_nonzero(~valid_rollout_mask_np))
        num_invalid_params = int(np.count_nonzero(~valid_param_mask_np))

        if num_invalid > 0:
            preview = invalid_indices[:24].tolist()
            suffix = "..." if num_invalid > len(preview) else ""
            print(
                f"Step {step}: filtered invalid trajectories "
                f"{num_invalid}/{batch_total}. "
                f"invalid_rollout={num_invalid_rollout}, invalid_params={num_invalid_params}. "
                f"invalid_indices={preview}{suffix}"
            )
        if num_valid_raw <= 0:
            print(
                f"Step {step}: no valid trajectories in sampled batch "
                f"(limits: |q|<={_VALID_Q_ABS_LIMIT}, |qd|<={_VALID_QD_ABS_LIMIT}, |u|<={_VALID_U_ABS_LIMIT}, "
                "plus param/body-IPOS validation); skipping save."
            )
            continue

        # Keep only full shard-size multiples so every saved shard has identical batch size.
        num_valid_aligned = (num_valid_raw // _SAVE_MAX_BATCH_PER_FILE) * _SAVE_MAX_BATCH_PER_FILE
        num_dropped_for_chunk_alignment = int(num_valid_raw - num_valid_aligned)
        if num_dropped_for_chunk_alignment > 0:
            print(
                f"Step {step}: dropped {num_dropped_for_chunk_alignment} valid trajectories "
                f"to align with chunk size {_SAVE_MAX_BATCH_PER_FILE} "
                f"(kept={num_valid_aligned}/{num_valid_raw})."
            )
        if num_valid_aligned <= 0:
            print(
                f"Step {step}: valid trajectories ({num_valid_raw}) are fewer than chunk size "
                f"{_SAVE_MAX_BATCH_PER_FILE}; skipping save."
            )
            continue

        valid_indices = valid_indices[:num_valid_aligned]
        num_valid = int(valid_indices.shape[0])

        rollout_inputs = _filter_batched_tree(rollout_inputs, valid_indices, batch_total)
        perturbed_rollout_params = _filter_batched_tree(perturbed_rollout_params, valid_indices, batch_total)
        if save_original_split:
            test_rollout = _filter_batched_tree(test_rollout, valid_indices, batch_total)
            original_rollout_params = _filter_batched_tree(original_rollout_params, valid_indices, batch_total)
        ee_meta = _summarize_ee_com_meta(
            perturbed_rollout_params,
            mjx_model=mjx_model,
            ee_body_id=ee_body_id,
            ee_body_name=ee_body_name,
        )

        validity_meta = {
            "batch_requested": int(batch_total),
            "batch_valid_raw": int(num_valid_raw),
            "batch_valid": int(num_valid),
            "batch_invalid": int(num_invalid),
            "batch_invalid_rollout": int(num_invalid_rollout),
            "batch_invalid_params": int(num_invalid_params),
            "batch_dropped_for_chunk_alignment": int(num_dropped_for_chunk_alignment),
            "save_max_batch_per_file": int(_SAVE_MAX_BATCH_PER_FILE),
            "validity_q_abs_limit": float(_VALID_Q_ABS_LIMIT),
            "validity_qd_abs_limit": float(_VALID_QD_ABS_LIMIT),
            "validity_u_abs_limit": float(_VALID_U_ABS_LIMIT),
            "valid_trajectory_indices": valid_indices.tolist(),
        }
        force_meta = dict(base_force_meta)
        force_meta.update(
            {
                "external_force_body_id": int(force_body_id),
                "external_force_body_name": force_body_name,
            }
        )
        if sampled_force_target is not None:
            pos_min = sampled_force_target.get("position_min_local_m", None)
            pos_max = sampled_force_target.get("position_max_local_m", None)
            if pos_min is not None and pos_max is not None:
                force_meta.update(
                    {
                        "external_force_position_frame": "body_local",
                        "external_force_position_min_local_m": np.asarray(
                            pos_min, dtype=np.float32
                        ).reshape(3).astype(float).tolist(),
                        "external_force_position_max_local_m": np.asarray(
                            pos_max, dtype=np.float32
                        ).reshape(3).astype(float).tolist(),
                    }
                )

        # -----------------
        # Save perturbed
        # -----------------
        perturbed_dataset_path = os.path.join(
            robot_dataset_base, "perturbed", f"{CUR_TIME_STR}_{step:06d}_perturbed.zarr"
        )
        os.makedirs(os.path.dirname(perturbed_dataset_path), exist_ok=True)
        save_rollout_zarr_v3(
            perturbed_dataset_path,
            rollout=rollout_inputs,
            params=perturbed_rollout_params,
            meta={
                "split": "perturbed",
                "step": int(step),
                "cur_time": CUR_TIME_STR,
                "robot_xml": asset_rel_path,
                **force_meta,
                **ee_meta,
                **validity_meta,
            },
            overwrite=True,
            max_batch_per_file=_SAVE_MAX_BATCH_PER_FILE,
        )
        print(f"Saved perturbed dataset to {perturbed_dataset_path} (valid={num_valid}/{batch_total})")

        if save_original_split:
            # -----------------
            # Save original
            # -----------------
            original_dataset_path = os.path.join(
                robot_dataset_base, "original", f"{CUR_TIME_STR}_{step:06d}_original.zarr"
            )
            os.makedirs(os.path.dirname(original_dataset_path), exist_ok=True)
            save_rollout_zarr_v3(
                original_dataset_path,
                rollout=test_rollout,
                params=original_rollout_params,
                meta={
                    "split": "original",
                    "step": int(step),
                    "cur_time": CUR_TIME_STR,
                    "robot_xml": asset_rel_path,
                    **force_meta,
                    **ee_meta,
                    **validity_meta,
                },
                overwrite=True,
                max_batch_per_file=_SAVE_MAX_BATCH_PER_FILE,
            )
            print(f"Saved original dataset to {original_dataset_path} (valid={num_valid}/{batch_total})")

        print(f"Step {step} took {time.time() - start_time:.2f}s")

        if step % max(1, getattr(cfg, "log_every", 10)) == 0:
            dt = time.time() - t0
            if save_original_split:
                print(f"[{step}/{cfg.num_steps}] saved two samples in {dt:.2f}s")
            else:
                print(f"[{step}/{cfg.num_steps}] saved perturbed sample in {dt:.2f}s")
            t0 = time.time()

    print("Done!")

if __name__ == "__main__":
    cfg = parse_tyro_config(configs.DataConfig)
    main(cfg)
    # quick_check(cfg)
