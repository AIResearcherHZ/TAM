from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np


DATAGEN_VECTOR_KEYS = {
    "armature_min_profile",
    "armature_max_profile",
    "base_kp_profile",
    "white_base_profile",
    "walk_base_profile",
    "waypoint_max_delta_deg_profile",
    "ee_payload_com_offset_min_local_m",
    "ee_payload_com_offset_max_local_m",
    "external_force_position_min_local_m",
    "external_force_position_max_local_m",
}
DATAGEN_VECTOR3_KEYS = {
    "ee_payload_com_offset_min_local_m",
    "ee_payload_com_offset_max_local_m",
    "external_force_position_min_local_m",
    "external_force_position_max_local_m",
}
DATAGEN_RANGE_KEYS = {
    "white_scale_range",
    "walk_scale_range",
    "kp_scale_small_range",
    "kp_scale_large_range",
    "ee_payload_mass_delta_range",
}
DATAGEN_SCALAR_KEYS = {
    "kp_small_prob",
    "joint_model_major_ee_scale",
    "joint_model_major_global_scale",
    "rollout_cmd_noise_std",
    "external_force_magnitude_min_n",
    "external_force_magnitude_max_n",
}
DATAGEN_FORCE_TARGET_KEYS = {
    "external_force_body_names",
    "external_force_position_boxes_local_m_by_body_name",
}
DATAGEN_ALLOWED_KEYS = (
    DATAGEN_VECTOR_KEYS | DATAGEN_RANGE_KEYS | DATAGEN_SCALAR_KEYS | DATAGEN_FORCE_TARGET_KEYS
)


def _sanitize_robot_stem(stem: str) -> str:
    safe = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in stem)
    return safe if safe else "robot"


def derive_robot_key(xml_path: str | Path) -> str:
    xml_path = Path(xml_path)
    return _sanitize_robot_stem(xml_path.stem)


def _normalize_vector3_field(profile_key: str, field_name: str, values: Any) -> jnp.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if int(arr.size) != 3:
        raise ValueError(
            f"Datagen profile '{profile_key}' field '{field_name}' must be length 3, got {arr!r}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            f"Datagen profile '{profile_key}' field '{field_name}' must contain only finite values, got {arr!r}."
        )
    return jnp.asarray(arr, dtype=jnp.float32)


def _normalize_force_targets(
    profile: Dict[str, Any],
    profile_key: str,
) -> Dict[str, Any]:
    names_raw = profile.get("external_force_body_names", None)
    boxes_raw = profile.get("external_force_position_boxes_local_m_by_body_name", None)
    if names_raw is None and boxes_raw is None:
        return {}
    if names_raw is None or boxes_raw is None:
        raise ValueError(
            f"Datagen profile '{profile_key}' must define both "
            "'external_force_body_names' and "
            "'external_force_position_boxes_local_m_by_body_name' together."
        )
    if not isinstance(names_raw, (list, tuple)) or len(names_raw) == 0:
        raise ValueError(
            f"Datagen profile '{profile_key}' field 'external_force_body_names' "
            f"must be a non-empty list/tuple of strings, got {names_raw!r}."
        )
    if not isinstance(boxes_raw, dict):
        raise ValueError(
            f"Datagen profile '{profile_key}' field "
            "'external_force_position_boxes_local_m_by_body_name' must be a JSON object, "
            f"got {type(boxes_raw)}."
        )

    body_names: List[str] = []
    seen_names = set()
    for idx, raw_name in enumerate(names_raw):
        if not isinstance(raw_name, str):
            raise ValueError(
                f"Datagen profile '{profile_key}' field 'external_force_body_names[{idx}]' "
                f"must be a string, got {type(raw_name)}."
            )
        body_name = raw_name.strip()
        if not body_name:
            raise ValueError(
                f"Datagen profile '{profile_key}' field 'external_force_body_names[{idx}]' "
                "must be non-empty."
            )
        if body_name in seen_names:
            raise ValueError(
                f"Datagen profile '{profile_key}' field 'external_force_body_names' "
                f"contains duplicate body name '{body_name}'."
            )
        seen_names.add(body_name)
        body_names.append(body_name)

    extra_box_names = sorted(set(str(k) for k in boxes_raw.keys()) - set(body_names))
    if extra_box_names:
        raise ValueError(
            f"Datagen profile '{profile_key}' field "
            "'external_force_position_boxes_local_m_by_body_name' contains entries for "
            f"bodies not listed in 'external_force_body_names': {extra_box_names}."
        )

    normalized_boxes: Dict[str, Dict[str, jnp.ndarray]] = {}
    normalized_targets: List[Dict[str, Any]] = []
    for body_name in body_names:
        if body_name not in boxes_raw:
            raise ValueError(
                f"Datagen profile '{profile_key}' is missing an external-force box entry for "
                f"body '{body_name}'."
            )
        raw_box = boxes_raw[body_name]
        if not isinstance(raw_box, dict):
            raise ValueError(
                f"Datagen profile '{profile_key}' external-force box for body '{body_name}' "
                f"must be a JSON object with 'min'/'max', got {type(raw_box)}."
            )
        if "min" not in raw_box or "max" not in raw_box:
            raise ValueError(
                f"Datagen profile '{profile_key}' external-force box for body '{body_name}' "
                "must define both 'min' and 'max'."
            )
        pos_min = _normalize_vector3_field(
            profile_key,
            f"external_force_position_boxes_local_m_by_body_name[{body_name!r}].min",
            raw_box["min"],
        )
        pos_max = _normalize_vector3_field(
            profile_key,
            f"external_force_position_boxes_local_m_by_body_name[{body_name!r}].max",
            raw_box["max"],
        )
        pos_min, pos_max = jnp.minimum(pos_min, pos_max), jnp.maximum(pos_min, pos_max)
        normalized_box = {"min": pos_min, "max": pos_max}
        normalized_boxes[body_name] = normalized_box
        normalized_targets.append(
            {
                "body_name": body_name,
                "position_min_local_m": pos_min,
                "position_max_local_m": pos_max,
            }
        )

    return {
        "external_force_body_names": list(body_names),
        "external_force_position_boxes_local_m_by_body_name": normalized_boxes,
        "external_force_targets": normalized_targets,
    }


def normalize_datagen_profile(profile: Dict[str, Any], profile_key: str) -> Dict[str, Any]:
    unknown_keys = sorted(set(profile.keys()) - DATAGEN_ALLOWED_KEYS)
    if unknown_keys:
        raise ValueError(
            f"Unknown datagen profile keys for '{profile_key}': {unknown_keys}. "
            f"Allowed keys: {sorted(DATAGEN_ALLOWED_KEYS)}"
        )

    out: Dict[str, Any] = {}
    for k in DATAGEN_VECTOR_KEYS:
        if k not in profile or profile[k] is None:
            continue
        arr = np.asarray(profile[k], dtype=np.float32).reshape(-1)
        if arr.size == 0:
            raise ValueError(f"Datagen profile '{profile_key}' has empty vector for '{k}'.")
        if not np.all(np.isfinite(arr)):
            raise ValueError(
                f"Datagen profile '{profile_key}' field '{k}' must contain only finite values, got {arr!r}."
            )
        if k in DATAGEN_VECTOR3_KEYS:
            if int(arr.size) != 3:
                raise ValueError(
                    f"Datagen profile '{profile_key}' field '{k}' must be length 3, got {arr!r}."
                )
            out[k] = jnp.asarray(arr, dtype=jnp.float32)
            continue
        out[k] = jnp.asarray(arr, dtype=jnp.float32)

    if (
        "ee_payload_com_offset_min_local_m" in out
        or "ee_payload_com_offset_max_local_m" in out
    ):
        if (
            "ee_payload_com_offset_min_local_m" not in out
            or "ee_payload_com_offset_max_local_m" not in out
        ):
            raise ValueError(
                f"Datagen profile '{profile_key}' must define both "
                "'ee_payload_com_offset_min_local_m' and 'ee_payload_com_offset_max_local_m' together."
            )
        pos_min = jnp.minimum(
            out["ee_payload_com_offset_min_local_m"],
            out["ee_payload_com_offset_max_local_m"],
        )
        pos_max = jnp.maximum(
            out["ee_payload_com_offset_min_local_m"],
            out["ee_payload_com_offset_max_local_m"],
        )
        out["ee_payload_com_offset_min_local_m"] = pos_min
        out["ee_payload_com_offset_max_local_m"] = pos_max

    if (
        "external_force_position_min_local_m" in out
        or "external_force_position_max_local_m" in out
    ):
        if (
            "external_force_position_min_local_m" not in out
            or "external_force_position_max_local_m" not in out
        ):
            raise ValueError(
                f"Datagen profile '{profile_key}' must define both "
                "'external_force_position_min_local_m' and 'external_force_position_max_local_m' together."
            )
        pos_min = jnp.minimum(
            out["external_force_position_min_local_m"],
            out["external_force_position_max_local_m"],
        )
        pos_max = jnp.maximum(
            out["external_force_position_min_local_m"],
            out["external_force_position_max_local_m"],
        )
        out["external_force_position_min_local_m"] = pos_min
        out["external_force_position_max_local_m"] = pos_max

    for k in DATAGEN_RANGE_KEYS:
        if k not in profile or profile[k] is None:
            continue
        vals = profile[k]
        if not isinstance(vals, (list, tuple)) or len(vals) != 2:
            raise ValueError(
                f"Datagen profile '{profile_key}' field '{k}' must be length-2 list/tuple, got {vals!r}."
            )
        out[k] = (float(vals[0]), float(vals[1]))

    for k in DATAGEN_SCALAR_KEYS:
        if k not in profile or profile[k] is None:
            continue
        out[k] = float(profile[k])

    out.update(_normalize_force_targets(profile, profile_key))

    return out


def load_datagen_profile(
    table_path: str | Path,
    robot_key: str,
    profile_key: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    table_path = Path(table_path).expanduser()
    if not table_path.exists():
        raise FileNotFoundError(
            f"Datagen profile table not found: {table_path}. "
            "Set datagen profile table path to a valid JSON file."
        )
    with open(table_path, "r") as f:
        table = json.load(f)
    if not isinstance(table, dict):
        raise ValueError(f"Datagen profile table must be a JSON object: {table_path}")

    resolved_profile_key = profile_key or robot_key
    profile_raw = table.get(resolved_profile_key)
    if profile_raw is None:
        raise KeyError(
            f"Datagen profile key '{resolved_profile_key}' not found in {table_path}. "
            f"Available keys: {sorted(table.keys())}"
        )
    if not isinstance(profile_raw, dict):
        raise ValueError(
            f"Datagen profile '{resolved_profile_key}' must map to a JSON object, "
            f"got {type(profile_raw)}."
        )

    return resolved_profile_key, normalize_datagen_profile(profile_raw, resolved_profile_key)
