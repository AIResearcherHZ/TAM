from pathlib import Path
import importlib.util
import sys

import numpy as np
import pytest

from tests.repo_paths import REPO_ROOT as ROOT
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

MODULE_PATH = SRC_ROOT / "simadaptor" / "data" / "datagen_profiles.py"
MODULE_SPEC = importlib.util.spec_from_file_location("test_datagen_profiles_module", MODULE_PATH)
datagen_profiles = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
MODULE_SPEC.loader.exec_module(datagen_profiles)
normalize_datagen_profile = datagen_profiles.normalize_datagen_profile


def _base_profile() -> dict:
    return {
        "armature_min_profile": [0.01, 0.01],
        "armature_max_profile": [0.1, 0.1],
        "base_kp_profile": [10.0, 10.0],
        "white_base_profile": [1.0, 1.0],
        "walk_base_profile": [0.1, 0.1],
        "waypoint_max_delta_deg_profile": [30.0, 30.0],
        "white_scale_range": [0.0, 1.0],
        "walk_scale_range": [0.0, 1.0],
        "kp_scale_small_range": [0.1, 0.2],
        "kp_scale_large_range": [1.0, 2.0],
        "kp_small_prob": 0.5,
    }


def test_normalize_datagen_profile_accepts_force_position_bounds():
    profile = _base_profile()
    profile["external_force_position_min_local_m"] = [0.02, -0.03, 0.11]
    profile["external_force_position_max_local_m"] = [0.07, 0.04, 0.15]

    normalized = normalize_datagen_profile(profile, "robot")

    np.testing.assert_allclose(
        np.asarray(normalized["external_force_position_min_local_m"]),
        np.asarray([0.02, -0.03, 0.11], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(normalized["external_force_position_max_local_m"]),
        np.asarray([0.07, 0.04, 0.15], dtype=np.float32),
    )


def test_normalize_datagen_profile_accepts_force_target_boxes():
    profile = _base_profile()
    profile["external_force_body_names"] = ["wrist", "hand"]
    profile["external_force_position_boxes_local_m_by_body_name"] = {
        "wrist": {"min": [0.03, 0.02, 0.11], "max": [-0.04, 0.05, 0.09]},
        "hand": {"min": [-0.05, -0.05, 0.01], "max": [0.05, 0.05, 0.11]},
    }

    normalized = normalize_datagen_profile(profile, "robot")

    assert normalized["external_force_body_names"] == ["wrist", "hand"]
    assert [target["body_name"] for target in normalized["external_force_targets"]] == ["wrist", "hand"]
    np.testing.assert_allclose(
        np.asarray(
            normalized["external_force_position_boxes_local_m_by_body_name"]["wrist"]["min"]
        ),
        np.asarray([-0.04, 0.02, 0.09], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(
            normalized["external_force_position_boxes_local_m_by_body_name"]["wrist"]["max"]
        ),
        np.asarray([0.03, 0.05, 0.11], dtype=np.float32),
    )


def test_normalize_datagen_profile_accepts_joint_major_global_scale():
    profile = _base_profile()
    profile["joint_model_major_global_scale"] = 0.02

    normalized = normalize_datagen_profile(profile, "robot")

    assert normalized["joint_model_major_global_scale"] == pytest.approx(0.02)


def test_normalize_datagen_profile_reorders_force_position_bounds_per_axis():
    profile = _base_profile()
    profile["external_force_position_min_local_m"] = [0.07, 0.04, 0.15]
    profile["external_force_position_max_local_m"] = [0.02, -0.03, 0.11]

    normalized = normalize_datagen_profile(profile, "robot")

    np.testing.assert_allclose(
        np.asarray(normalized["external_force_position_min_local_m"]),
        np.asarray([0.02, -0.03, 0.11], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(normalized["external_force_position_max_local_m"]),
        np.asarray([0.07, 0.04, 0.15], dtype=np.float32),
    )


def test_normalize_datagen_profile_rejects_invalid_force_position_length():
    profile = _base_profile()
    profile["external_force_position_min_local_m"] = [0.0, 0.1]
    profile["external_force_position_max_local_m"] = [0.0, 0.1, 0.2]

    with pytest.raises(ValueError, match="must be length 3"):
        normalize_datagen_profile(profile, "robot")


def test_normalize_datagen_profile_rejects_nonfinite_force_position_bounds():
    profile = _base_profile()
    profile["external_force_position_min_local_m"] = [0.0, np.nan, 0.1]
    profile["external_force_position_max_local_m"] = [0.0, 0.1, 0.2]

    with pytest.raises(ValueError, match="must contain only finite values"):
        normalize_datagen_profile(profile, "robot")


def test_normalize_datagen_profile_rejects_missing_force_target_box():
    profile = _base_profile()
    profile["external_force_body_names"] = ["hand", "wrist"]
    profile["external_force_position_boxes_local_m_by_body_name"] = {
        "hand": {"min": [-0.05, -0.05, 0.01], "max": [0.05, 0.05, 0.11]},
    }

    with pytest.raises(ValueError, match="missing an external-force box entry for body 'wrist'"):
        normalize_datagen_profile(profile, "robot")


def test_normalize_datagen_profile_rejects_duplicate_force_target_body_names():
    profile = _base_profile()
    profile["external_force_body_names"] = ["hand", "hand"]
    profile["external_force_position_boxes_local_m_by_body_name"] = {
        "hand": {"min": [-0.05, -0.05, 0.01], "max": [0.05, 0.05, 0.11]},
    }

    with pytest.raises(ValueError, match="duplicate body name 'hand'"):
        normalize_datagen_profile(profile, "robot")
