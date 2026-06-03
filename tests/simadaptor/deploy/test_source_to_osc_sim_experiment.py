import numpy as np
import pytest
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from simadaptor.deploy import source_to_osc_common as common_exp
from simadaptor.deploy import source_to_osc_sim_experiment as sim_exp


def test_default_sim_reference_uses_sampled_real_style_waypoints() -> None:
    args = sim_exp.build_arg_parser().parse_args(["--dt", "0.01"])
    initial_q = np.asarray(sim_exp.HOME_Q, dtype=np.float64).reshape(7)

    assert args.osc_nullspace_stiffness == sim_exp.DEFAULT_OSC_NULLSPACE_STIFFNESS
    assert args.osc_nullspace_stiffness == common_exp.DEFAULT_OSC_NULLSPACE_STIFFNESS

    ref = sim_exp._randomized_iteration_reference(
        args=args,
        iteration=0,
        initial_q=initial_q,
    )

    assert len(ref.osc_waypoint_xyz) == args.osc_num_waypoints
    assert len(ref.osc_waypoint_rpy_deg) == args.osc_num_waypoints
    assert ref.source_t[-1] > 15.0
    assert ref.target_t[-1] > 7.0
    np.testing.assert_allclose(ref.target_pos[0], ref.osc_start_pos, atol=1e-8)
    np.testing.assert_allclose(
        ref.target_pos[-1],
        ref.osc_start_pos + np.asarray(ref.osc_waypoint_xyz[-1], dtype=np.float64),
        atol=1e-8,
    )
    assert np.max(np.abs(ref.source_q - initial_q[None, :])) > 0.1


def test_float_vector_parser_accepts_non_panda_dof() -> None:
    assert sim_exp.parse_float_vec("0,1,2,3,4,5") == (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
    assert sim_exp.parse_optional_float_vec("auto") is None


def test_vector_defaults_resize_to_robot_dof() -> None:
    values = sim_exp._resize_float_vector((1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0), 6, name="x")
    assert values == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

    cycles = sim_exp._resize_int_vector((3, 4), 6, name="cycles")
    assert cycles == [3, 4, 4, 4, 4, 4]


def test_piper_reference_uses_six_dof(monkeypatch) -> None:
    def fake_fk_site_pose(*, xml_path, q, site_name):
        del xml_path, site_name
        q_arr = np.asarray(q, dtype=np.float64).reshape(-1)
        return np.asarray([0.4 + 0.01 * q_arr[0], 0.0, 0.3], dtype=np.float32), np.asarray(
            [1.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )

    monkeypatch.setattr(sim_exp, "fk_site_pose", fake_fk_site_pose)
    args = sim_exp.build_arg_parser().parse_args(
        [
            "--robot-preset",
            "piper",
            "--dt",
            "0.01",
            "--num-iterations",
            "1",
        ]
    )
    args.xml = sim_exp.DEFAULT_PIPER_XML
    args.initial_q = sim_exp.DEFAULT_PIPER_HOME_Q
    args.source_amp_deg = sim_exp.DEFAULT_PIPER_SOURCE_AMP_DEG
    args.source_cycles = list(sim_exp.DEFAULT_PIPER_SOURCE_CYCLES)
    initial_q = np.asarray(args.initial_q, dtype=np.float64).reshape(6)

    ref = sim_exp._randomized_iteration_reference(
        args=args,
        iteration=0,
        initial_q=initial_q,
    )

    assert ref.source_q.shape[1] == 6
    assert len(ref.source_amp_deg) == 6
    assert len(ref.source_cycles) == 6


def test_rby1_reference_uses_one_arm_seven_dof(monkeypatch) -> None:
    def fake_fk_site_pose(*, xml_path, q, site_name):
        del xml_path, site_name
        q_arr = np.asarray(q, dtype=np.float64).reshape(-1)
        return np.asarray([0.25, 0.1 + 0.01 * q_arr[1], 0.9], dtype=np.float32), np.asarray(
            [1.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )

    monkeypatch.setattr(sim_exp, "fk_site_pose", fake_fk_site_pose)
    args = sim_exp.build_arg_parser().parse_args(
        [
            "--robot-preset",
            "rby1",
            "--dt",
            "0.01",
            "--num-iterations",
            "1",
        ]
    )
    args.xml = sim_exp.DEFAULT_RBY1_ONEARM_XML
    args.profile_key = "rby1_onearm"
    args.initial_q = sim_exp.DEFAULT_RBY1_ONEARM_HOME_Q
    args.source_amp_deg = sim_exp.DEFAULT_RBY1_ONEARM_SOURCE_AMP_DEG
    args.source_cycles = list(sim_exp.DEFAULT_RBY1_ONEARM_SOURCE_CYCLES)
    initial_q = np.asarray(args.initial_q, dtype=np.float64).reshape(7)

    ref = sim_exp._randomized_iteration_reference(
        args=args,
        iteration=0,
        initial_q=initial_q,
    )

    assert args.profile_key == "rby1_onearm"
    assert ref.source_q.shape[1] == 7
    assert len(ref.source_amp_deg) == 7
    assert len(ref.source_cycles) == 7


@pytest.mark.parametrize(
    ("preset", "expected_dof", "profile_key"),
    [
        ("iiwa14", 7, "iiwa14"),
        ("google_robot", 7, "google_robot"),
        ("unitree_z1", 6, "unitree_z1"),
        ("flexiv_rizon4", 7, "flexiv_rizon4"),
    ],
)
def test_menagerie_arm_presets_are_arm_only_assets(
    preset: str,
    expected_dof: int,
    profile_key: str,
) -> None:
    args = sim_exp.build_arg_parser().parse_args(["--robot-preset", preset])
    dof = sim_exp._resolve_robot_args(args)

    assert dof == expected_dof
    assert args.profile_key == profile_key
    assert Path(args.xml).exists()

    model = sim_exp.mujoco.MjModel.from_xml_path(str(args.xml))
    assert int(model.nu) == expected_dof
    assert int(model.njnt) == expected_dof
    assert sim_exp.mujoco.mj_name2id(model, sim_exp.mujoco.mjtObj.mjOBJ_SITE, "gripper") >= 0
    assert sim_exp.mujoco.mj_name2id(model, sim_exp.mujoco.mjtObj.mjOBJ_BODY, "hand") >= 0

    root = ET.parse(args.xml).getroot()
    assert not root.findall(".//default[@class='collision']")
    assert not root.findall(".//geom[@class='collision']")
    assert not root.findall(".//contact/*")
    assert all("gear" not in elem.attrib for elem in root.iter())
    assert all(
        "gripper" not in (joint.attrib.get("name", "").lower())
        and "finger" not in (joint.attrib.get("name", "").lower())
        for joint in root.findall(".//worldbody//joint")
    )
    assert all(
        "gripper" not in " ".join(act.attrib.values()).lower()
        and "finger" not in " ".join(act.attrib.values()).lower()
        for act in root.findall(".//actuator/*")
    )

    with open(sim_exp.DEFAULT_PROFILE_TABLE, "r", encoding="utf-8") as f:
        profile = json.load(f)[profile_key]
    for key in (
        "armature_min_profile",
        "armature_max_profile",
        "base_kp_profile",
        "white_base_profile",
        "walk_base_profile",
        "waypoint_max_delta_deg_profile",
    ):
        assert len(profile[key]) == expected_dof


def test_explicit_sim_nullspace_stiffness_overrides_real_default() -> None:
    args = sim_exp.build_arg_parser().parse_args(
        ["--dt", "0.01", "--osc-nullspace-stiffness", "10"]
    )

    assert args.osc_nullspace_stiffness == 10.0


def test_sim_backend_auto_uses_batched_for_table_conditions() -> None:
    args = sim_exp.build_arg_parser().parse_args(
        ["--conditions", "direct_osc", "tam_carried"]
    )
    conditions = sim_exp.resolve_sim_conditions(args.conditions)

    assert sim_exp._sim_backend_choice(args, conditions) == "batched"


def test_sim_backend_auto_keeps_legacy_for_non_table_conditions() -> None:
    args = sim_exp.build_arg_parser().parse_args(
        ["--conditions", "direct_osc", "tam_reset", "tam_carried"]
    )
    conditions = sim_exp.resolve_sim_conditions(args.conditions)

    assert sim_exp._sim_backend_choice(args, conditions) == "legacy"


def test_sim_backend_override_can_force_legacy_for_table_conditions() -> None:
    args = sim_exp.build_arg_parser().parse_args(
        ["--conditions", "direct_osc", "tam_carried", "--sim-backend", "legacy"]
    )
    conditions = sim_exp.resolve_sim_conditions(args.conditions)

    assert sim_exp._sim_backend_choice(args, conditions) == "legacy"


def test_batched_eval_batch_size_default_means_single_batch() -> None:
    args = sim_exp.build_arg_parser().parse_args([])

    assert args.batched_eval_batch_size == 0


def test_non_tam_conditions_are_not_public() -> None:
    with pytest.raises(SystemExit):
        sim_exp.resolve_sim_conditions(["legacy_baseline"])


def test_source_only_flag_is_parsed() -> None:
    args = sim_exp.build_arg_parser().parse_args(["--source-only"])

    assert args.source_only


def test_controller_side_guard_defaults_to_datagen_enabled() -> None:
    args = sim_exp.build_arg_parser().parse_args([])

    assert args.controller_side_guard
    assert args.controller_guard_velocity_threshold == pytest.approx(4.0)


def test_controller_side_guard_torque_matches_datagen_rule() -> None:
    q = np.asarray([-1.2, 0.0, 1.3], dtype=np.float64)
    dq = np.asarray([0.0, 5.5, -4.5], dtype=np.float64)
    kp = np.asarray([10.0, 20.0, 30.0], dtype=np.float64)
    kd = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    joint_range = np.asarray([[-1.0, 1.0], [-0.5, 0.5], [-1.0, 1.0]], dtype=np.float64)

    tau = sim_exp._controller_side_guard_torque(
        q=q,
        dq=dq,
        kp=kp,
        kd=kd,
        joint_range=joint_range,
    )

    np.testing.assert_allclose(
        tau,
        np.asarray(
            [
                10.0 * 20.0 * 0.2,
                -1.0 * 2.0 * 10.0 * (5.5 - 4.0),
                30.0 * 20.0 * -0.3 + 3.0 * 10.0 * (4.5 - 4.0),
            ],
            dtype=np.float64,
        ),
    )


def test_explicit_sim_waypoints_require_xyz_and_rpy_pairs() -> None:
    args = sim_exp.build_arg_parser().parse_args(
        [
            "--dt",
            "0.01",
            "--osc-waypoint-xyz",
            "0.02,0.0,0.0",
        ]
    )
    initial_q = np.asarray(sim_exp.HOME_Q, dtype=np.float64).reshape(7)

    with pytest.raises(ValueError, match="must be provided together"):
        sim_exp._randomized_iteration_reference(
            args=args,
            iteration=0,
            initial_q=initial_q,
        )


def test_trajectory_error_metrics_aligns_by_time_not_index() -> None:
    actual_t = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    ideal_t = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
    actual_pos = np.stack([actual_t, 2.0 * actual_t, -actual_t], axis=1)
    ideal_pos = np.stack([ideal_t, 2.0 * ideal_t, -ideal_t], axis=1)

    metrics = sim_exp._trajectory_error_metrics(
        target_log_np={"t": actual_t, "ee_pos": actual_pos},
        ideal_log_np={"t": ideal_t, "ee_pos": ideal_pos},
    )

    assert metrics["target_ideal_ee_samples"] == ideal_t.shape[0]
    assert metrics["target_vs_ideal_ee_pos_samples"] == actual_t.shape[0]
    assert metrics["target_vs_ideal_ee_pos_rmse_m"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["target_vs_ideal_ee_final_error_m"] == pytest.approx(0.0, abs=1e-12)


def test_trajectory_error_metrics_returns_nan_for_missing_logs() -> None:
    metrics = sim_exp._trajectory_error_metrics(
        target_log_np={"t": np.array([0.0]), "ee_pos": np.zeros((1, 3))},
        ideal_log_np={"t": np.array([0.0]), "ee_pos": np.zeros((1, 3))},
    )

    assert metrics["target_ideal_ee_samples"] == 1
    assert np.isnan(metrics["target_vs_ideal_ee_pos_rmse_m"])


def test_source_vs_ideal_metrics_aligns_joint_and_ee_by_time() -> None:
    actual_t = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    ideal_t = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
    ideal_q = np.stack([ideal_t, 2.0 * ideal_t], axis=1)
    actual_q = np.stack([actual_t, 2.0 * actual_t], axis=1) + 0.1
    ideal_ee = np.stack([ideal_t, 0.0 * ideal_t, -ideal_t], axis=1)
    actual_ee = np.stack([actual_t, 0.0 * actual_t, -ideal_t[::2]], axis=1)
    actual_ee = actual_ee + np.array([0.0, 0.003, 0.004])

    metrics = sim_exp._source_vs_ideal_metrics(
        source_log_np={"t": actual_t, "q": actual_q, "ee_pos": actual_ee},
        ideal_source_log_np={"t": ideal_t, "q": ideal_q, "ee_pos": ideal_ee},
    )

    assert metrics["source_vs_ideal_joint_pos_samples"] == actual_t.shape[0]
    assert metrics["source_vs_ideal_joint_pos_rmse_rad"] == pytest.approx(0.1)
    assert metrics["source_vs_ideal_joint_pos_rmse_deg"] == pytest.approx(np.rad2deg(0.1))
    assert metrics["source_vs_ideal_ee_pos_rmse_m"] == pytest.approx(0.005)


def test_write_aggregate_files_uses_sample_std_and_mm_summary(tmp_path) -> None:
    rows = [
        {
            "condition_key": "tam_carried",
            "condition": "TAM carried",
            "target_vs_ideal_ee_pos_rmse_m": value,
            "source_vs_ideal_ee_pos_rmse_m": value * 2.0,
            "source_vs_ideal_joint_pos_rmse_deg": value * 1000.0,
            "target_ee_pos_rmse_m": value,
        }
        for value in (0.001, 0.003)
    ]

    sim_exp._write_aggregate_files(tmp_path, rows)

    aggregate = json.loads((tmp_path / "summary_aggregate.json").read_text())
    row = aggregate[0]
    assert row["target_vs_ideal_ee_pos_rmse_m_n"] == 2
    assert row["target_vs_ideal_ee_pos_rmse_m_mean"] == pytest.approx(0.002)
    assert row["target_vs_ideal_ee_pos_rmse_m_std"] == pytest.approx(np.sqrt(2.0e-6))
    assert row["target_vs_ideal_ee_pos_rmse_mm_pm"] == "2.00 +/- 1.41"
    assert row["source_vs_ideal_ee_pos_rmse_mm_pm"] == "4.00 +/- 2.83"
    assert row["source_vs_ideal_joint_pos_rmse_deg_pm"] == "2.00 +/- 1.41"
