from __future__ import annotations

import math
import subprocess
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from simadaptor.core import transform_util as tcore


from tests.repo_paths import REPO_ROOT as ROOT


def _sample_rotvec() -> np.ndarray:
    return np.asarray([0.31, -0.22, 0.47], dtype=np.float32)


def _sample_pose() -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray([0.4, -0.2, 0.8], dtype=np.float32)
    quat_wxyz = tcore.matrix_to_quat_wxyz(tcore.rotvec_to_matrix(_sample_rotvec()))
    return pos, np.asarray(quat_wxyz, dtype=np.float32)


def test_quaternion_matrix_round_trip_xyzw_and_wxyz():
    quat_xyzw = tcore.aa2q(_sample_rotvec())
    quat_wxyz = tcore.quat_xyzw_to_wxyz(quat_xyzw)

    rot_xyzw = tcore.quat_xyzw_to_matrix(quat_xyzw)
    rot_wxyz = tcore.quat_wxyz_to_matrix(quat_wxyz)
    np.testing.assert_allclose(rot_xyzw, rot_wxyz, atol=1e-6)

    quat_xyzw_rt = np.asarray(tcore.matrix_to_quat_xyzw(rot_xyzw), dtype=np.float32)
    quat_wxyz_rt = np.asarray(tcore.matrix_to_quat_wxyz(rot_wxyz), dtype=np.float32)
    np.testing.assert_allclose(tcore.quat_xyzw_to_matrix(quat_xyzw_rt), rot_xyzw, atol=1e-6)
    np.testing.assert_allclose(tcore.quat_wxyz_to_matrix(quat_wxyz_rt), rot_wxyz, atol=1e-6)


def test_matrix_to_quat_numpy_input_stays_numpy():
    rot = np.asarray(tcore.rotvec_to_matrix(_sample_rotvec()), dtype=np.float32)

    quat_xyzw = tcore.matrix_to_quat_xyzw(rot)
    quat_wxyz = tcore.matrix_to_quat_wxyz(rot)

    assert isinstance(quat_xyzw, np.ndarray)
    assert isinstance(quat_wxyz, np.ndarray)
    np.testing.assert_allclose(tcore.quat_xyzw_to_matrix(quat_xyzw), rot, atol=1e-6)
    np.testing.assert_allclose(tcore.quat_wxyz_to_matrix(quat_wxyz), rot, atol=1e-6)


def test_quaternion_normalize_multiply_and_apply():
    quat_a = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    quat_b = tcore.matrix_to_quat_wxyz(tcore.rotvec_to_matrix(np.asarray([0.0, 0.0, math.pi / 2.0], dtype=np.float32)))
    quat_c = tcore.quat_mul_wxyz(quat_a, quat_b)
    np.testing.assert_allclose(tcore.normalize_quat_wxyz(quat_c), quat_b, atol=1e-6)

    vec = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    rotated = np.asarray(tcore.quat_apply_wxyz(quat_b, vec), dtype=np.float32)
    np.testing.assert_allclose(rotated, np.asarray([0.0, 1.0, 0.0], dtype=np.float32), atol=1e-5)


def test_axis_angle_and_rotvec_matrix_agree():
    axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    angle = math.pi / 3.0
    from_axis_angle = np.asarray(tcore.axis_angle_to_matrix(axis, angle), dtype=np.float32)
    from_rotvec = np.asarray(tcore.rotvec_to_matrix(axis * angle), dtype=np.float32)
    np.testing.assert_allclose(from_axis_angle, from_rotvec, atol=1e-6)


def test_pose_matrix_round_trip_and_compose_invert():
    pos, quat_wxyz = _sample_pose()
    rot = tcore.quat_wxyz_to_matrix(quat_wxyz)
    H = tcore.pose_wxyz_to_matrix(pos, quat_wxyz)
    pos_rt, quat_rt = tcore.matrix_to_pose_wxyz(H)
    np.testing.assert_allclose(np.asarray(pos_rt, dtype=np.float32), pos, atol=1e-6)
    np.testing.assert_allclose(tcore.quat_wxyz_to_matrix(quat_rt), rot, atol=1e-6)

    child_pos = np.asarray([0.1, 0.0, -0.2], dtype=np.float32)
    child_rot = np.asarray(tcore.rotvec_to_matrix(np.asarray([0.0, 0.2, 0.0], dtype=np.float32)), dtype=np.float32)
    composed_pos, composed_rot = tcore.compose_pose(pos, rot, child_pos, child_rot)
    local_pos, local_rot = tcore.pose_world_to_local(composed_pos, composed_rot, pos, rot)
    np.testing.assert_allclose(np.asarray(local_pos, dtype=np.float32), child_pos, atol=1e-6)
    np.testing.assert_allclose(np.asarray(local_rot, dtype=np.float32), child_rot, atol=1e-6)

    inv_pos, inv_rot = tcore.invert_pose(pos, rot)
    origin_local = np.asarray(tcore.points_world_to_local(pos[None, :], pos, rot)[0], dtype=np.float32)
    np.testing.assert_allclose(origin_local, np.zeros((3,), dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(tcore.points_local_to_world(np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32), inv_pos, inv_rot)[0], dtype=np.float32),
        -pos @ rot,
        atol=1e-6,
    )


def test_numpy_and_jax_parity_for_shared_helpers():
    pos = np.asarray([0.2, 0.3, -0.1], dtype=np.float32)
    rot = np.asarray(tcore.rotvec_to_matrix(_sample_rotvec()), dtype=np.float32)
    pts = np.asarray([[0.1, 0.0, 0.2], [-0.3, 0.5, 0.4]], dtype=np.float32)

    np_world = np.asarray(tcore.points_local_to_world(pts, pos, rot), dtype=np.float32)
    jax_world = np.asarray(
        tcore.points_local_to_world(jnp.asarray(pts), jnp.asarray(pos), jnp.asarray(rot)),
        dtype=np.float32,
    )
    np.testing.assert_allclose(np_world, jax_world, atol=1e-6)

    np_pose = tcore.pose_world_to_local(pos, rot, np.zeros((3,), dtype=np.float32), np.eye(3, dtype=np.float32))
    jax_pose = tcore.pose_world_to_local(
        jnp.asarray(pos),
        jnp.asarray(rot),
        jnp.zeros((3,), dtype=jnp.float32),
        jnp.eye(3, dtype=jnp.float32),
    )
    np.testing.assert_allclose(np.asarray(np_pose[0], dtype=np.float32), np.asarray(jax_pose[0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(np.asarray(np_pose[1], dtype=np.float32), np.asarray(jax_pose[1], dtype=np.float32), atol=1e-6)


def test_points_helpers_broadcast_batched_frames_across_keypoints():
    frame_pos = np.asarray([[0.5, -0.1, 0.2], [-0.3, 0.4, 0.7]], dtype=np.float32)
    frame_rot = np.asarray(
        tcore.rotvec_to_matrix(
            np.asarray([[0.0, 0.0, math.pi / 2.0], [0.1, -0.2, 0.3]], dtype=np.float32)
        ),
        dtype=np.float32,
    )
    local_points = np.asarray(
        [
            [[0.1, 0.0, 0.0], [0.0, 0.2, -0.1], [0.3, -0.4, 0.2]],
            [[-0.2, 0.1, 0.5], [0.4, 0.3, -0.2], [0.0, -0.1, 0.2]],
        ],
        dtype=np.float32,
    )

    world_points_np = np.asarray(tcore.points_local_to_world(local_points, frame_pos, frame_rot), dtype=np.float32)
    round_trip_np = np.asarray(tcore.points_world_to_local(world_points_np, frame_pos, frame_rot), dtype=np.float32)
    np.testing.assert_allclose(round_trip_np, local_points, atol=1e-6)

    world_points_jax = np.asarray(
        tcore.points_local_to_world(jnp.asarray(local_points), jnp.asarray(frame_pos), jnp.asarray(frame_rot)),
        dtype=np.float32,
    )
    round_trip_jax = np.asarray(
        tcore.points_world_to_local(jnp.asarray(world_points_jax), jnp.asarray(frame_pos), jnp.asarray(frame_rot)),
        dtype=np.float32,
    )
    np.testing.assert_allclose(world_points_jax, world_points_np, atol=1e-6)
    np.testing.assert_allclose(round_trip_jax, local_points, atol=1e-6)


def test_removed_local_quaternion_helpers_do_not_reappear():
    pattern = r"def (_normalize_quat_wxyz|_quat_wxyz_to_matrix|_matrix_to_quat_wxyz|_quat_wxyz_from_rotmat)\\b"
    result = subprocess.run(
        ["rg", "-n", pattern, "src", "scripts"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, result.stdout
