from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np


DEFAULT_OSC_WAYPOINT_XYZ_MIN = (0.05, -0.18, -0.40)
DEFAULT_OSC_WAYPOINT_XYZ_MAX = (0.42, 0.18, 0.14)
DEFAULT_OSC_WAYPOINT_RPY_DEG_MIN = (-40.0, -35.0, -60.0)
DEFAULT_OSC_WAYPOINT_RPY_DEG_MAX = (40.0, 35.0, 60.0)
DEFAULT_OSC_NULLSPACE_STIFFNESS = 0.5
HISTORY_CONTROLLER_PSEUDOINVERSE_DAMPING = 0.2
HOME_Q = np.array(
    [
        0.0,
        -0.7853981633974483,
        0.0,
        -2.356194490192345,
        0.0,
        1.5707963267948966,
        0.7853981633974483,
    ],
    dtype=np.float64,
)


def _parse_csv_vec(raw: str, n: int, *, name: str) -> tuple[float, ...]:
    vals = [float(part) for part in str(raw).split(",") if part.strip() != ""]
    if len(vals) != n:
        raise argparse.ArgumentTypeError(
            f"{name} must contain {n} comma-separated floats; got {len(vals)}"
        )
    return tuple(vals)


def parse_vec3(raw: str) -> tuple[float, float, float]:
    return _parse_csv_vec(raw, 3, name="vec3")  # type: ignore[return-value]


def parse_vec6(raw: str) -> tuple[float, ...]:
    return _parse_csv_vec(raw, 6, name="vec6")


def sample_osc_waypoints(
    *,
    rng: np.random.Generator,
    num_waypoints: int,
    xyz_min: Sequence[float],
    xyz_max: Sequence[float],
    rpy_deg_min: Sequence[float],
    rpy_deg_max: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    count = int(num_waypoints)
    if count < 1:
        raise ValueError(f"num_waypoints must be positive, got {count}")
    xyz_lo = np.asarray(xyz_min, dtype=np.float64).reshape(3)
    xyz_hi = np.asarray(xyz_max, dtype=np.float64).reshape(3)
    rpy_lo = np.asarray(rpy_deg_min, dtype=np.float64).reshape(3)
    rpy_hi = np.asarray(rpy_deg_max, dtype=np.float64).reshape(3)
    xyz = rng.uniform(np.minimum(xyz_lo, xyz_hi), np.maximum(xyz_lo, xyz_hi), size=(count, 3))
    rpy = rng.uniform(np.minimum(rpy_lo, rpy_hi), np.maximum(rpy_lo, rpy_hi), size=(count, 3))
    return xyz.astype(np.float64), rpy.astype(np.float64)


def make_source_reference(
    *,
    start_q: Sequence[float],
    duration_s: float,
    dt_s: float,
    amp_deg: Sequence[float],
    cycles: Sequence[int],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start = np.asarray(start_q, dtype=np.float64).reshape(-1)
    dof = int(start.shape[0])
    if dof <= 0:
        raise ValueError("start_q must be non-empty.")
    duration_s = float(duration_s)
    dt_s = float(dt_s)
    if duration_s <= 0.0:
        raise ValueError(f"duration_s must be positive, got {duration_s}")
    if dt_s <= 0.0:
        raise ValueError(f"dt_s must be positive, got {dt_s}")

    n_steps = int(round(duration_s / dt_s)) + 1
    times = np.linspace(0.0, duration_s, n_steps, dtype=np.float64)
    phase_t = times / duration_s

    amp_raw = np.asarray(amp_deg, dtype=np.float64).reshape(-1)
    if amp_raw.size == 1:
        amp_raw = np.repeat(amp_raw, dof)
    if amp_raw.size < dof:
        amp_raw = np.pad(amp_raw, (0, dof - amp_raw.size), constant_values=float(amp_raw[-1]))
    amp = np.deg2rad(amp_raw[:dof])

    cycle_raw = np.asarray(cycles, dtype=np.int64).reshape(-1)
    if cycle_raw.size == 1:
        cycle_raw = np.repeat(cycle_raw, dof)
    if cycle_raw.size < dof:
        cycle_raw = np.pad(cycle_raw, (0, dof - cycle_raw.size), constant_values=int(cycle_raw[-1]))
    cycle_raw = np.maximum(cycle_raw[:dof], 1)

    rng = np.random.default_rng(int(seed))
    sign = rng.choice(np.asarray([-1.0, 1.0]), size=dof)
    phase_offset = rng.uniform(0.0, 2.0 * np.pi, size=dof)

    envelope = np.sin(np.pi * phase_t) ** 2
    envelope_dot = (np.pi / duration_s) * np.sin(2.0 * np.pi * phase_t)
    osc_phase = 2.0 * np.pi * phase_t[:, None] * cycle_raw[None, :] + phase_offset[None, :]
    osc_dot = (2.0 * np.pi / duration_s) * cycle_raw[None, :]
    deviation = sign[None, :] * amp[None, :] * envelope[:, None] * np.sin(osc_phase)
    deviation_dot = sign[None, :] * amp[None, :] * (
        envelope_dot[:, None] * np.sin(osc_phase)
        + envelope[:, None] * np.cos(osc_phase) * osc_dot
    )

    q_ref = start[None, :] + deviation
    dq_ref = deviation_dot
    q_ref[0] = start
    q_ref[-1] = start
    dq_ref[0] = 0.0
    dq_ref[-1] = 0.0
    return times, q_ref.astype(np.float32), dq_ref.astype(np.float32)


def make_cartesian_target_reference(
    *,
    start_pos: Sequence[float],
    start_quat_wxyz: Sequence[float],
    duration_s: float,
    dt_s: float,
    delta_xyz: Sequence[float],
    delta_rpy_deg: Sequence[float] = (0.0, 0.0, 0.0),
    waypoint_xyz: Optional[Sequence[Sequence[float]]] = None,
    waypoint_rpy_deg: Optional[Sequence[Sequence[float]]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    duration_s = float(duration_s)
    dt_s = float(dt_s)
    if duration_s <= 0.0:
        raise ValueError(f"duration_s must be positive, got {duration_s}")
    if dt_s <= 0.0:
        raise ValueError(f"dt_s must be positive, got {dt_s}")
    start_pos_arr = np.asarray(start_pos, dtype=np.float64).reshape(3)
    quat = np.asarray(start_quat_wxyz, dtype=np.float64).reshape(4)
    quat_norm = float(np.linalg.norm(quat))
    quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64) if quat_norm <= 1e-12 else quat / quat_norm

    waypoint_xyz_arr = (
        np.asarray([delta_xyz], dtype=np.float64).reshape(-1, 3)
        if waypoint_xyz is None
        else np.asarray(waypoint_xyz, dtype=np.float64).reshape(-1, 3)
    )
    waypoint_rpy_arr = (
        np.asarray([delta_rpy_deg], dtype=np.float64).reshape(-1, 3)
        if waypoint_rpy_deg is None
        else np.asarray(waypoint_rpy_deg, dtype=np.float64).reshape(-1, 3)
    )
    if waypoint_xyz_arr.shape[0] < 1:
        raise ValueError("waypoint_xyz must contain at least one waypoint")
    if waypoint_rpy_arr.shape[0] != waypoint_xyz_arr.shape[0]:
        raise ValueError("waypoint_rpy_deg must contain the same number of waypoints as waypoint_xyz")

    n_steps = int(round(duration_s / dt_s)) + 1
    times = np.linspace(0.0, duration_s, n_steps, dtype=np.float64)
    offset_waypoints = np.vstack([np.zeros((1, 3), dtype=np.float64), waypoint_xyz_arr])
    rpy_waypoints = np.vstack([np.zeros((1, 3), dtype=np.float64), waypoint_rpy_arr])
    knot_times = np.linspace(0.0, duration_s, offset_waypoints.shape[0], dtype=np.float64)

    pos = start_pos_arr[None, :] + _sample_clamped_cubic_spline(
        knot_times=knot_times,
        values=offset_waypoints,
        sample_times=times,
    )
    rpy_ref_deg = _sample_clamped_cubic_spline(
        knot_times=knot_times,
        values=rpy_waypoints,
        sample_times=times,
    )
    quat_ref = np.empty((n_steps, 4), dtype=np.float64)
    for i, rpy_deg in enumerate(rpy_ref_deg):
        quat_ref[i] = _quat_wxyz_mul(quat, _quat_wxyz_from_rpy(np.deg2rad(rpy_deg)))
    pos[0] = start_pos_arr
    pos[-1] = start_pos_arr + waypoint_xyz_arr[-1]
    quat_ref[0] = quat
    quat_ref[-1] = _quat_wxyz_mul(quat, _quat_wxyz_from_rpy(np.deg2rad(waypoint_rpy_arr[-1])))
    return times, pos.astype(np.float32), quat_ref.astype(np.float32)


def _sample_clamped_cubic_spline(
    *,
    knot_times: np.ndarray,
    values: np.ndarray,
    sample_times: np.ndarray,
) -> np.ndarray:
    knot_times_arr = np.asarray(knot_times, dtype=np.float64).reshape(-1)
    values_arr = np.asarray(values, dtype=np.float64)
    sample_times_arr = np.asarray(sample_times, dtype=np.float64).reshape(-1)
    if knot_times_arr.shape[0] != values_arr.shape[0]:
        raise ValueError("knot_times and values must have matching first dimension")
    if knot_times_arr.shape[0] < 2:
        return np.repeat(values_arr[:1], sample_times_arr.shape[0], axis=0)
    try:
        from scipy.interpolate import CubicSpline  # type: ignore
    except Exception:
        return _sample_cosine_waypoint_path(
            knot_times=knot_times_arr,
            values=values_arr,
            sample_times=sample_times_arr,
        )
    cs = CubicSpline(
        knot_times_arr,
        values_arr,
        axis=0,
        bc_type=((1, np.zeros(values_arr.shape[1:], dtype=np.float64)), (1, np.zeros(values_arr.shape[1:], dtype=np.float64))),
    )
    out = np.asarray(cs(sample_times_arr), dtype=np.float64)
    out[0] = values_arr[0]
    out[-1] = values_arr[-1]
    return out


def _sample_cosine_waypoint_path(
    *,
    knot_times: np.ndarray,
    values: np.ndarray,
    sample_times: np.ndarray,
) -> np.ndarray:
    knot_times_arr = np.asarray(knot_times, dtype=np.float64).reshape(-1)
    values_arr = np.asarray(values, dtype=np.float64)
    sample_times_arr = np.asarray(sample_times, dtype=np.float64).reshape(-1)
    segment_idx = np.searchsorted(knot_times_arr, sample_times_arr, side="right") - 1
    segment_idx = np.clip(segment_idx, 0, knot_times_arr.shape[0] - 2)
    segment_start = knot_times_arr[segment_idx]
    segment_end = knot_times_arr[segment_idx + 1]
    phase = (sample_times_arr - segment_start) / np.maximum(segment_end - segment_start, 1e-12)
    alpha = 0.5 - 0.5 * np.cos(np.pi * np.clip(phase, 0.0, 1.0))
    out = (1.0 - alpha[..., None]) * values_arr[segment_idx] + alpha[..., None] * values_arr[segment_idx + 1]
    out[0] = values_arr[0]
    out[-1] = values_arr[-1]
    return out


def _quat_wxyz_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.asarray(a, dtype=np.float64).reshape(4)
    bw, bx, by, bz = np.asarray(b, dtype=np.float64).reshape(4)
    out = np.asarray(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(out))
    return out / norm if norm > 1e-12 else np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _quat_wxyz_conj(q: np.ndarray) -> np.ndarray:
    quat = np.asarray(q, dtype=np.float64).reshape(4)
    return np.asarray([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float64)


def _quat_wxyz_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-12 or abs(float(angle)) <= 1e-12:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = axis / axis_norm
    half = 0.5 * float(angle)
    return np.concatenate([np.asarray([np.cos(half)], dtype=np.float64), axis * np.sin(half)])


def _quat_wxyz_from_rpy(rpy_rad: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(rpy_rad, dtype=np.float64).reshape(3)
    qx = _quat_wxyz_from_axis_angle(np.asarray([1.0, 0.0, 0.0]), float(roll))
    qy = _quat_wxyz_from_axis_angle(np.asarray([0.0, 1.0, 0.0]), float(pitch))
    qz = _quat_wxyz_from_axis_angle(np.asarray([0.0, 0.0, 1.0]), float(yaw))
    return _quat_wxyz_mul(_quat_wxyz_mul(qz, qy), qx)


def _quat_wxyz_slerp(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    qa = np.asarray(a, dtype=np.float64).reshape(4)
    qb = np.asarray(b, dtype=np.float64).reshape(4)
    qa = qa / max(float(np.linalg.norm(qa)), 1e-12)
    qb = qb / max(float(np.linalg.norm(qb)), 1e-12)
    dot = float(np.dot(qa, qb))
    if dot < 0.0:
        qb = -qb
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        out = qa + float(alpha) * (qb - qa)
        return out / max(float(np.linalg.norm(out)), 1e-12)
    theta_0 = float(np.arccos(dot))
    theta = theta_0 * float(alpha)
    sin_theta = float(np.sin(theta))
    sin_theta_0 = float(np.sin(theta_0))
    s0 = float(np.cos(theta)) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return s0 * qa + s1 * qb


def _quat_wxyz_from_rotmat(rot: np.ndarray) -> np.ndarray:
    mat = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(mat))
    if trace > 0.0:
        s = float(np.sqrt(trace + 1.0) * 2.0)
        quat = np.asarray(
            [
                0.25 * s,
                (mat[2, 1] - mat[1, 2]) / s,
                (mat[0, 2] - mat[2, 0]) / s,
                (mat[1, 0] - mat[0, 1]) / s,
            ],
            dtype=np.float64,
        )
    else:
        diag_idx = int(np.argmax(np.diag(mat)))
        if diag_idx == 0:
            s = float(np.sqrt(max(0.0, 1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2])) * 2.0)
            quat = np.asarray(
                [
                    (mat[2, 1] - mat[1, 2]) / max(s, 1e-12),
                    0.25 * s,
                    (mat[0, 1] + mat[1, 0]) / max(s, 1e-12),
                    (mat[0, 2] + mat[2, 0]) / max(s, 1e-12),
                ],
                dtype=np.float64,
            )
        elif diag_idx == 1:
            s = float(np.sqrt(max(0.0, 1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2])) * 2.0)
            quat = np.asarray(
                [
                    (mat[0, 2] - mat[2, 0]) / max(s, 1e-12),
                    (mat[0, 1] + mat[1, 0]) / max(s, 1e-12),
                    0.25 * s,
                    (mat[1, 2] + mat[2, 1]) / max(s, 1e-12),
                ],
                dtype=np.float64,
            )
        else:
            s = float(np.sqrt(max(0.0, 1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1])) * 2.0)
            quat = np.asarray(
                [
                    (mat[1, 0] - mat[0, 1]) / max(s, 1e-12),
                    (mat[0, 2] + mat[2, 0]) / max(s, 1e-12),
                    (mat[1, 2] + mat[2, 1]) / max(s, 1e-12),
                    0.25 * s,
                ],
                dtype=np.float64,
            )
    norm = float(np.linalg.norm(quat))
    return quat / norm if norm > 1e-12 else np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _quat_angle_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    qa = np.asarray(a, dtype=np.float64).reshape(4)
    qb = np.asarray(b, dtype=np.float64).reshape(4)
    qa = qa / max(float(np.linalg.norm(qa)), 1e-12)
    qb = qb / max(float(np.linalg.norm(qb)), 1e-12)
    dot = min(1.0, max(-1.0, abs(float(np.dot(qa, qb)))))
    return float(np.rad2deg(2.0 * np.arccos(dot)))


def _interp_quat_wxyz(t_src: np.ndarray, q_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    src_t = np.asarray(t_src, dtype=np.float64).reshape(-1)
    src_q = np.asarray(q_src, dtype=np.float64).reshape(-1, 4)
    dst_t = np.asarray(t_dst, dtype=np.float64).reshape(-1)
    out = np.empty((dst_t.shape[0], 4), dtype=np.float64)
    for i, t in enumerate(dst_t):
        if t <= src_t[0]:
            out[i] = src_q[0]
        elif t >= src_t[-1]:
            out[i] = src_q[-1]
        else:
            hi = int(np.searchsorted(src_t, t, side="right"))
            lo = hi - 1
            denom = max(float(src_t[hi] - src_t[lo]), 1e-12)
            out[i] = _quat_wxyz_slerp(src_q[lo], src_q[hi], float((t - src_t[lo]) / denom))
    return out


def _np_or_empty(values: Any, *, shape_tail: tuple[int, ...] = ()) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, *shape_tail), dtype=np.float64)
    return arr.reshape((-1, *shape_tail)) if shape_tail else arr.reshape(-1)


def _interp_to(t_src: np.ndarray, y_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    return np.vstack([np.interp(t_dst, t_src, y_src[:, i]) for i in range(y_src.shape[1])]).T


def _orientation_series_from_log(target_log: dict[str, list]) -> tuple[np.ndarray, np.ndarray]:
    ee_rot = _np_or_empty(target_log.get("ee_rot", []), shape_tail=(3, 3)).astype(np.float64)
    ee_rot_t = _np_or_empty(target_log.get("ee_rot_t", [])).astype(np.float64).reshape(-1)
    if ee_rot.shape[0] == 0:
        ee_quat = _np_or_empty(target_log.get("ee_quat", []), shape_tail=(4,)).astype(np.float64)
        ee_quat_t = _np_or_empty(target_log.get("ee_quat_t", [])).astype(np.float64).reshape(-1)
        if ee_quat.shape[0] > 0 and ee_quat_t.shape[0] == ee_quat.shape[0]:
            return ee_quat_t, ee_quat
        return np.empty((0,), dtype=np.float64), np.empty((0, 4), dtype=np.float64)
    if ee_rot_t.shape[0] != ee_rot.shape[0]:
        return np.empty((0,), dtype=np.float64), np.empty((0, 4), dtype=np.float64)
    return ee_rot_t, np.asarray([_quat_wxyz_from_rotmat(rot) for rot in ee_rot], dtype=np.float64)


def _pose_tracking_metrics(
    *,
    prefix: str,
    actual_log: dict[str, list],
    reference_t: np.ndarray,
    reference_pos: np.ndarray,
    reference_quat: Optional[np.ndarray] = None,
) -> dict[str, float | int]:
    ee_pos = _np_or_empty(actual_log["ee_pos"], shape_tail=(3,)).astype(np.float64)
    ee_t = _np_or_empty(actual_log["ee_pos_t"]).astype(np.float64).reshape(-1)
    ref_t = np.asarray(reference_t, dtype=np.float64).reshape(-1)
    ref_pos = np.asarray(reference_pos, dtype=np.float64).reshape(-1, 3)
    out: dict[str, float | int] = {f"{prefix}_ee_pos_samples": int(ee_pos.shape[0])}
    if ee_pos.shape[0] < 2 or ee_t.shape[0] != ee_pos.shape[0]:
        out[f"{prefix}_ee_pos_rmse_m"] = float("nan")
        out[f"{prefix}_ee_final_error_m"] = float("nan")
    else:
        valid = (ref_t >= ee_t[0]) & (ref_t <= ee_t[-1])
        if np.any(valid):
            err = _interp_to(ee_t, ee_pos, ref_t[valid]) - ref_pos[valid]
            out[f"{prefix}_ee_pos_rmse_m"] = float(np.sqrt(np.mean(np.sum(err * err, axis=1))))
            out[f"{prefix}_ee_final_error_m"] = float(np.linalg.norm(ee_pos[-1] - ref_pos[-1]))
        else:
            out[f"{prefix}_ee_pos_rmse_m"] = float("nan")
            out[f"{prefix}_ee_final_error_m"] = float("nan")

    ref_quat_arr = None if reference_quat is None else np.asarray(reference_quat, dtype=np.float64).reshape(-1, 4)
    ori_t, ori_quat = _orientation_series_from_log(actual_log)
    out[f"{prefix}_ee_ori_samples"] = int(ori_quat.shape[0])
    if ref_quat_arr is None or ori_quat.shape[0] < 2 or ori_t.shape[0] != ori_quat.shape[0]:
        out[f"{prefix}_ee_ori_rmse_deg"] = float("nan")
        out[f"{prefix}_ee_final_ori_error_deg"] = float("nan")
        return out
    valid_ori = (ref_t >= ori_t[0]) & (ref_t <= ori_t[-1])
    if not np.any(valid_ori):
        out[f"{prefix}_ee_ori_rmse_deg"] = float("nan")
        out[f"{prefix}_ee_final_ori_error_deg"] = float("nan")
        return out
    ori_interp = _interp_quat_wxyz(ori_t, ori_quat, ref_t[valid_ori])
    ori_err = np.asarray(
        [_quat_angle_error_deg(actual, ref) for actual, ref in zip(ori_interp, ref_quat_arr[valid_ori])],
        dtype=np.float64,
    )
    out[f"{prefix}_ee_ori_rmse_deg"] = float(np.sqrt(np.mean(ori_err * ori_err)))
    out[f"{prefix}_ee_final_ori_error_deg"] = float(_quat_angle_error_deg(ori_quat[-1], ref_quat_arr[-1]))
    return out


def compute_target_metrics(
    *,
    target_log: dict[str, list],
    t_target: np.ndarray,
    pos_target: np.ndarray,
    quat_target: Optional[np.ndarray] = None,
    ideal_log: Optional[dict[str, Any]] = None,
) -> dict[str, float | int]:
    ee_pos = _np_or_empty(target_log["ee_pos"], shape_tail=(3,)).astype(np.float64)
    metrics: dict[str, float | int] = {"target_ee_samples": int(ee_pos.shape[0])}
    metrics.update(
        _pose_tracking_metrics(
            prefix="target_vs_reference",
            actual_log=target_log,
            reference_t=t_target,
            reference_pos=pos_target,
            reference_quat=quat_target,
        )
    )
    metrics["target_ee_pos_rmse_m"] = metrics["target_vs_reference_ee_pos_rmse_m"]
    metrics["target_ee_final_error_m"] = metrics["target_vs_reference_ee_final_error_m"]
    if ideal_log is not None:
        ideal_t = np.asarray(ideal_log.get("t", np.empty((0,))), dtype=np.float64).reshape(-1)
        ideal_pos = np.asarray(ideal_log.get("ee_pos", np.empty((0, 3))), dtype=np.float64).reshape(-1, 3)
        ideal_quat = np.asarray(ideal_log.get("ee_quat", np.empty((0, 4))), dtype=np.float64).reshape(-1, 4)
        if ideal_t.shape[0] == ideal_pos.shape[0] and ideal_t.shape[0] > 0:
            metrics.update(
                _pose_tracking_metrics(
                    prefix="target_vs_ideal",
                    actual_log=target_log,
                    reference_t=ideal_t,
                    reference_pos=ideal_pos,
                    reference_quat=ideal_quat if ideal_quat.shape[0] == ideal_t.shape[0] else None,
                )
            )
            metrics["target_ideal_ee_samples"] = int(ideal_pos.shape[0])
        else:
            metrics.update(
                {
                    "target_ideal_ee_samples": int(ideal_pos.shape[0]),
                    "target_vs_ideal_ee_pos_rmse_m": float("nan"),
                    "target_vs_ideal_ee_final_error_m": float("nan"),
                    "target_vs_ideal_ee_ori_rmse_deg": float("nan"),
                    "target_vs_ideal_ee_final_ori_error_deg": float("nan"),
                }
            )
    return metrics


def fk_site_pose(
    *,
    xml_path: Path,
    q: Sequence[float],
    site_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    import mujoco  # type: ignore

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    dof_target = len(np.asarray(q).reshape(-1))
    joint_ids = _arm_joint_ids(model, dof_target=dof_target, mujoco=mujoco)
    arm_qpos_idx = np.array([model.jnt_qposadr[jid] for jid in joint_ids], dtype=np.int32)
    data.qpos[:] = model.qpos0
    data.qpos[arm_qpos_idx] = np.asarray(q, dtype=np.float64).reshape(-1)
    mujoco.mj_forward(model, data)
    site_id = int(model.site(site_name).id)
    pos = np.asarray(data.site_xpos[site_id], dtype=np.float32).reshape(3).copy()
    quat = np.zeros((4,), dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(9))
    return pos, quat.astype(np.float32)


def _arm_joint_ids(model: Any, *, dof_target: int, mujoco: Any) -> np.ndarray:
    joint_ids = [
        i
        for i in range(model.njnt)
        if (model.joint(i).name or "").startswith("panda_joint")
    ]
    if len(joint_ids) < dof_target:
        joint_ids = [
            i
            for i in range(model.njnt)
            if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
        ]
    if len(joint_ids) < dof_target:
        raise ValueError(f"Model has only {len(joint_ids)} candidate arm joints; need {dof_target}.")
    return np.asarray(joint_ids[:dof_target], dtype=np.int32)


def _history_controller_damped_pseudoinverse(
    matrix: np.ndarray,
    *,
    damping: float = HISTORY_CONTROLLER_PSEUDOINVERSE_DAMPING,
) -> np.ndarray:
    mat = np.asarray(matrix, dtype=np.float64)
    u, singular_values, vh = np.linalg.svd(mat, full_matrices=False)
    lam = float(damping)
    singular_values_inv = singular_values / (singular_values * singular_values + lam * lam)
    return vh.T @ np.diag(singular_values_inv) @ u.T


def _history_controller_cartesian_error(
    *,
    current_pos: np.ndarray,
    current_quat_wxyz: np.ndarray,
    current_rot: np.ndarray,
    target_pos: np.ndarray,
    target_quat_wxyz: np.ndarray,
) -> np.ndarray:
    goal_quat = np.asarray(target_quat_wxyz, dtype=np.float64).reshape(4).copy()
    goal_quat /= max(float(np.linalg.norm(goal_quat)), 1e-12)
    cur_quat_for_control = np.asarray(current_quat_wxyz, dtype=np.float64).reshape(4).copy()
    cur_quat_for_control /= max(float(np.linalg.norm(cur_quat_for_control)), 1e-12)
    if float(np.dot(goal_quat, cur_quat_for_control)) < 0.0:
        cur_quat_for_control *= -1.0
    q_err = _quat_wxyz_mul(_quat_wxyz_conj(cur_quat_for_control), goal_quat)
    ori_vec_local = q_err[1:].copy()
    return np.concatenate(
        [
            np.asarray(current_pos, dtype=np.float64).reshape(3)
            - np.asarray(target_pos, dtype=np.float64).reshape(3),
            -np.asarray(current_rot, dtype=np.float64).reshape(3, 3) @ ori_vec_local,
        ],
        axis=0,
    )
