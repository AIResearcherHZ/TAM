#!/usr/bin/env python3
"""Trajectory tracking error evaluation for the real controller.

This deploy path only streams reference joint targets and records tracking
error. Run ``tam-mapping-server`` for TAM mapping.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from simadaptor.deploy.fast_policy_transport import FAST_ACTION_ENDPOINT, FAST_STATE_ENDPOINT
from simadaptor.deploy.mapping_server_meta import (
    DEFAULT_MAPPING_CONTROL_ENDPOINT,
    query_mapping_server_meta,
)
from simadaptor.deploy.runtime_common import extract_history_window_arrays


JOINT_CONTROL_MODE = "joint"

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
    dtype=float,
)

FRANKA_Q_LOWER = np.array(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
    dtype=float,
)
FRANKA_Q_UPPER = np.array(
    [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
    dtype=float,
)


def send_joint_control_mode(
    client: Any,
    *,
    log_prefix: str = "[control_mode]",
) -> None:
    client.send_command(extra_fields={"control_mode": JOINT_CONTROL_MODE})
    print(f"{log_prefix} Requested controller control_mode={JOINT_CONTROL_MODE}.")


def _read_mapping_server_meta(endpoint: str, *, label: str) -> dict[str, Any]:
    return query_mapping_server_meta(
        str(endpoint),
        source=f"trajectory_tracking:{label}",
    )


def _parse_vec(arg: str, n: int) -> Tuple[float, ...]:
    vals = [float(x) for x in str(arg).split(",") if x.strip() != ""]
    if len(vals) != n:
        raise argparse.ArgumentTypeError(f"Expected {n} comma-separated floats, got {len(vals)}: {arg}")
    return tuple(vals)


def _as_vec7(values: Optional[Sequence[float]], default: Sequence[float]) -> np.ndarray:
    arr = np.asarray(default if values is None else values, dtype=float).reshape(-1)
    if arr.size == 1:
        arr = np.repeat(arr, 7)
    if arr.size < 7:
        raise ValueError(f"Expected at least 7 values, got {arr.size}")
    return arr[:7].astype(float)


def generate_sine_reference(
    *,
    duration: float,
    dt_s: float,
    amp_deg_range: Tuple[float, float],
    freq_hz_range: Tuple[float, float],
    seed: int,
    num_joints: int = 7,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a smooth sine reference around ``HOME_Q``."""
    rng = np.random.default_rng(int(seed))
    t_ref = np.arange(0.0, float(duration), float(dt_s), dtype=float)
    home = HOME_Q[:num_joints]

    amp_min, amp_max = float(min(amp_deg_range)), float(max(amp_deg_range))
    amp = rng.uniform(amp_min, amp_max, size=num_joints) * np.pi / 180.0
    amp_scales = np.array([0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 2.0], dtype=float)[:num_joints]
    amp *= amp_scales

    sign = rng.choice([-1.0, 1.0], size=num_joints)
    fmin, fmax = float(min(freq_hz_range)), float(max(freq_hz_range))
    freq = rng.uniform(fmin, fmax, size=num_joints)
    omega = 2.0 * np.pi * freq

    ramp_dur = max(float(dt_s), min(2.0, float(duration)))
    s = np.clip(t_ref / ramp_dur, 0.0, 1.0)
    env = 0.5 * (1.0 - np.cos(np.pi * s))
    g = np.empty_like(t_ref)
    in_ramp = t_ref <= ramp_dur
    g[in_ramp] = 0.5 * (
        t_ref[in_ramp] - (ramp_dur / np.pi) * np.sin(np.pi * t_ref[in_ramp] / ramp_dur)
    )
    g[~in_ramp] = t_ref[~in_ramp] - 0.5 * ramp_dur

    phase = g[:, None] * omega[None, :]
    phase_dot = env[:, None] * omega[None, :]
    q_ref = home[None, :] + sign[None, :] * amp[None, :] * np.sin(phase)
    dq_ref = sign[None, :] * amp[None, :] * np.cos(phase) * phase_dot
    return t_ref, q_ref.astype(np.float32), dq_ref.astype(np.float32)


def _smooth_waypoint_interpolate(
    waypoints: np.ndarray,
    *,
    duration: float,
    dt_s: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    waypoints = np.asarray(waypoints, dtype=float)
    if waypoints.ndim != 2 or waypoints.shape[1] != 7 or waypoints.shape[0] < 2:
        raise ValueError(f"Expected waypoints shaped (N, 7) with N>=2, got {waypoints.shape}")
    t_ref = np.arange(0.0, float(duration), float(dt_s), dtype=float)
    if t_ref.size == 0:
        raise ValueError("Reference duration/dt produced zero samples.")
    num_segments = waypoints.shape[0] - 1
    segment_duration = float(duration) / float(num_segments)
    u = np.clip(t_ref / max(float(duration), 1e-9) * num_segments, 0.0, num_segments - 1e-9)
    seg_idx = np.floor(u).astype(int)
    s = u - seg_idx
    h = 3.0 * s**2 - 2.0 * s**3
    dh_dt = (6.0 * s * (1.0 - s)) / max(segment_duration, 1e-9)
    q0 = waypoints[seg_idx]
    q1 = waypoints[seg_idx + 1]
    dq = q1 - q0
    q_ref = q0 + h[:, None] * dq
    dq_ref = dh_dt[:, None] * dq
    return t_ref, q_ref.astype(np.float32), dq_ref.astype(np.float32)


def generate_waypoint_reference(
    *,
    duration: float,
    dt_s: float,
    num_waypoints: int,
    amp_deg_range: Tuple[float, float],
    seed: int,
    return_home: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a smooth random joint waypoint reference without local TAM."""
    rng = np.random.default_rng(int(seed))
    n = max(2, int(num_waypoints))
    if return_home and n < 3:
        n = 3
    amp_min, amp_max = float(min(amp_deg_range)), float(max(amp_deg_range))
    amp = rng.uniform(amp_min, amp_max, size=(n, 7)) * np.pi / 180.0
    direction = rng.choice([-1.0, 1.0], size=(n, 7))
    waypoints = HOME_Q[None, :] + direction * amp
    waypoints[0] = HOME_Q
    if return_home:
        waypoints[-1] = HOME_Q
    waypoints = np.clip(waypoints, FRANKA_Q_LOWER + 0.05, FRANKA_Q_UPPER - 0.05)
    t_ref, q_ref, dq_ref = _smooth_waypoint_interpolate(
        waypoints,
        duration=float(duration),
        dt_s=float(dt_s),
    )
    return t_ref, q_ref, dq_ref, waypoints.astype(np.float32)


def _load_reference_npz(path: Path, *, dt_s: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path.expanduser().resolve(), allow_pickle=True)
    q_key = "q_ref" if "q_ref" in data else ("q" if "q" in data else None)
    if q_key is None:
        raise ValueError(f"{path} must contain q_ref or q.")
    q_ref = np.asarray(data[q_key], dtype=np.float32)
    if q_ref.ndim != 2 or q_ref.shape[1] < 7:
        raise ValueError(f"Reference q must be shaped (T, >=7), got {q_ref.shape}")
    q_ref = q_ref[:, :7]
    if "t_ref" in data:
        t_ref = np.asarray(data["t_ref"], dtype=np.float64)
    elif "t" in data:
        t_ref = np.asarray(data["t"], dtype=np.float64)
    elif "times" in data:
        t_ref = np.asarray(data["times"], dtype=np.float64)
    else:
        t_ref = np.arange(q_ref.shape[0], dtype=np.float64) * float(dt_s)

    if "dq_ref" in data:
        dq_ref = np.asarray(data["dq_ref"], dtype=np.float32)
    elif "qd_ref" in data:
        dq_ref = np.asarray(data["qd_ref"], dtype=np.float32)
    elif "dq" in data:
        dq_ref = np.asarray(data["dq"], dtype=np.float32)
    elif "qd" in data:
        dq_ref = np.asarray(data["qd"], dtype=np.float32)
    else:
        dq_ref = np.gradient(q_ref, t_ref, axis=0).astype(np.float32)
    return t_ref, q_ref, dq_ref[:, :7]


def build_reference(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if args.reference == "npz":
        if args.reference_npz is None:
            raise SystemExit("--reference npz requires --reference-npz.")
        t_ref, q_ref, dq_ref = _load_reference_npz(args.reference_npz, dt_s=float(args.dt))
        return t_ref, q_ref, dq_ref, None
    if args.waypoints_npz is not None:
        data = np.load(args.waypoints_npz.expanduser().resolve(), allow_pickle=True)
        key = "q_waypoints" if "q_waypoints" in data else ("waypoints" if "waypoints" in data else None)
        if key is None:
            raise ValueError(f"{args.waypoints_npz} must contain q_waypoints or waypoints.")
        waypoints = np.asarray(data[key], dtype=np.float32)[:, :7]
        t_ref, q_ref, dq_ref = _smooth_waypoint_interpolate(
            waypoints,
            duration=float(args.duration),
            dt_s=float(args.dt),
        )
        return t_ref, q_ref, dq_ref, waypoints
    if args.reference == "waypoints":
        t_ref, q_ref, dq_ref, waypoints = generate_waypoint_reference(
            duration=float(args.duration),
            dt_s=float(args.dt),
            num_waypoints=int(args.num_waypoints),
            amp_deg_range=tuple(args.amp_deg),
            seed=int(args.seed),
            return_home=bool(args.return_home_reference),
        )
        return t_ref, q_ref, dq_ref, waypoints
    t_ref, q_ref, dq_ref = generate_sine_reference(
        duration=float(args.duration),
        dt_s=float(args.dt),
        amp_deg_range=tuple(args.amp_deg),
        freq_hz_range=tuple(args.freq_hz),
        seed=int(args.seed),
    )
    return t_ref, q_ref, dq_ref, None


def _maybe_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except Exception:
        return None


def _plot_compare(t_ref, q_ref, t_meas, q_meas, t_sim, q_sim, out_path: Path) -> None:
    plt = _maybe_import_matplotlib()
    if plt is None or not (np.asarray(q_ref).size and np.asarray(t_ref).size):
        return
    fig, ax = plt.subplots(q_ref.shape[1], 1, figsize=(10, 12), sharex=True)
    for i in range(q_ref.shape[1]):
        if np.asarray(q_sim).size and np.asarray(t_sim).size:
            ax[i].plot(t_sim, q_sim[:, i], label="sim_pd", lw=1.0)
        if np.asarray(t_meas).size and np.asarray(q_meas).size:
            ax[i].plot(t_meas, q_meas[:, i], label="meas", lw=0.7)
        ax[i].plot(t_ref, q_ref[:, i], label="ref", lw=0.6, linestyle="--")
        ax[i].set_ylabel(f"q{i+1}")
    ax[-1].set_xlabel("time (s)")
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_velocity(t_ref, qd_ref, t_meas, qd_meas, t_sim, qd_sim, out_path: Path) -> None:
    plt = _maybe_import_matplotlib()
    if plt is None or not (np.asarray(qd_ref).size and np.asarray(t_ref).size):
        return
    fig, ax = plt.subplots(qd_ref.shape[1], 1, figsize=(10, 12), sharex=True)
    for i in range(qd_ref.shape[1]):
        if np.asarray(qd_sim).size and np.asarray(t_sim).size:
            ax[i].plot(t_sim, qd_sim[:, i], label="sim_pd", lw=1.0)
        if np.asarray(t_meas).size and np.asarray(qd_meas).size:
            ax[i].plot(t_meas, qd_meas[:, i], label="meas", lw=0.7)
        ax[i].plot(t_ref, qd_ref[:, i], label="ref", lw=0.6, linestyle="--")
        ax[i].set_ylabel(f"qd{i+1}")
    ax[-1].set_xlabel("time (s)")
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_torque(t_meas, tau_cmd, tau_delta, out_path: Path) -> None:
    plt = _maybe_import_matplotlib()
    if plt is None or not (np.asarray(t_meas).size and np.asarray(tau_cmd).size):
        return
    min_len = min(t_meas.shape[0], tau_cmd.shape[0])
    if tau_delta is not None and np.asarray(tau_delta).size:
        min_len = min(min_len, tau_delta.shape[0])
    if min_len <= 1:
        return
    t_plot = t_meas[:min_len]
    tau_plot = tau_cmd[:min_len]
    delta_plot = tau_delta[:min_len] if tau_delta is not None and np.asarray(tau_delta).size else None
    fig, ax = plt.subplots(tau_plot.shape[1], 1, figsize=(10, 12), sharex=True)
    for i in range(tau_plot.shape[1]):
        ax[i].plot(t_plot, tau_plot[:, i], label="tau_cmd", lw=0.8)
        if delta_plot is not None:
            ax[i].plot(t_plot, delta_plot[:, i], label="tau_delta", lw=0.8)
        ax[i].set_ylabel(f"tau{i+1}")
    ax[-1].set_xlabel("time (s)")
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_metric_over_time(metrics: Sequence[dict], key: str, out_path: Path, ylabel: str) -> None:
    plt = _maybe_import_matplotlib()
    if plt is None:
        return
    xs = []
    ys = []
    for m in metrics:
        val = m.get(key)
        if val is None or not np.isfinite(val):
            continue
        xs.append(int(m.get("iter", len(xs))))
        ys.append(float(val))
    if not xs:
        return
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(xs, ys, marker="o", lw=1.0)
    ax.set_xlabel("iteration")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _first_present(sample: dict, keys: Sequence[str]) -> Optional[np.ndarray]:
    for key in keys:
        if key in sample and sample[key] is not None:
            arr = np.asarray(sample[key], dtype=np.float32).reshape(-1)
            if arr.size:
                return arr
    return None


def _window_to_arrays(window: Sequence[dict]):
    return extract_history_window_arrays(
        window,
        dof=7,
        q_keys=("q", "qpos"),
        dq_keys=("dq", "qd", "qvel"),
        tau_keys=("tau_cmd", "tau_commanded", "tau", "u_des", "u", "tau_measured"),
        gravity_keys=("gravity",),
        t_keys=("t", "t_raw", "timestamp"),
    )


def _log_window(
    *,
    window: Sequence[dict],
    log: dict,
    start_wall: float,
    last_logged_t: float,
) -> float:
    rows = []
    for sample in window:
        if not isinstance(sample, dict):
            continue
        t_controller = None
        for key in ("t", "t_raw", "timestamp"):
            if key in sample and sample[key] is not None:
                try:
                    t_controller = float(sample[key])
                    break
                except Exception:
                    t_controller = None
        if t_controller is None:
            continue
        q = _first_present(sample, ("q", "qpos"))
        dq = _first_present(sample, ("dq", "qd", "qvel"))
        if q is None or dq is None:
            continue
        tau = _first_present(sample, ("tau_cmd", "tau_commanded", "tau", "u_des", "u"))
        gravity = _first_present(sample, ("gravity",))
        delta = sample.get("tau_adaptor_delta", None)
        rows.append(
            (
                t_controller,
                np.asarray(q[:7], dtype=np.float32),
                np.asarray(dq[:7], dtype=np.float32),
                np.full((7,), np.nan, dtype=np.float32) if tau is None else np.asarray(tau[:7], dtype=np.float32),
                np.full((7,), np.nan, dtype=np.float32) if gravity is None else np.asarray(gravity[:7], dtype=np.float32),
                np.full((7,), np.nan, dtype=np.float32) if delta is None else np.asarray(delta, dtype=np.float32)[:7],
            )
        )
    if not rows:
        return last_logged_t
    t_controller_arr = np.asarray([r[0] for r in rows], dtype=np.float64)
    now_ts = time.perf_counter()
    t_used_arr = (now_ts + (t_controller_arr - t_controller_arr[-1]) - start_wall).astype(np.float64)
    for (t_controller, q, dq, tau, gravity, delta), t_used in zip(rows, t_used_arr):
        t_used = float(t_used)
        if t_used <= last_logged_t:
            continue
        log["t"].append(t_used)
        log["t_controller"].append(float(t_controller))
        log["q"].append(q)
        log["dq"].append(dq)
        log["tau_cmd"].append(tau)
        log["gravity"].append(gravity)
        log["tau_adaptor_delta"].append(delta)
        last_logged_t = t_used
    return last_logged_t


def _interp_to(t_src: np.ndarray, data_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    return np.vstack([np.interp(t_dst, t_src, data_src[:, i]) for i in range(data_src.shape[1])]).T


def _find_best_time_shift(
    t_meas: np.ndarray,
    q_meas: np.ndarray,
    t_target: np.ndarray,
    q_target: np.ndarray,
    search_margin: Optional[float] = None,
    num_candidates: int = 301,
) -> float:
    if not (t_meas.size and q_meas.size and t_target.size and q_target.size):
        return 0.0
    span = float(t_target[-1] - t_target[0]) if t_target.size else 0.0
    margin = search_margin if search_margin is not None else min(5.0, max(0.5, 0.2 * span))
    best_shift, best_mse = 0.0, np.inf
    for shift in np.linspace(-margin, margin, int(num_candidates)):
        shifted_t = t_meas + float(shift)
        valid = (t_target >= shifted_t[0]) & (t_target <= shifted_t[-1])
        if not np.any(valid):
            continue
        q_interp = _interp_to(shifted_t, q_meas, t_target[valid])
        mse = float(np.mean((q_interp - q_target[valid]) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_shift = float(shift)
    return best_shift


def _compute_metrics(
    *,
    t_meas: np.ndarray,
    q_meas: np.ndarray,
    qd_meas: np.ndarray,
    t_ref: np.ndarray,
    q_ref: np.ndarray,
    qd_ref: np.ndarray,
    t_sim: np.ndarray,
    q_sim: np.ndarray,
    qd_sim: np.ndarray,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {"num_samples": int(t_meas.shape[0])}
    if t_meas.size == 0 or q_meas.size == 0:
        metrics.update({k: float("nan") for k in ("time_shift_s", "q_mse", "q_rmse", "qd_mse", "qd_rmse", "q_ref_mse", "q_ref_rmse")})
        return metrics
    use_sim = bool(t_sim.size and q_sim.size)
    t_target = t_sim if use_sim else t_ref
    q_target = q_sim if use_sim else q_ref
    qd_target = qd_sim if use_sim else qd_ref
    time_shift = _find_best_time_shift(t_meas, q_meas, t_target, q_target)
    metrics["time_shift_s"] = float(time_shift)
    t_aligned = t_meas + time_shift
    valid = (t_target >= t_aligned[0]) & (t_target <= t_aligned[-1])
    if not np.any(valid):
        metrics.update({k: float("nan") for k in ("q_mse", "q_rmse", "qd_mse", "qd_rmse", "q_ref_mse", "q_ref_rmse")})
        return metrics
    q_interp = _interp_to(t_aligned, q_meas, t_target[valid])
    q_mse = float(np.mean((q_interp - q_target[valid]) ** 2))
    metrics["q_mse"] = q_mse
    metrics["q_rmse"] = float(np.sqrt(q_mse))
    if qd_meas.size and qd_target.size:
        qd_interp = _interp_to(t_aligned, qd_meas, t_target[valid])
        qd_mse = float(np.mean((qd_interp - qd_target[valid]) ** 2))
        metrics["qd_mse"] = qd_mse
        metrics["qd_rmse"] = float(np.sqrt(qd_mse))
    else:
        metrics["qd_mse"] = float("nan")
        metrics["qd_rmse"] = float("nan")
    valid_ref = (t_ref >= t_aligned[0]) & (t_ref <= t_aligned[-1])
    if np.any(valid_ref):
        q_ref_interp = _interp_to(t_aligned, q_meas, t_ref[valid_ref])
        q_ref_mse = float(np.mean((q_ref_interp - q_ref[valid_ref]) ** 2))
        metrics["q_ref_mse"] = q_ref_mse
        metrics["q_ref_rmse"] = float(np.sqrt(q_ref_mse))
    else:
        metrics["q_ref_mse"] = float("nan")
        metrics["q_ref_rmse"] = float("nan")
    return metrics


def _simulate_ideal_pd(
    *,
    q_ref: np.ndarray,
    dq_ref: np.ndarray,
    dt_s: float,
    stiffness: np.ndarray,
    damping: np.ndarray,
    xml_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import mujoco  # type: ignore
        from simadaptor.physics import rollout
    except Exception as exc:
        print(f"[ideal] Skipping ideal PD simulation: {exc}")
        return np.empty((0, 7), dtype=np.float32), np.empty((0, 7), dtype=np.float32), np.empty((0,), dtype=np.float64)
    xml_path = xml_path.expanduser().resolve()
    if not xml_path.exists():
        print(f"[ideal] XML not found; skipping ideal PD simulation: {xml_path}")
        return np.empty((0, 7), dtype=np.float32), np.empty((0, 7), dtype=np.float32), np.empty((0,), dtype=np.float64)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    joint_ids = rollout.guess_arm_joint_ids(model, dof_target=7)
    arm_idx = np.asarray([model.jnt_qposadr[j] for j in joint_ids], dtype=np.int32)
    model.dof_frictionloss[:] = 0.0
    model.dof_damping[:] = 0.0
    base_dt = float(model.opt.timestep)
    substeps = max(1, int(round(float(dt_s) / base_dt)))
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qpos[arm_idx] = np.asarray(q_ref[0], dtype=np.float32)
    data.qvel[:] = 0.0
    qs: list[np.ndarray] = []
    qds: list[np.ndarray] = []
    for qt, qdt in zip(np.asarray(q_ref), np.asarray(dq_ref)):
        for _ in range(substeps):
            q_cur = np.asarray(data.qpos[arm_idx])
            qd_cur = np.asarray(data.qvel[arm_idx])
            tau = stiffness * (qt - q_cur) + damping * (qdt - qd_cur)
            data.ctrl[:] = 0.0
            data.ctrl[: tau.shape[0]] = tau.astype(np.float32)
            mujoco.mj_step(model, data)
        qs.append(np.asarray(data.qpos[arm_idx], dtype=np.float32))
        qds.append(np.asarray(data.qvel[arm_idx], dtype=np.float32))
    q_sim = np.asarray(qs, dtype=np.float32)
    qd_sim = np.asarray(qds, dtype=np.float32)
    t_sim = np.arange(q_sim.shape[0], dtype=np.float64) * float(dt_s)
    return q_sim, qd_sim, t_sim


def _run_iteration(
    *,
    client: Any,
    t_ref: np.ndarray,
    q_ref: np.ndarray,
    dq_ref: np.ndarray,
    stiffness: np.ndarray,
    damping: np.ndarray,
    poll_timeout_ms: int,
    reset_controller: bool,
    reset_sleep_s: float,
    go_home: bool,
    home_duration_s: float,
    home_dt: float,
    debug: bool,
) -> dict[str, list[Any]]:
    if reset_controller:
        client.reset_history()
        client.reset_best_effort()
        time.sleep(float(reset_sleep_s))

    send_joint_control_mode(client, log_prefix="[trajectory_tracking]")
    client.send_command(stiffness=stiffness, damping=damping)
    client.reset_history()

    log: dict[str, list[Any]] = {
        "t": [],
        "t_controller": [],
        "q": [],
        "dq": [],
        "tau_cmd": [],
        "gravity": [],
        "tau_adaptor_delta": [],
    }
    last_logged_t = -float("inf")
    start_wall = time.perf_counter()
    last_status_wall = start_wall
    num_windows = 0
    num_valid = 0
    try:
        for q_t, dq_t, t_t in zip(q_ref, dq_ref, t_ref):
            target_wall = start_wall + float(t_t)
            client.send_command(target_q=q_t, target_dq=dq_t)
            while True:
                now = time.perf_counter()
                remaining = target_wall - now
                if remaining <= 0.0:
                    break
                timeout = int(min(float(poll_timeout_ms), max(0.0, remaining) * 1000.0))
                window = client.poll_window(timeout_ms=max(0, timeout))
                if window:
                    num_windows += 1
                    arrays = _window_to_arrays(window)
                    if arrays is not None:
                        num_valid += int(arrays[0].shape[0])
                    last_logged_t = _log_window(
                        window=window,
                        log=log,
                        start_wall=start_wall,
                        last_logged_t=last_logged_t,
                    )
                if remaining > 0.002:
                    time.sleep(min(0.001, remaining))
            if debug and (time.perf_counter() - last_status_wall) >= 1.0:
                print(f"[trajectory_tracking] windows={num_windows} valid_samples={num_valid}")
                last_status_wall = time.perf_counter()
    except KeyboardInterrupt:
        print("[trajectory_tracking] Interrupted; stopping trajectory.")
    finally:
        if go_home:
            try:
                send_joint_control_mode(client, log_prefix="[trajectory_tracking]")
                client.go_to_home(
                    HOME_Q,
                    duration_s=float(home_duration_s),
                    dt=float(home_dt),
                    stiffness=stiffness,
                    damping=damping,
                )
            except Exception as exc:
                print(f"[trajectory_tracking] go_to_home failed: {exc}")
    return log


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure real-robot joint tracking error for sine or waypoint references. "
            "TAM mapping is handled externally by tam-mapping-server."
        )
    )
    parser.add_argument("--reference", choices=["sine", "waypoints", "npz"], default="sine")
    parser.add_argument("--reference-npz", type=Path, default=None)
    parser.add_argument("--waypoints-npz", type=Path, default=None)
    parser.add_argument("--num-waypoints", type=int, default=5)
    parser.add_argument("--return-home-reference", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--outdir", type=Path, default=Path("eval_logs") / "trajectory_tracking")

    parser.add_argument("--history-endpoint", type=str, default="tcp://192.168.1.101:5555")
    parser.add_argument("--command-endpoint", type=str, default="tcp://192.168.1.101:5556")
    parser.add_argument("--request-endpoint", type=str, default="tcp://192.168.1.101:5557")
    parser.add_argument(
        "--mapping-server-endpoint",
        type=str,
        default=DEFAULT_MAPPING_CONTROL_ENDPOINT,
        help=(
            "REQ/REP status endpoint for scripts/deploy/mapping_server.py. "
            "If no mapping server answers, iteration logs record mapping_server_mode=none."
        ),
    )
    parser.add_argument("--history-buffer", type=int, default=5000)
    parser.add_argument("--poll-timeout-ms", type=int, default=5)
    parser.add_argument("--fast-transport", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fast-state-endpoint", type=str, default=FAST_STATE_ENDPOINT)
    parser.add_argument("--fast-action-endpoint", type=str, default=FAST_ACTION_ENDPOINT)
    parser.add_argument("--fast-state-max-age-s", type=float, default=0.05)
    parser.add_argument("--fast-startup-timeout-s", type=float, default=2.0)
    parser.add_argument("--fast-action-requires-state", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--duration", type=float, default=14.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--amp-deg", type=lambda s: _parse_vec(s, 2), default=(30.0, 35.0))
    parser.add_argument("--freq-hz", type=lambda s: _parse_vec(s, 2), default=(0.20, 0.30))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stiffness", type=float, nargs="+", default=None)
    parser.add_argument("--damping", type=float, nargs="+", default=None)
    parser.add_argument("--xml", type=Path, default=Path("assets/franka_panda/panda_pandagripper.xml"))
    parser.add_argument("--compare-ideal", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--reset-controller", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reset-sleep-s", type=float, default=0.5)
    parser.add_argument("--go-home", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--home-duration-s", type=float, default=3.0)
    parser.add_argument("--home-dt", type=float, default=0.01)
    parser.add_argument("--sleep-between-iters", type=float, default=0.0)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--excite", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--excite-duration", dest="duration", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--excite-dt", dest="dt", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--excite-amp-deg", dest="amp_deg", type=lambda s: _parse_vec(s, 2), default=argparse.SUPPRESS)
    parser.add_argument("--excite-freq-hz", dest="freq_hz", type=lambda s: _parse_vec(s, 2), default=argparse.SUPPRESS)
    parser.add_argument("--excite-seed", dest="seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--log-dir", dest="outdir", type=Path, default=argparse.SUPPRESS)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> Path:
    args = build_arg_parser().parse_args(argv)

    t_ref, q_ref, dq_ref, waypoints = build_reference(args)
    stiff = _as_vec7(args.stiffness, [10, 10, 10, 10, 5, 5, 5])
    damp = (
        2.0 * np.sqrt(stiff) * np.array([1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5], dtype=float)
        if args.damping is None
        else _as_vec7(args.damping, [0, 0, 0, 0, 0, 0, 0])
    )

    q_sim = np.empty((0, 7), dtype=np.float32)
    qd_sim = np.empty((0, 7), dtype=np.float32)
    t_sim = np.empty((0,), dtype=np.float64)
    if bool(args.compare_ideal):
        q_sim, qd_sim, t_sim = _simulate_ideal_pd(
            q_ref=q_ref,
            dq_ref=dq_ref,
            dt_s=float(args.dt),
            stiffness=stiff,
            damping=damp,
            xml_path=args.xml,
        )

    run_dir = args.outdir.expanduser().resolve() / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    initial_mapping_server_meta = _read_mapping_server_meta(
        str(args.mapping_server_endpoint),
        label="run_start",
    )
    np.savez(
        run_dir / "reference.npz",
        t_ref=t_ref,
        q_ref=q_ref,
        dq_ref=dq_ref,
        q_waypoints=np.empty((0, 7), dtype=np.float32) if waypoints is None else waypoints,
        q_sim=q_sim,
        qd_sim=qd_sim,
        t_sim=t_sim,
        stiffness=stiff,
        damping=damp,
    )
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "argv": list(argv) if argv is not None else None,
                "parsed_args": vars(args),
                "runtime_mapping_behavior": "external_only",
                "mapping_server_mode": str(initial_mapping_server_meta.get("mapping_mode", "none")),
                "mapping_server_status": initial_mapping_server_meta,
            },
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )

    from simadaptor.deploy.history_client import HistoryControllerClient

    client = HistoryControllerClient(
        history_endpoint=str(args.history_endpoint),
        command_endpoint=str(args.command_endpoint),
        history_buffer=int(args.history_buffer),
        request_endpoint=str(args.request_endpoint),
        fast_transport_enabled=bool(args.fast_transport),
        fast_state_endpoint=str(args.fast_state_endpoint),
        fast_action_endpoint=str(args.fast_action_endpoint),
        fast_state_max_age_s=float(args.fast_state_max_age_s),
        fast_action_requires_state=bool(args.fast_action_requires_state),
    )
    if bool(args.fast_transport):
        state = client.wait_for_fast_state(timeout_s=float(args.fast_startup_timeout_s))
        print(
            "[trajectory_tracking] Fast transport ready: "
            f"state_seq={int(state.seq)} age={float(state.receive_age_sec):.4f}s "
            f"state={args.fast_state_endpoint} action={args.fast_action_endpoint}"
        )

    metrics: list[dict[str, Any]] = []
    start_wall = time.perf_counter()
    try:
        for idx in range(int(args.iters)):
            iter_dir = run_dir / f"iter_{idx:03d}"
            iter_dir.mkdir(parents=True, exist_ok=True)
            mapping_server_meta = _read_mapping_server_meta(
                str(args.mapping_server_endpoint),
                label=f"iter_{idx:03d}:start",
            )
            (iter_dir / "mapping_server_status.json").write_text(
                json.dumps(mapping_server_meta, indent=2, sort_keys=True, default=str),
                encoding="utf-8",
            )
            print(
                f"[trajectory_tracking] Iteration {idx + 1}/{args.iters} -> {iter_dir} "
                f"mapping_server={mapping_server_meta.get('mapping_mode', 'none')}"
            )
            log = _run_iteration(
                client=client,
                t_ref=t_ref,
                q_ref=q_ref,
                dq_ref=dq_ref,
                stiffness=stiff,
                damping=damp,
                poll_timeout_ms=int(args.poll_timeout_ms),
                reset_controller=bool(args.reset_controller),
                reset_sleep_s=float(args.reset_sleep_s),
                go_home=bool(args.go_home),
                home_duration_s=float(args.home_duration_s),
                home_dt=float(args.home_dt),
                debug=bool(args.debug),
            )

            t_meas = np.asarray(log["t"], dtype=np.float64)
            t_controller = np.asarray(log["t_controller"], dtype=np.float64)
            q_meas = np.asarray(log["q"], dtype=np.float32) if log["q"] else np.empty((0, 7), dtype=np.float32)
            qd_meas = np.asarray(log["dq"], dtype=np.float32) if log["dq"] else np.empty((0, 7), dtype=np.float32)
            tau_cmd = np.asarray(log["tau_cmd"], dtype=np.float32) if log["tau_cmd"] else np.empty((0, 7), dtype=np.float32)
            gravity = np.asarray(log["gravity"], dtype=np.float32) if log["gravity"] else np.empty((0, 7), dtype=np.float32)
            tau_delta = np.asarray(log["tau_adaptor_delta"], dtype=np.float32) if log["tau_adaptor_delta"] else np.empty((0, 7), dtype=np.float32)

            iter_metrics = _compute_metrics(
                t_meas=t_meas,
                q_meas=q_meas,
                qd_meas=qd_meas,
                t_ref=t_ref,
                q_ref=q_ref,
                qd_ref=dq_ref,
                t_sim=t_sim,
                q_sim=q_sim,
                qd_sim=qd_sim,
            )
            iter_metrics["iter"] = int(idx)
            iter_metrics["elapsed_s"] = float(time.perf_counter() - start_wall)
            iter_metrics["reference"] = str(args.reference)
            iter_metrics["transport"] = "fast" if bool(args.fast_transport) else "history"
            iter_metrics["mapping_server_mode"] = str(mapping_server_meta.get("mapping_mode", "none"))
            iter_metrics["mapping_server_status"] = mapping_server_meta
            metrics.append(iter_metrics)
            (iter_dir / "metrics.json").write_text(json.dumps(iter_metrics, indent=2, sort_keys=True), encoding="utf-8")

            time_shift = float(iter_metrics.get("time_shift_s", 0.0))
            if not np.isfinite(time_shift):
                time_shift = 0.0
            t_aligned = t_meas + time_shift
            _plot_compare(t_ref, q_ref, t_aligned, q_meas, t_sim, q_sim, iter_dir / "tracking_compare.png")
            _plot_velocity(t_ref, dq_ref, t_aligned, qd_meas, t_sim, qd_sim, iter_dir / "velocity_compare.png")
            _plot_torque(t_aligned, tau_cmd, tau_delta, iter_dir / "torque_compare.png")

            np.savez(
                iter_dir / "tracking_log.npz",
                t=t_meas,
                t_aligned=t_aligned,
                t_controller=t_controller,
                q=q_meas,
                qd=qd_meas,
                tau_cmd=tau_cmd,
                gravity=gravity,
                tau_adaptor_delta=tau_delta,
                q_ref=q_ref,
                qd_ref=dq_ref,
                t_ref=t_ref,
                q_sim=q_sim,
                qd_sim=qd_sim,
                t_sim=t_sim,
                time_shift=np.asarray(time_shift, dtype=np.float64),
                stiffness=stiff,
                damping=damp,
            )
            if float(args.sleep_between_iters) > 0.0 and idx < int(args.iters) - 1:
                time.sleep(float(args.sleep_between_iters))
    finally:
        client.close()

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    with (run_dir / "metrics.csv").open("w", encoding="utf-8") as f:
        f.write("iter,elapsed_s,q_mse,q_rmse,qd_mse,qd_rmse,q_ref_mse,q_ref_rmse,time_shift_s,num_samples,transport,mapping_server_mode\n")
        for m in metrics:
            f.write(
                f"{m.get('iter', '')},{m.get('elapsed_s', '')},{m.get('q_mse', '')},"
                f"{m.get('q_rmse', '')},{m.get('qd_mse', '')},{m.get('qd_rmse', '')},"
                f"{m.get('q_ref_mse', '')},{m.get('q_ref_rmse', '')},{m.get('time_shift_s', '')},"
                f"{m.get('num_samples', '')},{m.get('transport', '')},{m.get('mapping_server_mode', '')}\n"
            )
    _plot_metric_over_time(metrics, "q_rmse", run_dir / "q_rmse_over_iters.png", "q RMSE")
    _plot_metric_over_time(metrics, "q_ref_rmse", run_dir / "q_ref_rmse_over_iters.png", "q RMSE vs ref")
    _plot_metric_over_time(metrics, "qd_rmse", run_dir / "qd_rmse_over_iters.png", "qd RMSE")
    print(f"[trajectory_tracking] Logs saved to {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
