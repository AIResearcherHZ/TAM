"""Replay source-to-OSC simulation logs in a MuJoCo viewer.

Loads the per-condition ``source_log.npz`` / ``target_log.npz`` trajectories
written by the legacy sim backend (``--sim-backend legacy``) and replays them
kinematically: all conditions are attached side by side in one MuJoCo scene so
``direct_osc`` vs ``tam_carried`` (and ``ideal_model``) can be compared frame
by frame. Also supports offline video export via ``--save-video``.

The batched backend only writes numeric summaries; runs produced with it
cannot be replayed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import mujoco
import numpy as np

DEFAULT_LOG_ROOT = Path("eval_logs") / "source_to_osc_tam_sim"

# Display order and colors for known condition keys; extras get palette colors.
PREFERRED_ORDER = ("ideal_model", "direct_osc", "tam_carried")
CONDITION_COLORS = {
    "ideal_model": (0.45, 0.55, 0.75, 1.0),
    "direct_osc": (0.85, 0.33, 0.25, 1.0),
    "tam_carried": (0.25, 0.70, 0.35, 1.0),
}
EXTRA_PALETTE = (
    (0.85, 0.65, 0.20, 1.0),
    (0.55, 0.35, 0.75, 1.0),
    (0.25, 0.65, 0.70, 1.0),
)
REF_PATH_RGBA = (0.75, 0.75, 0.78, 0.35)
GOAL_RGBA = (0.95, 0.80, 0.15, 0.9)


@dataclass
class ConditionTrajectory:
    key: str
    t: np.ndarray  # (T,) absolute time, source phase + OSC phase
    q: np.ndarray  # (T, dof)
    ee_pos: np.ndarray  # (T, 3)
    switch_t: float  # time at which the controller switches to OSC
    ref_ee_path: np.ndarray  # (R, 3) desired OSC end-effector path
    color: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    qpos_adr: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    base_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))


def _arm_joint_names(model: mujoco.MjModel, dof: Optional[int]) -> list[str]:
    """Mirror of the experiment's _arm_indices joint selection, returning names."""
    target_dof = int(dof) if dof else int(model.nu) or 7
    joint_ids = [
        i for i in range(model.njnt) if (model.joint(i).name or "").startswith("panda_joint")
    ]
    if len(joint_ids) < target_dof:
        seen: set[int] = set()
        joint_ids = []
        trnid = np.asarray(model.actuator_trnid, dtype=np.int32)
        for a in range(int(model.nu)):
            jid = int(trnid[a, 0])
            if 0 <= jid < model.njnt and jid not in seen and model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE:
                seen.add(jid)
                joint_ids.append(jid)
    if len(joint_ids) < target_dof:
        joint_ids = [i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
    if len(joint_ids) < target_dof:
        raise SystemExit(f"Model has only {len(joint_ids)} hinge joints; need {target_dof}.")
    return [model.joint(i).name for i in joint_ids[:target_dof]]


def _find_latest_run(root: Path) -> Optional[Path]:
    if not root.is_dir():
        return None
    candidates = sorted((d for d in root.iterdir() if d.is_dir()), reverse=True)
    for cand in candidates:
        if next(cand.rglob("target_log.npz"), None) is not None:
            return cand
    return None


def _resolve_iter_dir(run_dir: Path, iteration: Optional[int]) -> Path:
    iter_dirs = sorted(d for d in run_dir.glob("iter_*") if d.is_dir())
    if not iter_dirs:
        return run_dir
    if iteration is None:
        return iter_dirs[0]
    match = run_dir / f"iter_{int(iteration):03d}"
    if not match.is_dir():
        avail = ", ".join(d.name for d in iter_dirs)
        raise SystemExit(f"Iteration dir {match.name} not found; available: {avail}")
    return match


def _detect_conditions(iter_dir: Path) -> list[str]:
    found = [d.name for d in iter_dir.iterdir() if d.is_dir() and (d / "target_log.npz").is_file()]
    ordered = [k for k in PREFERRED_ORDER if k in found]
    ordered.extend(sorted(k for k in found if k not in PREFERRED_ORDER))
    return ordered


def _load_condition(iter_dir: Path, key: str) -> ConditionTrajectory:
    cond_dir = iter_dir / key
    src = np.load(cond_dir / "source_log.npz", allow_pickle=False)
    tgt = np.load(cond_dir / "target_log.npz", allow_pickle=False)

    src_t = np.asarray(src["t"], dtype=np.float64).reshape(-1)
    src_q = np.asarray(src["q"], dtype=np.float64)
    src_ee = np.asarray(src["ee_pos"], dtype=np.float64)
    tgt_t = np.asarray(tgt["t"], dtype=np.float64).reshape(-1)
    tgt_q = np.asarray(tgt["q"], dtype=np.float64)
    tgt_ee = np.asarray(tgt["ee_pos"], dtype=np.float64)

    # Source phase starts at t=0; the OSC phase log stores time relative to
    # the switch, so shift it behind the source phase.
    switch_t = float(src_t[-1]) if src_t.size else 0.0
    t = np.concatenate([src_t, switch_t + tgt_t])
    q = np.vstack([src_q, tgt_q])
    ee = np.vstack([src_ee, tgt_ee])
    ref = np.asarray(tgt["target_pos_ref"], dtype=np.float64) if "target_pos_ref" in tgt.files else np.zeros((0, 3))
    return ConditionTrajectory(key=key, t=t, q=q, ee_pos=ee, switch_t=switch_t, ref_ee_path=ref)


def _subsample(arr: np.ndarray, max_points: int) -> np.ndarray:
    if arr.shape[0] <= max_points:
        return arr
    idx = np.linspace(0, arr.shape[0] - 1, max_points).round().astype(np.int64)
    return arr[idx]


def _build_scene(
    xml_path: Path,
    trajectories: Sequence[ConditionTrajectory],
    spacing: float,
    ref_path: bool,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    base_spec = mujoco.MjSpec.from_file(str(xml_path))
    has_floor = any(g.type == mujoco.mjtGeom.mjGEOM_PLANE for g in base_spec.worldbody.geoms)

    parent = mujoco.MjSpec()
    parent.copy_during_attach = True
    parent.option.timestep = float(base_spec.option.timestep or 0.001)
    parent.visual.global_.offwidth = 2560
    parent.visual.global_.offheight = 1440
    parent.visual.headlight.ambient[:] = (0.35, 0.35, 0.35)
    parent.worldbody.add_light(pos=[0.5, 0.0, 3.0], dir=[0, 0, -1], castshadow=False)
    parent.worldbody.add_light(pos=[-2.0, 2.0, 2.5], dir=[0.5, -0.5, -0.7], castshadow=False)
    if not has_floor:
        floor = parent.worldbody.add_geom()
        floor.name = "replay_floor"
        floor.type = mujoco.mjtGeom.mjGEOM_PLANE
        floor.size = [0.0, 0.0, 0.05]
        floor.rgba = [0.22, 0.24, 0.28, 1.0]

    for i, traj in enumerate(trajectories):
        offset = np.array([0.0, i * spacing, 0.0])
        traj.base_offset = offset

        child = mujoco.MjSpec.from_file(str(xml_path))
        for key in list(child.keys):
            key.delete()
        frame = parent.worldbody.add_frame(pos=offset.tolist())
        parent.attach(child, prefix=f"{traj.key}/", frame=frame)

        # Colored base ring so each condition is identifiable at a glance.
        ring = parent.worldbody.add_geom()
        ring.name = f"{traj.key}/base_ring"
        ring.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        ring.size = [0.16, 0.004, 0.0]
        ring.pos = (offset + np.array([0.0, 0.0, 0.004])).tolist()
        ring.rgba = list(traj.color)
        ring.contype = 0
        ring.conaffinity = 0

        if ref_path and traj.ref_ee_path.shape[0] >= 2:
            for j, p in enumerate(_subsample(traj.ref_ee_path, 90)):
                dot = parent.worldbody.add_geom()
                dot.name = f"{traj.key}/ref_{j}"
                dot.type = mujoco.mjtGeom.mjGEOM_SPHERE
                dot.size = [0.0035, 0.0, 0.0]
                dot.pos = (offset + p).tolist()
                dot.rgba = list(REF_PATH_RGBA)
                dot.contype = 0
                dot.conaffinity = 0
            goal = parent.worldbody.add_geom()
            goal.name = f"{traj.key}/goal"
            goal.type = mujoco.mjtGeom.mjGEOM_SPHERE
            goal.size = [0.012, 0.0, 0.0]
            goal.pos = (offset + traj.ref_ee_path[-1]).tolist()
            goal.rgba = list(GOAL_RGBA)
            goal.contype = 0
            goal.conaffinity = 0

    model = parent.compile()
    data = mujoco.MjData(model)

    # Resolve per-robot qpos addresses via the single-robot joint names.
    single = mujoco.MjModel.from_xml_path(str(xml_path))
    dof = int(trajectories[0].q.shape[1])
    joint_names = _arm_joint_names(single, dof)
    for traj in trajectories:
        adr = []
        for name in joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{traj.key}/{name}")
            if jid < 0:
                raise SystemExit(f"Joint {traj.key}/{name} not found in combined model.")
            adr.append(int(model.jnt_qposadr[jid]))
        traj.qpos_adr = np.asarray(adr, dtype=np.int64)

    return model, data


def _set_pose(data: mujoco.MjData, trajectories: Sequence[ConditionTrajectory], sim_t: float) -> list[int]:
    frame_idx = []
    for traj in trajectories:
        idx = int(np.clip(np.searchsorted(traj.t, sim_t, side="right") - 1, 0, traj.t.shape[0] - 1))
        data.qpos[traj.qpos_adr] = traj.q[idx]
        frame_idx.append(idx)
    return frame_idx


def _init_sphere(geom: mujoco.MjvGeom, pos: np.ndarray, radius: float, rgba: Sequence[float]) -> None:
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0]),
        np.asarray(pos, dtype=np.float64),
        np.eye(3).reshape(9),
        np.asarray(rgba, dtype=np.float32),
    )


def _add_overlay_geoms(
    scn: mujoco.MjvScene,
    trajectories: Sequence[ConditionTrajectory],
    frame_idx: Sequence[int],
    sim_t: float,
    trail: bool,
    max_trail_points: int = 220,
) -> None:
    """Append per-condition EE trails and floating name labels to a scene."""
    for traj, idx in zip(trajectories, frame_idx):
        if trail and idx > 1:
            pts = _subsample(traj.ee_pos[: idx + 1], max_trail_points)
            for p in pts:
                if scn.ngeom >= scn.maxgeom:
                    break
                g = scn.geoms[scn.ngeom]
                _init_sphere(g, traj.base_offset + p, 0.004, traj.color)
                g.segid = -1
                scn.ngeom += 1
        if scn.ngeom < scn.maxgeom:
            g = scn.geoms[scn.ngeom]
            label_pos = traj.base_offset + np.array([0.0, 0.0, 1.05])
            _init_sphere(g, label_pos, 0.011, traj.color)
            phase = "warmup" if sim_t < traj.switch_t else "OSC"
            try:
                g.label = f"{traj.key} [{phase}]"
            except (AttributeError, TypeError, ValueError):
                pass
            scn.ngeom += 1


def _setup_camera(
    cam: mujoco.MjvCamera,
    trajectories: Sequence[ConditionTrajectory],
    azimuth: float,
    elevation: float,
    distance: Optional[float],
) -> None:
    n = len(trajectories)
    span = (n - 1) * float(np.linalg.norm(trajectories[1].base_offset - trajectories[0].base_offset)) if n > 1 else 0.0
    cam.lookat[:] = [0.25, span / 2.0, 0.45]
    cam.azimuth = azimuth
    cam.elevation = elevation
    cam.distance = distance if distance is not None else 2.0 + 0.65 * span


def _run_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    trajectories: Sequence[ConditionTrajectory],
    args: argparse.Namespace,
) -> None:
    import mujoco.viewer  # GLFW window; imported lazily so video mode stays headless

    total_t = max(float(traj.t[-1]) for traj in trajectories)
    state = {"paused": False, "rate": float(args.rate), "sim_t": 0.0, "loop": bool(args.loop), "seek": 0.0}

    def key_callback(keycode: int) -> None:
        if keycode == 32:  # SPACE
            state["paused"] = not state["paused"]
        elif keycode == 262:  # right arrow
            state["seek"] += 2.0
        elif keycode == 263:  # left arrow
            state["seek"] -= 2.0
        elif keycode == 265:  # up arrow
            state["rate"] = min(state["rate"] * 1.5, 16.0)
        elif keycode == 264:  # down arrow
            state["rate"] = max(state["rate"] / 1.5, 0.0625)
        elif keycode in (82, 114):  # R
            state["sim_t"] = 0.0
        elif keycode in (76, 108):  # L
            state["loop"] = not state["loop"]

    print(
        "[replay] 控制键: SPACE 暂停/继续 | ←/→ 快退/快进 2s | ↑/↓ 变速 | R 重播 | L 循环开关",
        flush=True,
    )
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        _setup_camera(viewer.cam, trajectories, args.azimuth, args.elevation, args.distance)
        last_wall = time.perf_counter()
        last_status = 0.0
        while viewer.is_running():
            now = time.perf_counter()
            dt_wall = now - last_wall
            last_wall = now
            if not state["paused"]:
                state["sim_t"] += dt_wall * state["rate"]
            if state["seek"]:
                state["sim_t"] = max(0.0, state["sim_t"] + state["seek"])
                state["seek"] = 0.0
            if state["sim_t"] > total_t:
                if state["loop"]:
                    state["sim_t"] = 0.0
                else:
                    state["sim_t"] = total_t
                    state["paused"] = True

            frame_idx = _set_pose(data, trajectories, state["sim_t"])
            mujoco.mj_forward(model, data)
            viewer.user_scn.ngeom = 0
            _add_overlay_geoms(viewer.user_scn, trajectories, frame_idx, state["sim_t"], bool(args.trail))
            viewer.sync()
            if now - last_status > 0.2:
                last_status = now
                sys.stdout.write(
                    f"\r[replay] t={state['sim_t']:7.2f}s / {total_t:.2f}s  speed={state['rate']:.2f}x"
                    f"  {'⏸' if state['paused'] else '▶'}  loop={'on' if state['loop'] else 'off'}   "
                )
                sys.stdout.flush()
            time.sleep(max(0.0, 1.0 / 60.0 - (time.perf_counter() - now)))
    print()


def _make_video_writer(path: Path, fps: int, width: int, height: int):
    try:
        import imageio.v2 as imageio

        writer = imageio.get_writer(str(path), fps=fps, macro_block_size=1)
        return writer.append_data, writer.close
    except Exception:
        import subprocess

        proc = subprocess.Popen(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
                "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
            ],
            stdin=subprocess.PIPE,
        )
        assert proc.stdin is not None
        return (lambda frame: proc.stdin.write(np.ascontiguousarray(frame).tobytes())), (
            lambda: (proc.stdin.close(), proc.wait())
        )


def _run_video(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    trajectories: Sequence[ConditionTrajectory],
    args: argparse.Namespace,
) -> None:
    total_t = max(float(traj.t[-1]) for traj in trajectories)
    fps = int(args.fps)
    n_frames = int(np.ceil(total_t * fps / float(args.rate))) + 1

    cam = mujoco.MjvCamera()
    _setup_camera(cam, trajectories, args.azimuth, args.elevation, args.distance)

    renderer = mujoco.Renderer(model, height=int(args.height), width=int(args.width), max_geom=5000)
    append_frame, close_writer = _make_video_writer(args.save_video, fps, int(args.width), int(args.height))
    print(f"[replay] 渲染 {n_frames} 帧 ({args.width}x{args.height} @ {fps}fps, 速度 {args.rate}x) ...", flush=True)
    try:
        for i in range(n_frames):
            sim_t = min(i / fps * float(args.rate), total_t)
            frame_idx = _set_pose(data, trajectories, sim_t)
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            _add_overlay_geoms(renderer.scene, trajectories, frame_idx, sim_t, bool(args.trail))
            append_frame(renderer.render())
            if i % (fps * 5) == 0:
                print(f"[replay]   {i}/{n_frames} 帧 (t={sim_t:.1f}s)", flush=True)
    finally:
        close_writer()
        renderer.close()
    print(f"[replay] 视频已保存: {args.save_video}", flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="评估 run 目录 (默认: eval_logs/source_to_osc_tam_sim 下最新的含 npz 轨迹的 run)。",
    )
    parser.add_argument("--iter", type=int, default=None, help="多 iteration run 中选择的 iteration 序号。")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help="要并排回放的 condition 目录名 (默认自动检测, 如 ideal_model direct_osc tam_carried)。",
    )
    parser.add_argument("--xml", type=Path, default=None, help="机器人 MJCF (默认从 run_config.json 读取)。")
    parser.add_argument("--spacing", type=float, default=1.2, help="并排机器人之间的间距 (米)。")
    parser.add_argument("--rate", type=float, default=1.0, help="回放速度倍率。")
    parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True, help="到结尾后循环播放。")
    parser.add_argument("--trail", action=argparse.BooleanOptionalAction, default=True, help="绘制末端执行器轨迹。")
    parser.add_argument(
        "--ref-path", action=argparse.BooleanOptionalAction, default=True, help="绘制 OSC 期望末端路径与目标点。"
    )
    parser.add_argument("--azimuth", type=float, default=140.0)
    parser.add_argument("--elevation", type=float, default=-20.0)
    parser.add_argument("--distance", type=float, default=None)
    parser.add_argument("--save-video", type=Path, default=None, help="不开交互窗口, 离线渲染并保存 mp4。")
    parser.add_argument("--fps", type=int, default=30, help="视频帧率 (--save-video 模式)。")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--list", action="store_true", help="列出可回放的 run / iteration / condition 后退出。")
    return parser


def _list_runs(root: Path) -> None:
    if not root.is_dir():
        print(f"[replay] 日志根目录不存在: {root}")
        return
    found_any = False
    for run in sorted(d for d in root.iterdir() if d.is_dir()):
        npz_dirs = sorted({p.parent for p in run.rglob("target_log.npz")})
        if not npz_dirs:
            continue
        found_any = True
        iters = sorted({d.parent.name for d in npz_dirs if d.parent.name.startswith("iter_")})
        conds = sorted({d.name for d in npz_dirs})
        iter_info = f" iterations: {', '.join(iters)};" if iters else ""
        print(f"  {run}:{iter_info} conditions: {', '.join(conds)}")
    if not found_any:
        print(f"[replay] {root} 下没有任何含 npz 轨迹的 run (batched 后端不存轨迹, 需要 --sim-backend legacy)。")


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)

    if args.list:
        _list_runs(DEFAULT_LOG_ROOT)
        return

    run_dir = args.run_dir
    if run_dir is None:
        run_dir = _find_latest_run(DEFAULT_LOG_ROOT)
        if run_dir is None:
            raise SystemExit(
                f"在 {DEFAULT_LOG_ROOT} 下找不到含轨迹 npz 的 run。"
                "batched 后端只写数值汇总; 请先用 --sim-backend legacy 重跑评估, "
                "或用 --run-dir 指定目录。(--list 可列出可用 run)"
            )
        print(f"[replay] 自动选择最新 run: {run_dir}", flush=True)
    run_dir = run_dir.expanduser().resolve()
    if next(run_dir.rglob("target_log.npz"), None) is None:
        raise SystemExit(
            f"{run_dir} 下没有 target_log.npz 轨迹文件 — 该 run 很可能是 batched 后端产物 "
            "(只有 summary 数值)。请用 --sim-backend legacy 重跑评估。"
        )

    iter_dir = _resolve_iter_dir(run_dir, args.iter)
    conditions = args.conditions or _detect_conditions(iter_dir)
    if not conditions:
        raise SystemExit(f"{iter_dir} 下未检测到任何含 target_log.npz 的 condition 目录。")

    xml_path = args.xml
    if xml_path is None:
        for cfg_dir in (run_dir, iter_dir.parent):
            cfg = cfg_dir / "run_config.json"
            if cfg.is_file():
                xml_value = json.loads(cfg.read_text(encoding="utf-8")).get("xml")
                if xml_value:
                    xml_path = Path(xml_value)
                    break
    if xml_path is None or not Path(xml_path).is_file():
        raise SystemExit("无法确定机器人 MJCF: run_config.json 缺失或 xml 不存在, 请用 --xml 指定。")
    xml_path = Path(xml_path).expanduser().resolve()

    trajectories: list[ConditionTrajectory] = []
    extra_color = iter(EXTRA_PALETTE)
    for key in conditions:
        if not (iter_dir / key / "target_log.npz").is_file():
            raise SystemExit(f"Condition 目录缺少轨迹: {iter_dir / key}")
        traj = _load_condition(iter_dir, key)
        traj.color = CONDITION_COLORS.get(key) or next(extra_color, (0.6, 0.6, 0.6, 1.0))
        trajectories.append(traj)

    print(f"[replay] run: {iter_dir}", flush=True)
    print(f"[replay] 模型: {xml_path}", flush=True)
    for traj in trajectories:
        rgb = ", ".join(f"{c:.2f}" for c in traj.color[:3])
        print(
            f"[replay]   {traj.key}: {traj.t.shape[0]} 帧, 总时长 {traj.t[-1]:.2f}s "
            f"(t={traj.switch_t:.2f}s 切换到 OSC), 颜色 rgb({rgb})",
            flush=True,
        )

    model, data = _build_scene(xml_path, trajectories, float(args.spacing), bool(args.ref_path))
    _set_pose(data, trajectories, 0.0)
    mujoco.mj_forward(model, data)

    if args.save_video is not None:
        args.save_video.parent.mkdir(parents=True, exist_ok=True)
        _run_video(model, data, trajectories, args)
    else:
        _run_viewer(model, data, trajectories, args)


if __name__ == "__main__":
    main()
