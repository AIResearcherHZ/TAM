"""One-page verdict report for source-to-OSC sim evaluations.

Reads a run directory produced by ``tam-eval-source-to-osc-sim`` and answers
the question the summary tables bury: does TAM help, and by how much? The
plant robot is simulated with perturbed dynamics, so the comparison that
matters is how close each condition stays to the ``ideal_model`` upper bound:

* ``direct_osc``  — uncompensated lower bound
* ``tam_carried`` — TAM compensating the torque residual

Works on both backends. CSV metrics (win rate, improvement) are always
available; the error-over-time and 3D end-effector plots additionally need
the npz trajectories that only the legacy backend writes.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

DEFAULT_LOG_ROOT = Path("eval_logs") / "source_to_osc_tam_sim"

# (csv key, label, scale to display unit, unit, lower is better assumed for all)
METRICS = (
    ("target_vs_ideal_ee_pos_rmse_m", "OSC段 EE轨迹偏离理想 RMSE", 1000.0, "mm"),
    ("target_vs_ideal_ee_final_error_m", "OSC段 终点位置偏离理想", 1000.0, "mm"),
    ("target_ee_pos_rmse_m", "OSC段 指令跟踪 RMSE", 1000.0, "mm"),
    ("source_vs_ideal_ee_pos_rmse_m", "热身段 EE轨迹偏离理想 RMSE", 1000.0, "mm"),
    ("source_vs_ideal_joint_pos_rmse_deg", "热身段 关节角偏离理想 RMSE", 1.0, "deg"),
)
PRIMARY_METRIC = "target_vs_ideal_ee_pos_rmse_m"
WARMUP_METRIC = "source_vs_ideal_ee_pos_rmse_m"

COLOR_BASE = "#d8543f"
COLOR_TREAT = "#3fa85a"
COLOR_IDEAL = "#7388bf"


def _maybe_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def _find_latest_run(root: Path) -> Optional[Path]:
    if not root.is_dir():
        return None
    for cand in sorted((d for d in root.iterdir() if d.is_dir()), reverse=True):
        if (cand / "summary.csv").is_file():
            return cand
    return None


def _load_rows(run_dir: Path) -> dict[int, dict[str, dict[str, Any]]]:
    """summary.csv → {iteration: {condition_key: row}}."""
    path = run_dir / "summary.csv"
    if not path.is_file():
        raise SystemExit(f"{run_dir} 下没有 summary.csv, 不是一个评估 run 目录。")
    by_iter: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            by_iter[int(row["iteration"])][str(row["condition_key"])] = row
    return dict(by_iter)


def _metric_value(row: dict[str, Any], key: str) -> Optional[float]:
    raw = row.get(key)
    if raw is None or raw == "":
        return None
    value = float(raw)
    return value if np.isfinite(value) else None


def _paired_stats(
    by_iter: dict[int, dict[str, dict[str, Any]]],
    baseline: str,
    treatment: str,
    key: str,
) -> Optional[dict[str, Any]]:
    base_vals, treat_vals, imps = [], [], []
    for conds in by_iter.values():
        if baseline not in conds or treatment not in conds:
            continue
        b = _metric_value(conds[baseline], key)
        t = _metric_value(conds[treatment], key)
        if b is None or t is None or b <= 0.0:
            continue
        base_vals.append(b)
        treat_vals.append(t)
        imps.append((b - t) / b * 100.0)
    if not imps:
        return None
    base = np.asarray(base_vals)
    treat = np.asarray(treat_vals)
    imps_arr = np.asarray(imps)
    return {
        "n": int(imps_arr.size),
        "base_mean": float(base.mean()),
        "base_std": float(base.std()),
        "treat_mean": float(treat.mean()),
        "treat_std": float(treat.std()),
        "wins": int(np.sum(treat < base)),
        "median_improvement_pct": float(np.median(imps_arr)),
        "mean_improvement_pct": float(imps_arr.mean()),
        "improvements_pct": imps_arr,
        "base_vals": base,
        "treat_vals": treat,
    }


def _verdict(primary: Optional[dict[str, Any]], warmup: Optional[dict[str, Any]]) -> tuple[str, str]:
    """Return (one-line verdict, qualifier)."""
    if primary is None:
        return "无法判定: 缺少可配对的 direct_osc / tam_carried 结果。", ""
    win_rate = primary["wins"] / primary["n"]
    med = primary["median_improvement_pct"]
    if win_rate >= 0.8 and med >= 30.0:
        verdict = "TAM 明显有效"
    elif win_rate >= 0.65 and med >= 10.0:
        verdict = "TAM 有效"
    elif win_rate >= 0.45:
        verdict = "效果不明显 (与直接 OSC 相当)"
    else:
        verdict = "TAM 反而变差"
    detail = (
        f"OSC 段 EE 偏离理想: {primary['n']} 轮配对中 TAM 胜 {primary['wins']} 轮, "
        f"误差中位数降低 {med:.0f}% "
        f"({primary['base_mean'] * 1000:.1f}mm → {primary['treat_mean'] * 1000:.1f}mm)"
    )
    if warmup is not None:
        detail += (
            f"; 热身段降低 {warmup['median_improvement_pct']:.0f}% "
            f"({warmup['base_mean'] * 1000:.1f}mm → {warmup['treat_mean'] * 1000:.1f}mm)"
        )
    qualifier = " [注意: 轮数较少, 结论参考性有限]" if primary["n"] < 5 else ""
    return verdict, detail + qualifier


def _iter_dirs_with_npz(run_dir: Path, baseline: str, treatment: str) -> list[Path]:
    candidates = sorted(d for d in run_dir.glob("iter_*") if d.is_dir()) or [run_dir]
    out = []
    for d in candidates:
        if all((d / key / "target_log.npz").is_file() for key in ("ideal_model", baseline, treatment)):
            out.append(d)
    return out


def _load_ee_timeline(cond_dir: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (t, ee_pos, switch_index) over warmup + OSC phases."""
    src = np.load(cond_dir / "source_log.npz", allow_pickle=False)
    tgt = np.load(cond_dir / "target_log.npz", allow_pickle=False)
    src_t = np.asarray(src["t"], dtype=np.float64).reshape(-1)
    offset = float(src_t[-1]) if src_t.size else 0.0
    t = np.concatenate([src_t, offset + np.asarray(tgt["t"], dtype=np.float64).reshape(-1)])
    ee = np.vstack([np.asarray(src["ee_pos"], dtype=np.float64), np.asarray(tgt["ee_pos"], dtype=np.float64)])
    return t, ee, int(src_t.size)


def _plot_paired(plt, stats_by_key, baseline: str, treatment: str, out_path: Path) -> None:
    keys = [k for k in (PRIMARY_METRIC, WARMUP_METRIC) if stats_by_key.get(k)]
    if not keys:
        return
    fig, axes = plt.subplots(len(keys), 1, figsize=(10, 4 * len(keys)), squeeze=False)
    titles = {
        PRIMARY_METRIC: "OSC phase: EE RMSE vs ideal model (lower is better)",
        WARMUP_METRIC: "Warmup phase: EE RMSE vs ideal model (lower is better)",
    }
    for ax, key in zip(axes.ravel(), keys):
        st = stats_by_key[key]
        idx = np.arange(st["n"])
        width = 0.38
        ax.bar(idx - width / 2, st["base_vals"] * 1000, width, label=baseline, color=COLOR_BASE)
        ax.bar(idx + width / 2, st["treat_vals"] * 1000, width, label=treatment, color=COLOR_TREAT)
        ax.set_title(titles[key])
        ax.set_xlabel("iteration")
        ax.set_ylabel("RMSE (mm)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_timeline(plt, iter_dirs, baseline: str, treatment: str, out_path: Path) -> None:
    curves: dict[str, list[np.ndarray]] = {baseline: [], treatment: []}
    t_axis: Optional[np.ndarray] = None
    switch_t = 0.0
    for d in iter_dirs:
        t_ideal, ee_ideal, switch_idx = _load_ee_timeline(d / "ideal_model")
        for key in (baseline, treatment):
            t_cond, ee_cond, _ = _load_ee_timeline(d / key)
            n = min(ee_ideal.shape[0], ee_cond.shape[0])
            curves[key].append(np.linalg.norm(ee_cond[:n] - ee_ideal[:n], axis=1) * 1000.0)
        if t_axis is None or t_ideal.size < t_axis.size:
            t_axis = t_ideal
            switch_t = float(t_ideal[min(switch_idx, t_ideal.size - 1)])
    if t_axis is None or not curves[baseline]:
        return
    n = min(min(c.size for c in curves[baseline]), min(c.size for c in curves[treatment]), t_axis.size)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for key, color in ((baseline, COLOR_BASE), (treatment, COLOR_TREAT)):
        arr = np.stack([c[:n] for c in curves[key]])
        mean = arr.mean(axis=0)
        ax.plot(t_axis[:n], mean, color=color, label=f"{key} (mean of {arr.shape[0]} iters)")
        if arr.shape[0] > 1:
            ax.fill_between(t_axis[:n], arr.min(axis=0), arr.max(axis=0), color=color, alpha=0.15)
    ax.axvline(switch_t, color="gray", linestyle="--", linewidth=1)
    ax.text(switch_t, ax.get_ylim()[1] * 0.95, " switch to OSC", color="gray", va="top")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("EE distance to ideal model (mm)")
    ax.set_title("How far each condition drifts from the ideal model over time")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_path3d(plt, iter_dir: Path, baseline: str, treatment: str, out_path: Path) -> None:
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(projection="3d")
    specs = (("ideal_model", COLOR_IDEAL), (baseline, COLOR_BASE), (treatment, COLOR_TREAT))
    ref = None
    for key, color in specs:
        tgt = np.load(iter_dir / key / "target_log.npz", allow_pickle=False)
        ee = np.asarray(tgt["ee_pos"], dtype=np.float64)
        ax.plot(ee[:, 0], ee[:, 1], ee[:, 2], color=color, label=key, linewidth=1.4)
        if ref is None and "target_pos_ref" in tgt.files:
            ref = np.asarray(tgt["target_pos_ref"], dtype=np.float64)
    if ref is not None:
        ax.plot(ref[:, 0], ref[:, 1], ref[:, 2], color="black", linestyle=":", linewidth=1.0, label="commanded path")
        ax.scatter(*ref[-1], color="#e6b822", s=45, marker="*", label="goal")
    ax.set_title(f"OSC-phase end-effector paths ({iter_dir.name})")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _format_table(stats_by_key, baseline: str, treatment: str) -> list[str]:
    lines = [
        f"| 指标 | {baseline} | {treatment} | TAM 胜率 | 误差中位数变化 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key, label, scale, unit in METRICS:
        st = stats_by_key.get(key)
        if st is None:
            continue
        lines.append(
            f"| {label} | {st['base_mean'] * scale:.1f} {unit} | {st['treat_mean'] * scale:.1f} {unit} "
            f"| {st['wins']}/{st['n']} | ↓{st['median_improvement_pct']:.0f}% |"
        )
    return lines


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "run_dir",
        type=Path,
        nargs="?",
        default=None,
        help="评估 run 目录 (默认: eval_logs/source_to_osc_tam_sim 下最新一次)。",
    )
    parser.add_argument("--baseline", default="direct_osc")
    parser.add_argument("--treatment", default="tam_carried")
    parser.add_argument("--no-plots", action="store_true", help="只输出文字结论, 不画图。")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)

    run_dir = args.run_dir
    if run_dir is None:
        run_dir = _find_latest_run(DEFAULT_LOG_ROOT)
        if run_dir is None:
            raise SystemExit(f"{DEFAULT_LOG_ROOT} 下找不到任何评估 run。")
    run_dir = run_dir.expanduser().resolve()

    by_iter = _load_rows(run_dir)
    stats_by_key = {
        key: _paired_stats(by_iter, args.baseline, args.treatment, key) for key, *_ in METRICS
    }
    verdict, detail = _verdict(stats_by_key.get(PRIMARY_METRIC), stats_by_key.get(WARMUP_METRIC))

    table_lines = _format_table(stats_by_key, args.baseline, args.treatment)

    plot_files: list[Path] = []
    plt = None if args.no_plots else _maybe_import_matplotlib()
    if plt is not None:
        paired_png = run_dir / "report_paired.png"
        _plot_paired(plt, stats_by_key, args.baseline, args.treatment, paired_png)
        if paired_png.is_file():
            plot_files.append(paired_png)
        iter_dirs = _iter_dirs_with_npz(run_dir, args.baseline, args.treatment)
        if iter_dirs:
            timeline_png = run_dir / "report_timeline.png"
            _plot_timeline(plt, iter_dirs, args.baseline, args.treatment, timeline_png)
            if timeline_png.is_file():
                plot_files.append(timeline_png)
            # Representative iteration for the 3D overlay: median improvement.
            primary = stats_by_key.get(PRIMARY_METRIC)
            rep = iter_dirs[0]
            if primary is not None and len(iter_dirs) > 1:
                imps = primary["improvements_pct"]
                rep = iter_dirs[int(np.argsort(imps)[imps.size // 2])] if imps.size == len(iter_dirs) else rep
            path3d_png = run_dir / "report_ee_path3d.png"
            _plot_path3d(plt, rep, args.baseline, args.treatment, path3d_png)
            if path3d_png.is_file():
                plot_files.append(path3d_png)

    report_lines = [
        f"# TAM Sim 评估报告 — {run_dir.name}",
        "",
        f"**结论: {verdict}**",
        "",
        detail,
        "",
        "实验设定: 被控机器人动力学被随机扰动 (模拟真机与标称模型失配)。`ideal_model` 为参数完美的",
        "上限, `direct_osc` 为不补偿的下限; 指标衡量各条件偏离上限的程度, 越小越好。",
        "",
        *table_lines,
        "",
    ]
    if plot_files:
        report_lines.append("## 图表")
        report_lines.append("")
        for p in plot_files:
            report_lines.append(f"![{p.stem}]({p.name})")
        report_lines.append("")
    elif not args.no_plots:
        report_lines.append("(本 run 无 npz 轨迹或缺少 matplotlib, 仅含 CSV 统计; legacy 后端可解锁时序/3D 图)")
        report_lines.append("")

    report_path = run_dir / "report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"\n{'=' * 64}")
    print(f"  结论: {verdict}")
    print(f"{'=' * 64}")
    print(f"  {detail}\n")
    for line in table_lines:
        print(f"  {line}")
    print(f"\n  报告: {report_path}")
    for p in plot_files:
        print(f"  图表: {p}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
