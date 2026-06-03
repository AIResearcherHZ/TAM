from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx

import simadaptor.physics.dynamics as dynamics
import simadaptor.physics.smoothing as smoothing_util
from simadaptor.deploy.runtime_common import (
    prepare_history_inputs,
    prepare_model_space_torque,
    zero_torque_history_keep_mask,
)


def _first_present(container: Any, keys: Sequence[str]) -> Tuple[Optional[Any], Optional[str]]:
    for k in keys:
        try:
            if isinstance(container, Mapping) and k in container:
                return container[k], k
            if hasattr(container, "__contains__") and k in container:
                return container[k], k
        except Exception:
            continue
    return None, None


def _finite_difference_vel(q: jax.Array, dt: float) -> jax.Array:
    q = jnp.asarray(q)
    dq = jnp.zeros_like(q)
    if q.shape[-2] < 3:
        return dq
    dq = dq.at[..., 1:-1, :].set((q[..., 2:, :] - q[..., :-2, :]) / (2.0 * dt))
    dq = dq.at[..., 0, :].set((-3.0 * q[..., 0, :] + 4.0 * q[..., 1, :] - q[..., 2, :]) / (2.0 * dt))
    dq = dq.at[..., -1, :].set((3.0 * q[..., -1, :] - 4.0 * q[..., -2, :] + q[..., -3, :]) / (2.0 * dt))
    return dq


def _last_history_embedding(history_emb: jax.Array) -> jax.Array:
    """Return last history token in a shape compatible with adaptor inputs.

    Accepted encoder outputs:
      - [C] -> [1, C]
      - [B, C] -> [B, C] (global history)
      - [B, N, C] -> [B, C] (time-token history)
      - [B, N, DoF, C] -> [B, DoF, C] (joint-wise history)
    """
    if history_emb.ndim == 1:
        return history_emb[None]
    if history_emb.ndim == 2:
        return history_emb
    if history_emb.ndim == 3:
        return history_emb[:, -1, :]
    if history_emb.ndim == 4:
        return history_emb[:, -1, :, :]
    raise ValueError(f"Unsupported history embedding rank: {history_emb.ndim} shape={history_emb.shape}")


def _cfg_value(cfg: Any, name: str, default: Any, *, legacy_name: Optional[str] = None) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, name):
        return getattr(cfg, name)
    sim_eval_cfg = getattr(cfg, "sim_eval", None)
    if sim_eval_cfg is not None and hasattr(sim_eval_cfg, name):
        return getattr(sim_eval_cfg, name)
    if legacy_name is not None and hasattr(cfg, legacy_name):
        return getattr(cfg, legacy_name)
    return default


def _fn_supports_keep_mask(fn: Any) -> bool:
    if fn is None:
        return False
    try:
        return "input_keep_mask" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class TauReconInputs:
    q: jax.Array  # (B, T, D)
    qd: jax.Array  # (B, T, D)
    tau: jax.Array  # (B, T, D)
    tau_hist: jax.Array  # (B, T, D)
    tau_grav: jax.Array  # (B, T, D)
    keep_mask: jax.Array  # (B, T, 1)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class TauReconResult:
    mae: jax.Array  # ()
    rmse: jax.Array  # ()
    per_joint_mae: jax.Array  # (D,)
    q_s: jax.Array  # (B, T', D)
    qd_s: jax.Array  # (B, T', D)
    qdd_s: jax.Array  # (B, T', D)
    tau_gt: jax.Array  # (B, T', D)
    tau_des: jax.Array  # (B, T', D)
    tau_pred: jax.Array  # (B, T', D)

    def tree_flatten(self):
        children = (
            self.mae,
            self.rmse,
            self.per_joint_mae,
            self.q_s,
            self.qd_s,
            self.qdd_s,
            self.tau_gt,
            self.tau_des,
            self.tau_pred,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


class TauReconstructionTester:
    """Fast tau reconstruction evaluation with cached JIT and local-fit smoothing."""

    def __init__(
        self,
        ideal_mjx_model: mjx.Model,
        history_apply_fn,
        adaptor_apply_fn,
        history_last_apply_fn=None,
        *,
        ideal_model_has_gravity: bool = True,
        external_force_body_id: int = -1,
        window_size: int,
        dt: float = 1e-3,
        apply_zero_torque_mask: bool = True,
        zero_torque_threshold: float = 1e-5,
        masked_fit_max_neighbors_each_side: int = 50,
        masked_fit_q_weight: float = 2.0,
        masked_fit_qd_weight: float = 1.0,
    ) -> None:
        self._ideal_mjx_model = ideal_mjx_model
        self._ideal_model_has_gravity: bool = bool(ideal_model_has_gravity)
        self._ideal_mjx_model_gc_on = ideal_mjx_model.replace(
            body_gravcomp=jnp.ones_like(ideal_mjx_model.body_gravcomp).at[..., 0].set(0)
        )
        self._ideal_mjx_model_gc_off = ideal_mjx_model.replace(
            body_gravcomp=jnp.zeros_like(ideal_mjx_model.body_gravcomp)
        )
        self._history_apply_fn = history_apply_fn
        self._history_last_apply_fn = history_last_apply_fn
        self._adaptor_apply_fn = adaptor_apply_fn
        self._history_apply_supports_keep_mask = _fn_supports_keep_mask(history_apply_fn)
        self._history_last_apply_supports_keep_mask = _fn_supports_keep_mask(history_last_apply_fn)
        self._external_force_body_id = int(external_force_body_id)

        self._dt = float(dt)
        self._window_size = max(int(window_size), 1)
        self._apply_zero_torque_mask = bool(apply_zero_torque_mask)
        self._zero_torque_threshold = float(zero_torque_threshold)
        self._masked_fit_max_neighbors_each_side = max(int(masked_fit_max_neighbors_each_side), 1)
        self._masked_fit_q_weight = float(masked_fit_q_weight)
        self._masked_fit_qd_weight = float(masked_fit_qd_weight)

        self._compute_jit = jax.jit(self._compute_impl, static_argnames=("stride", "history_max_steps"))

    def prepare(
        self,
        rollout_like: Any,
        *,
        tau_keys: Sequence[str] = ("tau_cmd", "tau_commanded", "tau", "u", "u_des"),
        tau_model_keys: Sequence[str] = ("tau_model", "tau_measured"),
        gravity_keys: Sequence[str] = ("gravity",),
        tau_hist_keys: Optional[Sequence[str]] = None,
    ) -> Optional[TauReconInputs]:
        q_raw, _ = _first_present(rollout_like, ["q"])
        if q_raw is None:
            return None
        qd_raw, _ = _first_present(rollout_like, ["qd", "dq"])
        tau_raw, tau_key = _first_present(rollout_like, list(tau_keys))
        tau_model_raw, tau_model_key = _first_present(rollout_like, list(tau_model_keys))
        tau_is_model_space = False
        if tau_raw is None:
            if tau_model_raw is None:
                return None
            tau_raw = tau_model_raw
            tau_key = tau_model_key
            tau_is_model_space = True
        tau_grav_raw, gravity_key = _first_present(rollout_like, list(gravity_keys))
        if (
            tau_grav_raw is None
            and self._ideal_model_has_gravity
            and not tau_is_model_space
        ):
            raise ValueError(
                "Real-log tau reconstruction requires logged gravity torque when "
                "ideal_model_has_gravity=True and the selected torque key is controller-space "
                f"({tau_key!r})."
            )
        print(
            "[tau-recon] Selected "
            f"tau_key={tau_key!r}, tau_is_model_space={tau_is_model_space}."
        )
        if tau_hist_keys is None:
            tau_hist_raw = tau_raw
            tau_hist_is_model_space = tau_is_model_space
        else:
            tau_hist_raw, tau_hist_key = _first_present(rollout_like, list(tau_hist_keys))
            if tau_hist_raw is None:
                return None
            tau_hist_is_model_space = tau_hist_key in set(tau_model_keys)

        q = jnp.asarray(np.asarray(q_raw))
        tau = jnp.asarray(np.asarray(tau_raw))
        tau_hist = jnp.asarray(np.asarray(tau_hist_raw))
        qd = jnp.asarray(np.asarray(qd_raw)) if qd_raw is not None else None
        if tau_grav_raw is None:
            tau_grav = jnp.zeros_like(tau)
        else:
            tau_grav_np = np.asarray(tau_grav_raw)
            if tau_grav_np.size == 0:
                raise ValueError(
                    f"Gravity torque key {gravity_key!r} is present but empty."
                )
            tau_grav = jnp.asarray(tau_grav_np)
            print(f"[tau-recon] Using logged gravity torque from key={gravity_key!r}.")

        if q.ndim == 2:
            q = q[None]
        if tau.ndim == 2:
            tau = tau[None]
        if tau_hist.ndim == 2:
            tau_hist = tau_hist[None]
        if qd is not None and qd.ndim == 2:
            qd = qd[None]
        if tau_grav.ndim == 2:
            tau_grav = tau_grav[None]
        if q.ndim != 3 or tau.ndim != 3 or tau_hist.ndim != 3:
            raise ValueError(
                f"Expected q/tau shaped (B,T,D) or (T,D); got q={q.shape}, tau={tau.shape}, tau_hist={tau_hist.shape}"
            )
        if tau_grav.ndim != 3:
            raise ValueError(
                f"Gravity torque key {gravity_key!r} has unsupported shape {tau_grav.shape}; "
                "expected (T,D) or (B,T,D)."
            )

        T_all = min(q.shape[1], tau.shape[1], tau_hist.shape[1])
        if qd is not None:
            T_all = min(T_all, qd.shape[1])
        T_all = min(T_all, tau_grav.shape[1])
        if T_all < 2:
            return None
        q = q[:, :T_all]
        tau = tau[:, :T_all]
        tau_hist = tau_hist[:, :T_all]
        tau_grav = tau_grav[:, :T_all]
        if qd is None:
            qd = _finite_difference_vel(q, self._dt)
        else:
            qd = qd[:, :T_all]

        common_dof = min(
            q.shape[-1],
            tau.shape[-1],
            tau_hist.shape[-1],
            qd.shape[-1],
            self._ideal_mjx_model.nu,
            self._ideal_mjx_model.nv,
        )
        common_dof = min(common_dof, tau_grav.shape[-1])
        q = q[..., :common_dof].astype(jnp.float32)
        qd = qd[..., :common_dof].astype(jnp.float32)
        tau = tau[..., :common_dof].astype(jnp.float32)
        tau_hist = tau_hist[..., :common_dof].astype(jnp.float32)
        tau_grav = tau_grav[..., :common_dof].astype(jnp.float32)
        tau_model = prepare_model_space_torque(
            tau,
            gravity=tau_grav,
            ideal_model_has_gravity=self._ideal_model_has_gravity,
            context="TauReconstructionTester.prepare.target",
            tau_is_model_space=tau_is_model_space,
        )
        if tau_hist_is_model_space:
            tau_grav_for_hist = jnp.zeros_like(tau_grav)
        else:
            tau_grav_for_hist = tau_grav
        keep_mask = zero_torque_history_keep_mask(
            tau_hist,
            threshold=self._zero_torque_threshold,
        ).astype(jnp.float32)[..., None]
        return TauReconInputs(
            q=q,
            qd=qd,
            tau=tau_model.astype(jnp.float32),
            tau_hist=tau_hist,
            tau_grav=tau_grav_for_hist,
            keep_mask=keep_mask,
        )

    def warmup(
        self,
        rng: jax.Array,
        *,
        params: Mapping[str, Any],
        norm_stats: Any,
        inputs: TauReconInputs,
        history_inputs: Optional[TauReconInputs] = None,
        stride: int = 1,
        history_max_steps: Optional[int] = None,
    ) -> None:
        hist_inputs = inputs if history_inputs is None else history_inputs
        res = self._compute_jit(
            rng,
            params,
            norm_stats,
            inputs.q,
            inputs.qd,
            inputs.tau,
            inputs.tau_hist,
            inputs.tau_grav,
            inputs.keep_mask,
            hist_inputs.q,
            hist_inputs.qd,
            hist_inputs.tau_hist,
            hist_inputs.tau_grav,
            hist_inputs.keep_mask,
            stride=stride,
            history_max_steps=history_max_steps,
        )
        jax.block_until_ready(res.mae)

    def compute(
        self,
        rng: jax.Array,
        *,
        params: Mapping[str, Any],
        norm_stats: Any,
        inputs: TauReconInputs,
        history_inputs: Optional[TauReconInputs] = None,
        stride: int = 1,
        history_max_steps: Optional[int] = None,
    ) -> TauReconResult:
        hist_inputs = inputs if history_inputs is None else history_inputs
        return self._compute_jit(
            rng,
            params,
            norm_stats,
            inputs.q,
            inputs.qd,
            inputs.tau,
            inputs.tau_hist,
            inputs.tau_grav,
            inputs.keep_mask,
            hist_inputs.q,
            hist_inputs.qd,
            hist_inputs.tau_hist,
            hist_inputs.tau_grav,
            hist_inputs.keep_mask,
            stride=stride,
            history_max_steps=history_max_steps,
        )

    def run(
        self,
        rng: jax.Array,
        rollout_like: Any,
        *,
        params: Mapping[str, Any],
        norm_stats: Any,
        out_dir: Optional[str] = None,
        prefix: str = "tau_recon",
        cfg: Optional[Any] = None,
    ) -> tuple[dict[str, float], Optional[str]]:
        inputs = rollout_like if isinstance(rollout_like, TauReconInputs) else self.prepare(rollout_like)
        if (
            inputs is None
            or (self._history_apply_fn is None and self._history_last_apply_fn is None)
            or self._adaptor_apply_fn is None
        ):
            return {}, None

        stride = int(_cfg_value(cfg, "real_recon_stride", 1, legacy_name="tau_recon_stride") or 1)
        history_max_steps = _cfg_value(
            cfg,
            "real_recon_hist_max_steps",
            None,
            legacy_name="tau_recon_hist_max_steps",
        )
        result = self.compute(
            rng,
            params=params,
            norm_stats=norm_stats,
            inputs=inputs,
            stride=stride,
            history_max_steps=history_max_steps,
        )

        mae = float(jax.device_get(result.mae))
        rmse = float(jax.device_get(result.rmse))
        per_joint_mae = np.asarray(jax.device_get(result.per_joint_mae))
        metrics: dict[str, float] = {
            "tau_recon/mae": mae,
            "tau_recon/rmse": rmse,
            "tau_recon/model_tau_absmean": float(
                np.mean(np.abs(np.asarray(jax.device_get(inputs.tau))))
            ),
            "tau_recon/history_gravity_absmean": float(
                np.mean(np.abs(np.asarray(jax.device_get(inputs.tau_grav))))
            ),
        }
        for j, v in enumerate(per_joint_mae.tolist()):
            metrics[f"tau_recon/mae_j{j}"] = float(v)

        plot_path = None
        want_plot = bool(_cfg_value(cfg, "real_recon_plot", True, legacy_name="tau_recon_plot"))
        if out_dir is not None and want_plot:
            plot_stride = int(
                _cfg_value(
                    cfg,
                    "real_recon_plot_stride",
                    1,
                    legacy_name="tau_recon_plot_stride",
                )
                or 1
            )
            max_points = int(
                _cfg_value(
                    cfg,
                    "real_recon_plot_max_points",
                    2000,
                    legacy_name="tau_recon_plot_max_points",
                )
                or 2000
            )
            plot_path = self._plot_timeseries(
                out_dir=Path(out_dir),
                prefix=prefix,
                tau_gt=np.asarray(jax.device_get(result.tau_gt[0])),
                tau_des=np.asarray(jax.device_get(result.tau_des[0])),
                tau_pred=np.asarray(jax.device_get(result.tau_pred[0])),
                stride=plot_stride,
                max_points=max_points,
            )

        return metrics, plot_path

    def plot_joint_sweep_from_real_logs(
        self,
        *,
        params: Mapping[str, Any],
        norm_stats: Any,
        out_dir: str | Path,
        act_log_path: str | Path,
        rest_log_path: str | Path,
        prefix: str = "rest_real",
        seed: int = 0,
        tau_min: float = -1.5,
        tau_max: float = 1.5,
        num_tau_samples: int = 21,
    ) -> Optional[str]:
        if (self._history_apply_fn is None and self._history_last_apply_fn is None) or self._adaptor_apply_fn is None:
            return None
        try:
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"[tau-recon] Matplotlib unavailable: {e}")
            return None

        act_path = Path(act_log_path)
        rest_path = Path(rest_log_path)
        if not (act_path.exists() and rest_path.exists()):
            print(f"[tau-recon] Missing activation/rest logs: {act_path} or {rest_path}")
            return None

        act = np.load(act_path, allow_pickle=True)
        rest = np.load(rest_path, allow_pickle=True)
        q_act = np.asarray(act["q"], dtype=np.float32)
        qd_act = np.asarray(act["qd"] if "qd" in act else act["dq"], dtype=np.float32)
        u_act = np.asarray(act["tau_cmd"], dtype=np.float32)
        grav_act = np.asarray(act["gravity"], dtype=np.float32) if "gravity" in act else None

        rng = jax.random.PRNGKey(int(seed))
        q_hist_act, qd_hist_act, tau_hist_act, keep_hist_act = prepare_history_inputs(
            q_act[None],
            qd_act[None],
            u_act[None],
            gravity=None if grav_act is None else grav_act[None],
            ideal_model_has_gravity=self._ideal_model_has_gravity,
            context="TauReconstructionTester.plot_joint_sweep_from_real_logs",
            apply_zero_torque_mask=self._apply_zero_torque_mask,
            threshold=self._zero_torque_threshold,
        )
        keep_hist_act = keep_hist_act[..., None]
        if self._history_last_apply_fn is not None:
            if self._history_last_apply_supports_keep_mask:
                history_last = self._history_last_apply_fn(
                    params["hist"],
                    q_hist_act,
                    qd_hist_act,
                    tau_hist_act,
                    keep_hist_act,
                    rng,
                    True,
                    norm_stats,
                )
            else:
                history_last = self._history_last_apply_fn(
                    params["hist"], q_hist_act, qd_hist_act, tau_hist_act, rng, True, norm_stats
                )
        else:
            if self._history_apply_supports_keep_mask:
                history_emb = self._history_apply_fn(
                    params["hist"],
                    q_hist_act,
                    qd_hist_act,
                    tau_hist_act,
                    keep_hist_act,
                    rng,
                    True,
                    norm_stats,
                )
            else:
                history_emb = self._history_apply_fn(
                    params["hist"], q_hist_act, qd_hist_act, tau_hist_act, rng, True, norm_stats
                )
            history_last = _last_history_embedding(history_emb)

        q_full = np.asarray(rest["q"], dtype=np.float32)
        qd_full = np.asarray(
            rest["qd"] if "qd" in rest else rest["dq"] if "dq" in rest else np.gradient(q_full, self._dt, axis=0),
            dtype=np.float32,
        )
        tau_full = np.asarray(rest["tau_cmd"], dtype=np.float32)
        keep_full = jnp.ones((1, q_full.shape[0], 1), dtype=jnp.float32)
        q_s, qd_s, _ = smoothing_util.estimate_masked_state_derivatives(
            jnp.asarray(q_full[None], dtype=jnp.float32),
            jnp.asarray(qd_full[None], dtype=jnp.float32),
            keep_full,
            base_dt=self._dt,
            max_neighbors_each_side=self._masked_fit_max_neighbors_each_side,
            q_weight=self._masked_fit_q_weight,
            qd_weight=self._masked_fit_qd_weight,
        )
        q_s = q_s[0].astype(jnp.float32)
        qd_s = qd_s[0].astype(jnp.float32)

        start = 1000
        end = min(3000, int(q_s.shape[0]))
        q_slice = q_s[start:end]
        qd_slice = qd_s[start:end]
        tau_slice = jnp.asarray(tau_full[start:end], dtype=jnp.float32)
        if q_slice.shape[0] < 2:
            q_slice = q_s
            qd_slice = qd_s
            tau_slice = jnp.asarray(tau_full, dtype=jnp.float32)

        adaptor_T = self._window_size
        start_idx = 10
        end_idx = start_idx + adaptor_T
        if q_slice.shape[0] < end_idx:
            start_idx = max(0, int(q_slice.shape[0]) - adaptor_T)
            end_idx = int(q_slice.shape[0])

        def _window(x):
            x = x[start_idx:end_idx]
            if x.shape[0] < adaptor_T:
                pad_len = adaptor_T - x.shape[0]
                pad = jnp.repeat(x[-1:], pad_len, axis=0)
                x = jnp.concatenate([x, pad], axis=0)
            return x

        q_window = _window(q_slice)
        qd_window = _window(qd_slice)
        tau_window = _window(tau_slice)
        ndof = int(q_window.shape[-1])
        if ndof < 1:
            return None

        tau_axis = jnp.linspace(float(tau_min), float(tau_max), int(num_tau_samples))
        tau_des_all = []
        tau_adapted_all = []

        for j_idx in range(ndof):
            tau_des_sweep = jnp.zeros((int(num_tau_samples), ndof), dtype=jnp.float32).at[:, j_idx].set(tau_axis)
            q_in = jnp.tile(q_window[None], (int(num_tau_samples), 1, 1))
            qd_in = jnp.tile(qd_window[None], (int(num_tau_samples), 1, 1))
            tau_in = jnp.tile(tau_window[None], (int(num_tau_samples), 1, 1)).at[:, -1, :].set(tau_des_sweep)
            hist_in = jnp.repeat(history_last, int(num_tau_samples), axis=0)
            delta_tau, _ = self._adaptor_apply_fn(
                params["adaptor"],
                q_in,
                qd_in,
                tau_in,
                hist_in,
                jax.random.PRNGKey(int(seed) + j_idx + 1),
                False,
                norm_stats,
            )
            tau_out = tau_in[..., -1, :] + delta_tau
            tau_des_all.append(np.asarray(jax.device_get(tau_des_sweep[:, j_idx])))
            tau_adapted_all.append(np.asarray(jax.device_get(tau_out[:, j_idx])))

        tau_des_all = np.stack(tau_des_all, axis=0)
        tau_adapted_all = np.stack(tau_adapted_all, axis=0)

        fig, axes = plt.subplots(ndof, 1, figsize=(8, 2.2 * ndof), sharex=True)
        axes = np.atleast_1d(axes)
        for j_idx, ax in enumerate(axes):
            ax.plot(np.asarray(tau_axis), tau_des_all[j_idx], "k--", label="Desired")
            ax.plot(np.asarray(tau_axis), tau_adapted_all[j_idx], "b-o", label="Adaptor")
            ax.set_ylabel(f"Joint {j_idx+1} torque (Nm)")
            ax.grid(True, linestyle="--", alpha=0.4)
            if j_idx == 0:
                ax.legend()
        axes[-1].set_xlabel("Desired torque sweep (Nm)")
        fig.tight_layout()

        out_dir_p = Path(out_dir)
        out_dir_p.mkdir(parents=True, exist_ok=True)
        plot_path = out_dir_p / f"{prefix}_torque_sweep_per_joint.png"
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)
        return str(plot_path)

    def _plot_timeseries(
        self,
        *,
        out_dir: Path,
        prefix: str,
        tau_gt: np.ndarray,
        tau_des: np.ndarray,
        tau_pred: np.ndarray,
        stride: int,
        max_points: int,
    ) -> str:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        out_dir.mkdir(parents=True, exist_ok=True)
        stride = max(int(stride), 1)
        tau_gt = np.asarray(tau_gt)
        tau_des = np.asarray(tau_des)
        tau_pred = np.asarray(tau_pred)

        T = min(tau_gt.shape[0], tau_des.shape[0], tau_pred.shape[0])
        if max_points and T > max_points:
            stride = max(stride, int(math.ceil(T / max_points)))
        sl = slice(0, T, stride)
        tau_gt = tau_gt[sl]
        tau_des = tau_des[sl]
        tau_pred = tau_pred[sl]

        ndof = int(tau_gt.shape[-1])
        fig, axes = plt.subplots(ndof, 1, figsize=(10, 2.0 * ndof), sharex=True)
        axes = np.atleast_1d(axes)
        x = np.arange(tau_gt.shape[0])
        for j, ax in enumerate(axes):
            ax.plot(x, tau_gt[:, j], "k", lw=1.2, label="tau_log_model")
            ax.plot(x, tau_des[:, j], "g--", lw=1.0, label="tau_id_model")
            ax.plot(x, tau_pred[:, j], "b", lw=1.2, label="tau_pred_model")
            ax.set_ylabel(f"j{j}")
            ax.grid(True, linestyle="--", alpha=0.3)
            if j == 0:
                ax.legend(ncol=3, fontsize=9)
        axes[-1].set_xlabel("t (downsampled)")
        fig.tight_layout()

        plot_path = out_dir / f"{prefix}_tau_reconstruction.png"
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)
        return str(plot_path)

    def _compute_impl(
        self,
        rng: jax.Array,
        params: Mapping[str, Any],
        norm_stats: Any,
        q: jax.Array,
        qd: jax.Array,
        tau: jax.Array,
        tau_hist: jax.Array,
        tau_grav: jax.Array,
        keep_mask: jax.Array,
        history_q: jax.Array,
        history_qd: jax.Array,
        history_tau_hist: jax.Array,
        history_tau_grav: jax.Array,
        history_keep_mask: jax.Array,
        *,
        stride: int,
        history_max_steps: Optional[int],
    ) -> TauReconResult:
        stride = max(int(stride), 1)

        if history_q.shape[0] != q.shape[0]:
            raise ValueError(
                "history_inputs batch size must match inputs batch size; "
                f"got history_q={history_q.shape}, q={q.shape}"
            )
        if history_q.shape[-1] != q.shape[-1]:
            raise ValueError(
                "history_inputs dof must match inputs dof; "
                f"got history_q={history_q.shape}, q={q.shape}"
            )

        # Use only the first half of the provided history trajectory as encoder context.
        T_history = int(history_q.shape[1])
        history_split = max(1, T_history // 2)
        history_q_half = history_q[:, :history_split, :]
        history_qd_half = history_qd[:, :history_split, :]
        history_tau_hist_half = history_tau_hist[:, :history_split, :]
        history_tau_grav_half = history_tau_grav[:, :history_split, :]
        history_keep_half = history_keep_mask[:, :history_split, :]

        # Optionally shorten the history encoder context further (still uses the last embedding).
        if history_max_steps is not None:
            history_max_steps = int(history_max_steps)
            q_hist = history_q_half[:, -history_max_steps:, :]
            qd_hist = history_qd_half[:, -history_max_steps:, :]
            tau_hist_seg = history_tau_hist_half[:, -history_max_steps:, :]
            tau_grav_hist_seg = history_tau_grav_half[:, -history_max_steps:, :]
            keep_hist = history_keep_half[:, -history_max_steps:, :]
        else:
            q_hist, qd_hist, tau_hist_seg = history_q_half, history_qd_half, history_tau_hist_half
            tau_grav_hist_seg = history_tau_grav_half
            keep_hist = history_keep_half

        q_hist_prepared, qd_hist_prepared, tau_hist_model, keep_hist_prepared = prepare_history_inputs(
            q_hist,
            qd_hist,
            tau_hist_seg,
            gravity=tau_grav_hist_seg,
            ideal_model_has_gravity=self._ideal_model_has_gravity,
            context="TauReconstructionTester.history",
            apply_zero_torque_mask=self._apply_zero_torque_mask,
            threshold=self._zero_torque_threshold,
        )
        if not self._apply_zero_torque_mask:
            keep_hist_prepared = jnp.asarray(keep_hist, dtype=jnp.float32)

        # `tau` is prepared in model-space before JIT entry. For real robot logs
        # with gc-off training this is controller tau_cmd + logged gravity FF;
        # for simulator logs it is the already model-space command stream.
        tau_model = tau

        rng, sub = jax.random.split(rng)
        if self._history_last_apply_fn is not None:
            if self._history_last_apply_supports_keep_mask:
                hist_last = self._history_last_apply_fn(
                    params["hist"],
                    q_hist_prepared,
                    qd_hist_prepared,
                    tau_hist_model,
                    keep_hist_prepared,
                    sub,
                    True,
                    norm_stats,
                )
            else:
                hist_last = self._history_last_apply_fn(
                    params["hist"], q_hist_prepared, qd_hist_prepared, tau_hist_model, sub, True, norm_stats
                )
        else:
            if self._history_apply_supports_keep_mask:
                hist_emb = self._history_apply_fn(
                    params["hist"],
                    q_hist_prepared,
                    qd_hist_prepared,
                    tau_hist_model,
                    keep_hist_prepared,
                    sub,
                    True,
                    norm_stats,
                )
            else:
                hist_emb = self._history_apply_fn(
                    params["hist"], q_hist_prepared, qd_hist_prepared, tau_hist_model, sub, True, norm_stats
                )
            hist_last = _last_history_embedding(hist_emb)

        keep_fit = jnp.asarray(keep_mask, dtype=jnp.float32)
        if not self._apply_zero_torque_mask:
            keep_fit = jnp.ones_like(keep_fit, dtype=jnp.float32)
        q_fit_src = jnp.where(keep_fit > 0.0, q, 0.0)
        qd_fit_src = jnp.where(keep_fit > 0.0, qd, 0.0)
        q_s, qd_s, qdd_s = smoothing_util.estimate_masked_state_derivatives(
            q_fit_src,
            qd_fit_src,
            keep_fit,
            base_dt=self._dt,
            max_neighbors_each_side=self._masked_fit_max_neighbors_each_side,
            q_weight=self._masked_fit_q_weight,
            qd_weight=self._masked_fit_qd_weight,
        )
        q_s = q_s.astype(q.dtype)
        qd_s = qd_s.astype(q.dtype)
        qdd_s = qdd_s.astype(q.dtype)
        tau_s = tau_model

        if stride != 1:
            q_s = q_s[:, ::stride, :]
            qd_s = qd_s[:, ::stride, :]
            qdd_s = qdd_s[:, ::stride, :]
            tau_s = tau_s[:, ::stride, :]

        # Ideal inverse dynamics for each timestep.
        # Two-mode formulation:
        #   - ideal_model_has_gravity=True  -> gc-off ID model.
        #   - ideal_model_has_gravity=False -> gc-on ID model.
        id_model = self._ideal_mjx_model_gc_off if self._ideal_model_has_gravity else self._ideal_mjx_model_gc_on
        flat_q = q_s.reshape((-1, q_s.shape[-1]))
        flat_qd = qd_s.reshape((-1, qd_s.shape[-1]))
        flat_qdd = qdd_s.reshape((-1, qdd_s.shape[-1]))
        tau_des_flat = jax.vmap(
            partial(dynamics.mjx_inverse_dynamics_rne, id_model)
        )(flat_q, flat_qd, flat_qdd)
        tau_des_full = tau_des_flat.reshape(q_s.shape)
        # Training-aligned reconstruction space:
        #   tau_des      = ID_selected_model(q, qd, qdd)
        #   tau_gt_model = model-space log command
        tau_des = tau_des_full
        tau_gt_model = tau_s

        win = self._window_size

        def _build_windows(x: jax.Array) -> jax.Array:
            if win == 1:
                return x[:, :, None, :]
            pad = jnp.repeat(x[:, :1, :], win - 1, axis=1)
            x_pad = jnp.concatenate([pad, x], axis=1)  # (B, T+win-1, D)

            def _slice(t):
                return jax.lax.dynamic_slice_in_dim(x_pad, t, win, axis=1)  # (B, win, D)

            windows = jax.vmap(_slice)(jnp.arange(x.shape[1]))  # (T, B, win, D)
            return jnp.swapaxes(windows, 0, 1)  # (B, T, win, D)

        q_win = _build_windows(q_s)
        qd_win = _build_windows(qd_s)
        tau_win = _build_windows(tau_gt_model).at[:, :, -1, :].set(tau_des)

        B, T, _, D = q_win.shape
        q_in = q_win.reshape((B * T, win, D))
        qd_in = qd_win.reshape((B * T, win, D))
        tau_in = tau_win.reshape((B * T, win, D))
        hist_in = jnp.repeat(hist_last, T, axis=0)

        rng, sub = jax.random.split(rng)
        delta_tau, _ = self._adaptor_apply_fn(params["adaptor"], q_in, qd_in, tau_in, hist_in, sub, False, norm_stats)
        tau_pred_flat = tau_in[:, -1, :] + delta_tau  # (B*T, D)
        tau_pred = tau_pred_flat.reshape((B, T, D))

        err = tau_pred - tau_gt_model
        mae = jnp.mean(jnp.abs(err))
        rmse = jnp.sqrt(jnp.mean(err * err))
        per_joint_mae = jnp.mean(jnp.abs(err), axis=(0, 1))
        return TauReconResult(
            mae=mae,
            rmse=rmse,
            per_joint_mae=per_joint_mae,
            q_s=q_s,
            qd_s=qd_s,
            qdd_s=qdd_s,
            tau_gt=tau_gt_model,
            tau_des=tau_des,
            tau_pred=tau_pred,
        )
