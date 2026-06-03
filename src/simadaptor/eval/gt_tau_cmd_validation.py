from mujoco import mjx

import jax
import jax.numpy as jnp
from typing import Literal

import simadaptor.core.structs as structs
import simadaptor.physics.actuator as actuator_util


GTTauCmdMethod = Literal["shared_linear", "reference"]


def _compute_gt_tau_cmd_reference(
    mjx_model_ideal: mjx.Model,
    rollout_params: structs.RolloutParams,
    q_cur: jax.Array,
    qd_cur: jax.Array,
    tau_ref: jax.Array,
    external_force_ee: jax.Array | None,
    *,
    external_force_body_id: int | jax.Array,
) -> jax.Array:
    """Old reference solve used by training before the helper refactor."""
    return actuator_util.calculate_gt_torque(
        mjx_model_ideal,
        q=q_cur,
        qd=qd_cur,
        mjx_model_ideal=mjx_model_ideal,
        tau=tau_ref,
        actuator_params=rollout_params,
        external_force_ee=external_force_ee,
        external_force_body_id=external_force_body_id,
    )


def _compute_gt_tau_cmd_single(
    mjx_model_ideal: mjx.Model,
    rollout_params: structs.RolloutParams,
    q_cur: jax.Array,
    qd_cur: jax.Array,
    tau_ref: jax.Array,
    *,
    method: GTTauCmdMethod,
    external_force_ee: jax.Array | None,
    external_force_body_id: int | jax.Array,
) -> jax.Array:
    if method == "reference":
        if tau_ref.ndim == 2:
            return jax.vmap(
                lambda tau_ref_i: _compute_gt_tau_cmd_reference(
                    mjx_model_ideal,
                    rollout_params,
                    q_cur,
                    qd_cur,
                    tau_ref_i,
                    external_force_ee,
                    external_force_body_id=external_force_body_id,
                )
            )(tau_ref)
        return _compute_gt_tau_cmd_reference(
            mjx_model_ideal,
            rollout_params,
            q_cur,
            qd_cur,
            tau_ref,
            external_force_ee,
            external_force_body_id=external_force_body_id,
        )
    if method == "shared_linear":
        return actuator_util.calculate_gt_torque_shared_linear(
            mjx_model_ideal,
            q=q_cur,
            qd=qd_cur,
            mjx_model_ideal=mjx_model_ideal,
            tau=tau_ref,
            actuator_params=rollout_params,
            external_force_ee=external_force_ee,
            external_force_body_id=external_force_body_id,
        )
    raise ValueError(f"Unsupported compute_gt_tau_cmd method: {method!r}")


def compute_gt_tau_cmd(
    mjx_model_ideal: mjx.Model,
    rollout_params: structs.RolloutParams,
    q_cur: jax.Array,
    qd_cur: jax.Array,
    tau_ref: jax.Array,
    *,
    external_force_ee: jax.Array | None = None,
    external_force_body_id: int | jax.Array = -1,
    method: GTTauCmdMethod = "reference",
) -> jax.Array:
    """Return the commanded torque that matches the ideal-model torque effect.

    Supported shapes:
      - `q_cur`, `qd_cur`: `[D]` with `tau_ref` shaped `[D]` or `[S, D]`
      - `q_cur`, `qd_cur`: `[B, D]` with `tau_ref` shaped `[B, D]` or `[B, S, D]`
    """
    method = str(method)
    q_cur = jnp.asarray(q_cur)
    qd_cur = jnp.asarray(qd_cur)
    tau_ref = jnp.asarray(tau_ref)
    force = None if external_force_ee is None else jnp.asarray(external_force_ee)
    body_id = jnp.asarray(
        -1 if external_force_body_id is None else external_force_body_id,
        dtype=jnp.int32,
    )

    if q_cur.ndim != qd_cur.ndim:
        raise ValueError(f"q_cur and qd_cur rank mismatch: q_cur={q_cur.shape}, qd_cur={qd_cur.shape}.")

    if q_cur.ndim == 1:
        if tau_ref.ndim not in (1, 2):
            raise ValueError(
                f"Expected tau_ref rank 1 or 2 for a single state; got tau_ref={tau_ref.shape}."
            )
        if force is not None:
            if force.ndim != 1:
                raise ValueError(
                    f"Expected external_force_ee rank 1 for a single state; got external_force_ee={force.shape}."
                )
            if force.shape[-1] not in (3, 6):
                raise ValueError(
                    f"external_force_ee must have trailing size 3 or 6; got {force.shape}."
                )
        if body_id.ndim != 0:
            raise ValueError(
                f"Expected scalar external_force_body_id for a single state; got {body_id.shape}."
            )
        return _compute_gt_tau_cmd_single(
            mjx_model_ideal,
            rollout_params,
            q_cur,
            qd_cur,
            tau_ref,
            method=method,
            external_force_ee=force,
            external_force_body_id=body_id,
        )

    if q_cur.ndim == 2:
        batch_size = int(q_cur.shape[0])
        if int(qd_cur.shape[0]) != batch_size:
            raise ValueError(f"Batch mismatch: q_cur={q_cur.shape}, qd_cur={qd_cur.shape}.")
        if tau_ref.ndim not in (2, 3):
            raise ValueError(
                f"Expected tau_ref rank 2 or 3 for batched states; got tau_ref={tau_ref.shape}."
            )
        if int(tau_ref.shape[0]) != batch_size:
            raise ValueError(
                f"Batch mismatch: q_cur batch={batch_size}, tau_ref shape={tau_ref.shape}."
            )
        if force is not None:
            if force.ndim != 2:
                raise ValueError(
                    f"Expected external_force_ee rank 2 for batched states; got external_force_ee={force.shape}."
                )
            if int(force.shape[0]) != batch_size:
                raise ValueError(
                    f"Batch mismatch: q_cur batch={batch_size}, external_force_ee shape={force.shape}."
                )
            if force.shape[-1] not in (3, 6):
                raise ValueError(
                    f"external_force_ee must have trailing size 3 or 6; got {force.shape}."
                )
        if body_id.ndim == 0:
            body_id = jnp.broadcast_to(body_id, (batch_size,))
        elif body_id.ndim == 1 and int(body_id.shape[0]) == batch_size:
            pass
        else:
            raise ValueError(
                f"Expected scalar or shape ({batch_size},) external_force_body_id for batched states; "
                f"got {body_id.shape}."
            )
        return jax.vmap(
            lambda rollout_param_i, q_i, qd_i, tau_ref_i, force_i, body_id_i: _compute_gt_tau_cmd_single(
                mjx_model_ideal,
                rollout_param_i,
                q_i,
                qd_i,
                tau_ref_i,
                method=method,
                external_force_ee=force_i,
                external_force_body_id=body_id_i,
            )
        )(
            rollout_params,
            q_cur,
            qd_cur,
            tau_ref,
            force if force is not None else jnp.zeros((batch_size, 3), dtype=q_cur.dtype),
            body_id,
        )

    raise ValueError(
        f"Unsupported q_cur/qd_cur rank for compute_gt_tau_cmd: q_cur={q_cur.shape}, qd_cur={qd_cur.shape}."
    )
