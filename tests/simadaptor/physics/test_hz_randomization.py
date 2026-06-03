import importlib.util
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.repo_paths import REPO_ROOT as ROOT
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from simadaptor.config import TrainConfig
from simadaptor.training.hz_randomization import (
    build_exact_step_keep_mask,
    build_history_bernoulli_keep_mask,
    compute_history_token_time,
    mix_time_series_by_cut,
    select_source_by_token_mask,
)

SMOOTHING_PATH = SRC_ROOT / "simadaptor" / "physics" / "smoothing.py"
SMOOTHING_SPEC = importlib.util.spec_from_file_location("smoothing_test_module", SMOOTHING_PATH)
if SMOOTHING_SPEC is None or SMOOTHING_SPEC.loader is None:
    raise RuntimeError(f"Failed to load smoothing module from {SMOOTHING_PATH}")
smoothing_util = importlib.util.module_from_spec(SMOOTHING_SPEC)
SMOOTHING_SPEC.loader.exec_module(smoothing_util)

try:
    from mujoco import mjx as _mjx  # noqa: F401
    import simadaptor.core.structs as structs
    import simadaptor.models.adaptor as models
    import simadaptor.models.transformer as models_transformer
    import simadaptor.physics.dynamics as dynamics

    TRAIN_SCRIPT_PATH = ROOT / "scripts" / "train" / "tam" / "train.py"
    TRAIN_SPEC = importlib.util.spec_from_file_location("tam_train_test_module", TRAIN_SCRIPT_PATH)
    if TRAIN_SPEC is None or TRAIN_SPEC.loader is None:
        raise RuntimeError(f"Failed to load training script from {TRAIN_SCRIPT_PATH}")
    train_v3 = importlib.util.module_from_spec(TRAIN_SPEC)
    TRAIN_SPEC.loader.exec_module(train_v3)
    MJX_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment dependent
    structs = None
    models = None
    models_transformer = None
    dynamics = None
    train_v3 = None
    MJX_IMPORT_ERROR = exc


def _analytic_traj_1d(t: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    w1 = 2.0 * jnp.pi * 1.2
    w2 = 2.0 * jnp.pi * 0.45
    phase = 0.35
    q = jnp.sin(w1 * t) + 0.3 * jnp.cos(w2 * t + phase) + 0.02 * t * t
    qd = w1 * jnp.cos(w1 * t) - 0.3 * w2 * jnp.sin(w2 * t + phase) + 0.04 * t
    qdd = -(w1 * w1) * jnp.sin(w1 * t) - 0.3 * (w2 * w2) * jnp.cos(w2 * t + phase) + 0.04
    return q[:, None], qd[:, None], qdd[:, None]


def _masked_qdd_rmse(pred: jax.Array, target: jax.Array, keep_mask: jax.Array) -> float:
    keep = np.asarray(keep_mask[..., 0] > 0.5)
    pred_np = np.asarray(pred[..., 0])
    target_np = np.asarray(target[..., 0])
    diff = pred_np[keep] - target_np[keep]
    return float(np.sqrt(np.mean(diff * diff)))


def test_history_bernoulli_keep_mask_matches_hz_probabilities():
    rng = jax.random.PRNGKey(0)
    sampled_hz = jnp.concatenate(
        [
            jnp.full((512,), 200, dtype=jnp.int32),
            jnp.full((512,), 500, dtype=jnp.int32),
            jnp.full((512,), 1000, dtype=jnp.int32),
        ],
        axis=0,
    )
    keep_mask = build_history_bernoulli_keep_mask(
        rng,
        sampled_hz,
        time_steps=256,
        base_hz=1000,
    )

    keep_mean = np.asarray(jnp.mean(keep_mask, axis=(1, 2)))
    assert np.isclose(keep_mean[:512].mean(), 0.2, atol=0.02)
    assert np.isclose(keep_mean[512:1024].mean(), 0.5, atol=0.03)
    assert np.isclose(keep_mean[1024:].mean(), 1.0, atol=1e-6)


def test_exact_step_keep_mask_patterns():
    keep_mask = build_exact_step_keep_mask(
        jnp.array([1000, 500, 200], dtype=jnp.int32),
        window_len=6,
        base_hz=1000,
    )
    mask_np = np.asarray(keep_mask[..., 0])

    np.testing.assert_array_equal(mask_np[0], np.array([1, 1, 1, 1, 1, 1], dtype=np.float32))
    np.testing.assert_array_equal(mask_np[1], np.array([0, 1, 0, 1, 0, 1], dtype=np.float32))
    np.testing.assert_array_equal(mask_np[2], np.array([1, 0, 0, 0, 0, 1], dtype=np.float32))


def test_exact_step_keep_mask_short_window_200hz_keeps_only_current():
    keep_mask = build_exact_step_keep_mask(
        jnp.array([200], dtype=jnp.int32),
        window_len=4,
        base_hz=1000,
    )
    np.testing.assert_array_equal(
        np.asarray(keep_mask[0, :, 0]),
        np.array([0, 0, 0, 1], dtype=np.float32),
    )


def test_exact_step_keep_mask_always_keeps_last_slot():
    keep_mask = build_exact_step_keep_mask(
        jnp.array([200, 500, 1000], dtype=jnp.int32),
        window_len=6,
        base_hz=1000,
    )
    np.testing.assert_array_equal(
        np.asarray(keep_mask[:, -1, 0]),
        np.ones((3,), dtype=np.float32),
    )


def test_mixed_history_hz_repeats_across_timebatch_tokens():
    sampled_hz_mixed = jnp.array([200, 500], dtype=jnp.int32)
    sampled_hz_timedim = jnp.repeat(sampled_hz_mixed[:, None], 3, axis=1)

    np.testing.assert_array_equal(
        np.asarray(sampled_hz_timedim),
        np.array(
            [
                [200, 200, 200],
                [500, 500, 500],
            ],
            dtype=np.int32,
        ),
    )


def test_adaptor_mask_uses_mixed_sample_hz_independent_of_source_boundary():
    sampled_hz_mixed = jnp.array([500, 200], dtype=jnp.int32)
    sampled_hz_timedim = jnp.repeat(sampled_hz_mixed[:, None], 3, axis=1)
    use_original_source = jnp.array(
        [
            [True, False, True],
            [False, False, True],
        ]
    )
    del use_original_source  # adaptor mask should not depend on the orig/shuf boundary anymore

    keep_mask = build_exact_step_keep_mask(
        sampled_hz_timedim.reshape((-1,)),
        window_len=6,
        base_hz=1000,
    )
    keep_mask = np.asarray(keep_mask[..., 0]).reshape(2, 3, 6)

    expected_500 = np.array([0, 1, 0, 1, 0, 1], dtype=np.float32)
    expected_200 = np.array([1, 0, 0, 0, 0, 1], dtype=np.float32)
    for token_idx in range(3):
        np.testing.assert_array_equal(keep_mask[0, token_idx], expected_500)
        np.testing.assert_array_equal(keep_mask[1, token_idx], expected_200)


def test_mix_time_series_by_cut_splices_prefix_and_suffix():
    x_orig = jnp.array(
        [
            [[1.0], [2.0], [3.0], [4.0]],
            [[10.0], [20.0], [30.0], [40.0]],
        ]
    )
    x_shuf = jnp.array(
        [
            [[101.0], [102.0], [103.0], [104.0]],
            [[110.0], [120.0], [130.0], [140.0]],
        ]
    )
    cut_t = jnp.array([2, 3], dtype=jnp.int32)

    mixed = mix_time_series_by_cut(x_orig, x_shuf, cut_t)

    np.testing.assert_array_equal(
        np.asarray(mixed[..., 0]),
        np.array(
            [
                [1.0, 2.0, 103.0, 104.0],
                [10.0, 20.0, 30.0, 140.0],
            ]
        ),
    )


def test_compute_history_token_time_uses_patch_end_from_trimmed_tail():
    hist_idx = jnp.array([[0, 1, 2]], dtype=jnp.int32)
    hist_token_time = compute_history_token_time(
        hist_idx,
        n_tokens=3,
        total_steps=10,
        patch_size=4,
        patch_stride=2,
    )
    np.testing.assert_array_equal(
        np.asarray(hist_token_time),
        np.array([[5, 7, 9]], dtype=np.int32),
    )


def test_select_source_by_token_mask_routes_whole_random_window_from_owner():
    x_orig = jnp.array(
        [
            [[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]],
            [[[10.0], [20.0]], [[30.0], [40.0]], [[50.0], [60.0]]],
        ]
    )
    x_shuf = x_orig + 100.0
    use_original_source = jnp.array(
        [
            [True, False, True],
            [False, False, True],
        ]
    )

    selected = select_source_by_token_mask(x_orig, x_shuf, use_original_source)

    np.testing.assert_array_equal(
        np.asarray(selected[..., 0]),
        np.array(
            [
                [[1.0, 2.0], [103.0, 104.0], [5.0, 6.0]],
                [[110.0, 120.0], [130.0, 140.0], [50.0, 60.0]],
            ]
        ),
    )


def test_select_source_by_token_mask_handles_optional_rollout_param_leaves():
    x_orig = jnp.arange(12, dtype=jnp.float32).reshape(2, 2, 3)
    x_shuf = x_orig + 100.0
    use_original_source = jnp.array([[True, False], [False, True]])

    mixed = select_source_by_token_mask(x_orig, x_shuf, use_original_source)
    np.testing.assert_array_equal(
        np.asarray(mixed),
        np.array(
            [
                [[0.0, 1.0, 2.0], [103.0, 104.0, 105.0]],
                [[106.0, 107.0, 108.0], [9.0, 10.0, 11.0]],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        np.asarray(select_source_by_token_mask(x_orig, None, use_original_source)),
        np.asarray(x_orig),
    )
    np.testing.assert_array_equal(
        np.asarray(select_source_by_token_mask(None, x_shuf, use_original_source)),
        np.asarray(x_shuf),
    )
    assert select_source_by_token_mask(None, None, use_original_source) is None


@pytest.mark.parametrize("target_hz", [200, 500])
def test_masked_state_derivatives_exact_step_beats_zero_filled_finite_difference(target_hz: int):
    base_hz = 1000
    dt = 1.0 / float(base_hz)
    num_steps = 2001
    t = jnp.arange(num_steps, dtype=jnp.float32) * jnp.asarray(dt, dtype=jnp.float32)
    q_gt, qd_gt, qdd_gt = _analytic_traj_1d(t)
    q = q_gt[None, ...]
    qd = qd_gt[None, ...]

    keep_mask = build_exact_step_keep_mask(
        jnp.array([target_hz], dtype=jnp.int32),
        window_len=num_steps,
        base_hz=base_hz,
        dtype=jnp.float32,
    )
    q_masked = q * keep_mask
    qd_masked = qd * keep_mask

    qd_zero = jnp.gradient(q_masked, dt, axis=1)
    qdd_zero = jnp.gradient(qd_zero, dt, axis=1)
    _, _, qdd_fit = smoothing_util.estimate_masked_state_derivatives(
        q_masked,
        qd_masked,
        keep_mask,
        base_dt=dt,
        max_neighbors_each_side=20,
    )

    rmse_zero = _masked_qdd_rmse(qdd_zero[0], qdd_gt, keep_mask[0])
    rmse_fit = _masked_qdd_rmse(qdd_fit[0], qdd_gt, keep_mask[0])
    assert rmse_fit < rmse_zero


def test_masked_state_derivatives_all_ones_matches_analytic_reference():
    base_hz = 1000
    dt = 1.0 / float(base_hz)
    num_steps = 1001
    t = jnp.arange(num_steps, dtype=jnp.float32) * jnp.asarray(dt, dtype=jnp.float32)
    q_gt, qd_gt, qdd_gt = _analytic_traj_1d(t)
    q = q_gt[None, ...]
    qd = qd_gt[None, ...]
    keep_mask = jnp.ones((1, num_steps, 1), dtype=jnp.float32)

    q_fit, qd_fit, qdd_fit = smoothing_util.estimate_masked_state_derivatives(
        q,
        qd,
        keep_mask,
        base_dt=dt,
        max_neighbors_each_side=20,
    )

    keep_np = np.asarray(keep_mask[0, :, 0] > 0.5)
    q_fit_np = np.asarray(q_fit[0, :, 0])
    qd_fit_np = np.asarray(qd_fit[0, :, 0])
    qdd_fit_np = np.asarray(qdd_fit[0, :, 0])
    q_gt_np = np.asarray(q_gt[:, 0])
    qd_gt_np = np.asarray(qd_gt[:, 0])
    qdd_gt_np = np.asarray(qdd_gt[:, 0])

    rmse_q = float(np.sqrt(np.mean((q_fit_np[keep_np] - q_gt_np[keep_np]) ** 2)))
    rmse_qd = float(np.sqrt(np.mean((qd_fit_np[keep_np] - qd_gt_np[keep_np]) ** 2)))
    rmse_qdd = float(np.sqrt(np.mean((qdd_fit_np[keep_np] - qdd_gt_np[keep_np]) ** 2)))
    assert rmse_q < 1e-5
    assert rmse_qd < 2e-2
    assert rmse_qdd < 2.0


def test_masked_state_derivatives_bernoulli_masks_stay_finite():
    base_hz = 1000
    dt = 1.0 / float(base_hz)
    num_steps = 1201
    t = jnp.arange(num_steps, dtype=jnp.float32) * jnp.asarray(dt, dtype=jnp.float32)
    q_gt, qd_gt, _ = _analytic_traj_1d(t)
    q = jnp.repeat(q_gt[None, ...], 3, axis=0)
    qd = jnp.repeat(qd_gt[None, ...], 3, axis=0)
    sampled_hz = jnp.array([200, 500, 1000], dtype=jnp.int32)
    keep_mask = build_history_bernoulli_keep_mask(
        jax.random.PRNGKey(7),
        sampled_hz,
        time_steps=num_steps,
        base_hz=base_hz,
        dtype=jnp.float32,
    )
    q_masked = q * keep_mask
    qd_masked = qd * keep_mask

    q_fit, qd_fit, qdd_fit = smoothing_util.estimate_masked_state_derivatives(
        q_masked,
        qd_masked,
        keep_mask,
        base_dt=dt,
        max_neighbors_each_side=20,
    )

    assert np.all(np.isfinite(np.asarray(q_fit)))
    assert np.all(np.isfinite(np.asarray(qd_fit)))
    assert np.all(np.isfinite(np.asarray(qdd_fit)))


def test_masked_state_derivatives_qd_weight_can_dominate_noisy_q():
    base_hz = 1000
    dt = 1.0 / float(base_hz)
    num_steps = 1001
    t = jnp.arange(num_steps, dtype=jnp.float32) * jnp.asarray(dt, dtype=jnp.float32)
    q_gt, qd_gt, qdd_gt = _analytic_traj_1d(t)
    keep_mask = jnp.ones((1, num_steps, 1), dtype=jnp.float32)

    noise = 0.02 * jax.random.normal(jax.random.PRNGKey(3), q_gt.shape)
    q_noisy = (q_gt + noise)[None, ...]
    qd_clean = qd_gt[None, ...]

    _, _, qdd_q_heavy = smoothing_util.estimate_masked_state_derivatives(
        q_noisy,
        qd_clean,
        keep_mask,
        base_dt=dt,
        max_neighbors_each_side=20,
        q_weight=4.0,
        qd_weight=1.0,
    )
    _, _, qdd_qd_heavy = smoothing_util.estimate_masked_state_derivatives(
        q_noisy,
        qd_clean,
        keep_mask,
        base_dt=dt,
        max_neighbors_each_side=20,
        q_weight=1.0,
        qd_weight=4.0,
    )

    keep_np = np.asarray(keep_mask[0, :, 0] > 0.5)
    qdd_gt_np = np.asarray(qdd_gt[:, 0])
    rmse_q_heavy = float(np.sqrt(np.mean((np.asarray(qdd_q_heavy[0, :, 0])[keep_np] - qdd_gt_np[keep_np]) ** 2)))
    rmse_qd_heavy = float(np.sqrt(np.mean((np.asarray(qdd_qd_heavy[0, :, 0])[keep_np] - qdd_gt_np[keep_np]) ** 2)))
    assert rmse_qd_heavy < rmse_q_heavy


def test_masked_state_derivatives_support_zero_qd_weight():
    base_hz = 1000
    dt = 1.0 / float(base_hz)
    num_steps = 1001
    t = jnp.arange(num_steps, dtype=jnp.float32) * jnp.asarray(dt, dtype=jnp.float32)
    q_gt, qd_gt, qdd_gt = _analytic_traj_1d(t)
    keep_mask = jnp.ones((1, num_steps, 1), dtype=jnp.float32)

    q_fit, qd_fit, qdd_fit = smoothing_util.estimate_masked_state_derivatives(
        q_gt[None, ...],
        qd_gt[None, ...],
        keep_mask,
        base_dt=dt,
        max_neighbors_each_side=20,
        q_weight=1.0,
        qd_weight=0.0,
    )

    assert np.all(np.isfinite(np.asarray(q_fit)))
    assert np.all(np.isfinite(np.asarray(qd_fit)))
    assert np.all(np.isfinite(np.asarray(qdd_fit)))
    keep_np = np.asarray(keep_mask[0, :, 0] > 0.5)
    qdd_gt_np = np.asarray(qdd_gt[:, 0])
    rmse_qdd = float(np.sqrt(np.mean((np.asarray(qdd_fit[0, :, 0])[keep_np] - qdd_gt_np[keep_np]) ** 2)))
    assert rmse_qdd < 2.5


def test_masked_state_derivatives_support_zero_q_weight():
    base_hz = 1000
    dt = 1.0 / float(base_hz)
    num_steps = 1001
    t = jnp.arange(num_steps, dtype=jnp.float32) * jnp.asarray(dt, dtype=jnp.float32)
    q_gt, qd_gt, qdd_gt = _analytic_traj_1d(t)
    keep_mask = jnp.ones((1, num_steps, 1), dtype=jnp.float32)
    q_noisy = (q_gt + 0.02 * jax.random.normal(jax.random.PRNGKey(5), q_gt.shape))[None, ...]

    q_fit, qd_fit, qdd_fit = smoothing_util.estimate_masked_state_derivatives(
        q_noisy,
        qd_gt[None, ...],
        keep_mask,
        base_dt=dt,
        max_neighbors_each_side=20,
        q_weight=0.0,
        qd_weight=1.0,
    )

    assert np.all(np.isfinite(np.asarray(q_fit)))
    assert np.all(np.isfinite(np.asarray(qd_fit)))
    assert np.all(np.isfinite(np.asarray(qdd_fit)))
    keep_np = np.asarray(keep_mask[0, :, 0] > 0.5)
    qdd_gt_np = np.asarray(qdd_gt[:, 0])
    rmse_qdd = float(np.sqrt(np.mean((np.asarray(qdd_fit[0, :, 0])[keep_np] - qdd_gt_np[keep_np]) ** 2)))
    assert rmse_qdd < 2.5


def test_masked_state_derivatives_ignore_masked_outliers_inside_window():
    base_hz = 1000
    dt = 1.0 / float(base_hz)
    num_steps = 401
    t = jnp.arange(num_steps, dtype=jnp.float32) * jnp.asarray(dt, dtype=jnp.float32)
    q_gt, qd_gt, _ = _analytic_traj_1d(t)
    q = q_gt[None, ...]
    qd = qd_gt[None, ...]

    keep_mask = jnp.ones((1, num_steps, 1), dtype=jnp.float32)
    drop_idx = jnp.arange(20, num_steps - 20, 7)
    keep_mask = keep_mask.at[0, drop_idx, 0].set(0.0)

    q_masked_clean = q * keep_mask
    qd_masked_clean = qd * keep_mask
    q_masked_outlier = q_masked_clean.at[0, drop_idx, 0].set(1e3)
    qd_masked_outlier = qd_masked_clean.at[0, drop_idx, 0].set(-1e3)

    q_fit_clean, qd_fit_clean, qdd_fit_clean = smoothing_util.estimate_masked_state_derivatives(
        q_masked_clean,
        qd_masked_clean,
        keep_mask,
        base_dt=dt,
        max_neighbors_each_side=20,
    )
    q_fit_out, qd_fit_out, qdd_fit_out = smoothing_util.estimate_masked_state_derivatives(
        q_masked_outlier,
        qd_masked_outlier,
        keep_mask,
        base_dt=dt,
        max_neighbors_each_side=20,
    )

    np.testing.assert_allclose(np.asarray(q_fit_out), np.asarray(q_fit_clean), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(qd_fit_out), np.asarray(qd_fit_clean), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(qdd_fit_out), np.asarray(qdd_fit_clean), atol=1e-6, rtol=1e-6)


def _make_smoke_cfg(
    hz_enabled: bool,
    *,
    tau_map_sample_no: int = 1,
) -> TrainConfig:
    cfg = TrainConfig()
    cfg.hz_randomization_enable = hz_enabled
    cfg.hz_randomization_choices = (200, 500, 1000)
    cfg.hz_randomization_base_hz = 1000
    cfg.traj_mix_enable = False
    cfg.training_seq_length = 1
    cfg.rollout_loss_weight = 1.0
    cfg.tau_map_sample_no = tau_map_sample_no
    cfg.emb_dim = 16
    cfg.adaptor_hidden = 16
    cfg.adaptor_depth = 2
    cfg.adaptor_seq_length = 4
    cfg.enc.d_model = 32
    cfg.enc.num_heads = 4
    cfg.enc.num_layers = 1
    cfg.enc.dropout = 0.0
    cfg.enc.emb_dropout = 0.0
    cfg.enc.patch_size = 64
    cfg.enc.patch_stride = 32
    return cfg


@pytest.mark.skipif(MJX_IMPORT_ERROR is not None, reason=f"MJX unavailable: {MJX_IMPORT_ERROR}")
def _make_smoke_models_and_data(
    cfg: TrainConfig,
    *,
    time_steps: int = 200,
    external_force_scale: float = 0.0,
):
    xml_path = ROOT / "assets" / "franka_emika_panda" / "mjx_panda_nohand.xml"
    mjx_model = dynamics.load_mjx_model_from_path(str(xml_path), remove_constraints=True)
    dof = int(min(mjx_model.nq, mjx_model.nv, mjx_model.nu))
    batch_size = 2

    hist_model = models_transformer.JointwiseFlatARTransformerDecoder(
        cfg=cfg.enc,
        emb_dim=cfg.emb_dim,
        ideal_mjx_model=mjx_model,
    )
    adaptor_model = models.SimAdaptorJointwiseFlat(
        emb_dim=cfg.emb_dim,
        hidden=cfg.adaptor_hidden,
        depth=cfg.adaptor_depth,
    )

    qpos0 = jnp.asarray(mjx_model.qpos0[:dof], dtype=jnp.float32)
    t = jnp.linspace(0.0, 0.199, time_steps, dtype=jnp.float32)
    phase = t[None, :, None]
    joint_phase = jnp.linspace(0.0, 0.3, dof, dtype=jnp.float32)[None, None, :]
    batch_offset = jnp.arange(batch_size, dtype=jnp.float32)[:, None, None] * 0.01

    q = qpos0[None, None, :] + 0.05 * jnp.sin(2.0 * jnp.pi * phase + joint_phase) + batch_offset
    qd = jnp.repeat(0.1 * jnp.cos(2.0 * jnp.pi * phase + joint_phase), batch_size, axis=0)
    u = jnp.repeat(0.2 * jnp.sin(4.0 * jnp.pi * phase + joint_phase), batch_size, axis=0)
    force = jnp.zeros((batch_size, time_steps, 3), dtype=jnp.float32)
    if external_force_scale != 0.0:
        pulse = ((t >= 0.04) & (t <= 0.11)).astype(jnp.float32)[None, :, None]
        base_force = jnp.asarray([1.0, -0.4, 0.2], dtype=jnp.float32)[None, None, :]
        force = jnp.repeat(external_force_scale * pulse * base_force, batch_size, axis=0)

    rollout_inputs = {
        "q": q,
        "qd": qd,
        "u": u,
        "external_force_ee": force,
    }

    rollout_params_single = structs.RolloutParams(None, None).from_mjx_model(mjx_model)
    rollout_params = jax.tree.map(
        lambda x: None if x is None else jnp.repeat(x[None], batch_size, axis=0),
        rollout_params_single,
    )

    init_key = jax.random.PRNGKey(0)
    hist_key, adapt_key = jax.random.split(init_key)
    hist_params = hist_model.init({"params": hist_key, "dropout": hist_key}, q, qd, u)
    hist_emb0 = jnp.zeros((batch_size, dof, cfg.emb_dim), dtype=jnp.float32)
    adaptor_params = adaptor_model.init(
        adapt_key,
        jnp.zeros((batch_size, cfg.adaptor_seq_length, dof), dtype=jnp.float32),
        jnp.zeros((batch_size, cfg.adaptor_seq_length, dof), dtype=jnp.float32),
        jnp.zeros((batch_size, cfg.adaptor_seq_length, dof), dtype=jnp.float32),
        hist_emb0,
    )
    params = {"hist": hist_params, "adaptor": adaptor_params}
    external_force_body_id = int(mjx_model.nbody - 1)
    return mjx_model, hist_model, adaptor_model, params, (rollout_inputs, rollout_params), dof, external_force_body_id


@pytest.mark.skipif(MJX_IMPORT_ERROR is not None, reason=f"MJX unavailable: {MJX_IMPORT_ERROR}")
def _run_loss_smoke(
    hz_enabled: bool,
    *,
    tau_map_sample_no: int = 1,
    rollout_cmd_noise_std: float = 0.0,
    external_force_scale: float = 0.0,
    training_seq_length: int | None = None,
    time_steps: int = 200,
):
    cfg = _make_smoke_cfg(
        hz_enabled,
        tau_map_sample_no=tau_map_sample_no,
    )
    if training_seq_length is not None:
        cfg.training_seq_length = training_seq_length
    mjx_model, hist_model, adaptor_model, params, datasets, dof, external_force_body_id = _make_smoke_models_and_data(
        cfg,
        time_steps=time_steps,
        external_force_scale=external_force_scale,
    )
    rng = jax.random.PRNGKey(123)

    loss, aux = train_v3.loss_function(
        params,
        rng,
        cfg,
        hist_model,
        adaptor_model,
        mjx_model,
        datasets=datasets,
        external_force_body_id=(external_force_body_id if external_force_scale != 0.0 else -1),
        rollout_cmd_noise_std=rollout_cmd_noise_std,
        norm_stats=train_v3.init_norm_stats(dof),
        is_eval=False,
    )
    return float(loss), jax.tree.map(np.asarray, aux)


@pytest.mark.skipif(MJX_IMPORT_ERROR is not None, reason=f"MJX unavailable: {MJX_IMPORT_ERROR}")
def test_history_encoder_apply_stays_finite_with_masked_history():
    cfg = _make_smoke_cfg(hz_enabled=True)
    mjx_model, hist_model, _, params, datasets, _, _ = _make_smoke_models_and_data(cfg)
    rollout_inputs, _ = datasets
    q = rollout_inputs["q"]
    qd = rollout_inputs["qd"]
    u = rollout_inputs["u"]
    sampled_hz = jnp.array([200, 500], dtype=jnp.int32)
    keep_mask = build_exact_step_keep_mask(
        sampled_hz,
        window_len=int(q.shape[1]),
        base_hz=1000,
        dtype=q.dtype,
    )
    q_masked = q * keep_mask
    qd_masked = qd * keep_mask
    u_masked = u * keep_mask

    hist_emb = hist_model.apply(
        params["hist"],
        q_masked,
        qd_masked,
        u_masked,
        deterministic=True,
        input_keep_mask=keep_mask,
    )
    assert np.all(np.isfinite(np.asarray(hist_emb)))

@pytest.mark.skipif(MJX_IMPORT_ERROR is not None, reason=f"MJX unavailable: {MJX_IMPORT_ERROR}")
def test_loss_function_smoke_with_hz_randomization_enabled():
    loss, aux = _run_loss_smoke(hz_enabled=True)

    assert np.isfinite(loss)
    assert np.isfinite(aux["tau_cmd_recon_loss"])
    assert np.isfinite(aux["tau_cmd_rollout_step0_loss"])
    assert np.isfinite(aux["tau_cmd_rollout_loss"])
    np.testing.assert_allclose(aux["tau_cmd_recon_loss"], aux["tau_cmd_rollout_step0_loss"])
    assert aux["hz_rand_enabled"] == 1.0
    assert 0.0 <= aux["history_keep_ratio"] <= 1.0
    assert 0.0 < aux["adaptor_keep_ratio"] <= 1.0
    assert set(["hz_rand_frac_200", "hz_rand_frac_500", "hz_rand_frac_1000"]).issubset(aux.keys())


@pytest.mark.skipif(MJX_IMPORT_ERROR is not None, reason=f"MJX unavailable: {MJX_IMPORT_ERROR}")
def test_loss_function_smoke_with_hz_randomization_disabled():
    loss, aux = _run_loss_smoke(hz_enabled=False)

    assert np.isfinite(loss)
    assert np.isfinite(aux["tau_cmd_recon_loss"])
    assert np.isfinite(aux["tau_cmd_rollout_step0_loss"])
    assert np.isfinite(aux["tau_cmd_rollout_loss"])
    np.testing.assert_allclose(aux["tau_cmd_recon_loss"], aux["tau_cmd_rollout_step0_loss"])
    assert aux["hz_rand_enabled"] == 0.0
    assert np.isclose(aux["history_keep_ratio"], 1.0)
    assert np.isclose(aux["adaptor_keep_ratio"], 1.0)


@pytest.mark.skipif(MJX_IMPORT_ERROR is not None, reason=f"MJX unavailable: {MJX_IMPORT_ERROR}")
def test_loss_function_smoke_with_multi_sample_rollout():
    loss, aux = _run_loss_smoke(
        hz_enabled=True,
        tau_map_sample_no=8,
        rollout_cmd_noise_std=0.25,
    )

    assert np.isfinite(loss)
    assert np.isfinite(aux["tau_cmd_rollout_step0_loss"])
    assert np.isfinite(aux["tau_cmd_rollout_loss"])
    assert np.isfinite(aux["tau_cmd_rollout_mae"])
    assert aux["tau_map_sample_no"] == 8.0


@pytest.mark.skipif(MJX_IMPORT_ERROR is not None, reason=f"MJX unavailable: {MJX_IMPORT_ERROR}")
def test_jointwise_direct_residual_head_accepts_sampled_tau_des_override():
    cfg = _make_smoke_cfg(
        hz_enabled=False,
        tau_map_sample_no=8,
    )
    _, _, adaptor_model, params, _, dof, _ = _make_smoke_models_and_data(cfg)
    batch_size = 2
    sample_no = 8
    q = jnp.zeros((batch_size, cfg.adaptor_seq_length, dof), dtype=jnp.float32)
    qd = jnp.zeros_like(q)
    tau = jnp.zeros_like(q).at[:, -1, :].set(0.25)
    hist_emb = jnp.zeros((batch_size, dof, cfg.emb_dim), dtype=jnp.float32)
    tau_des_samples = jnp.linspace(-0.5, 0.5, sample_no, dtype=jnp.float32)[None, :, None]
    tau_des_samples = jnp.broadcast_to(tau_des_samples, (batch_size, sample_no, dof))

    delta_tau, _ = adaptor_model.apply(
        params["adaptor"],
        q,
        qd,
        tau,
        hist_emb,
        tau_des_override=tau_des_samples,
    )

    assert delta_tau.shape == (batch_size, sample_no, dof)
    np.testing.assert_array_equal(np.asarray(delta_tau), np.zeros((batch_size, sample_no, dof), dtype=np.float32))


@pytest.mark.skipif(MJX_IMPORT_ERROR is not None, reason=f"MJX unavailable: {MJX_IMPORT_ERROR}")
def test_multi_sample_rollout_matches_single_sample_when_noise_is_zero():
    loss_single, aux_single = _run_loss_smoke(
        hz_enabled=False,
        tau_map_sample_no=1,
        rollout_cmd_noise_std=0.0,
    )
    loss_multi, aux_multi = _run_loss_smoke(
        hz_enabled=False,
        tau_map_sample_no=8,
        rollout_cmd_noise_std=0.0,
    )

    np.testing.assert_allclose(loss_multi, loss_single, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        aux_multi["tau_cmd_rollout_loss"],
        aux_single["tau_cmd_rollout_loss"],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        aux_multi["tau_cmd_rollout_step0_loss"],
        aux_single["tau_cmd_rollout_step0_loss"],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        aux_multi["tau_cmd_rollout_mae"],
        aux_single["tau_cmd_rollout_mae"],
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.skipif(MJX_IMPORT_ERROR is not None, reason=f"MJX unavailable: {MJX_IMPORT_ERROR}")
def test_multi_sample_rollout_supports_direct_residual():
    loss, aux = _run_loss_smoke(
        hz_enabled=True,
        tau_map_sample_no=8,
        rollout_cmd_noise_std=0.25,
    )

    assert np.isfinite(loss)
    assert np.isfinite(aux["tau_cmd_rollout_step0_loss"])
    assert np.isfinite(aux["tau_cmd_rollout_loss"])
    assert np.isfinite(aux["tau_cmd_rollout_mae"])
    assert aux["tau_map_sample_no"] == 8.0


@pytest.mark.skipif(MJX_IMPORT_ERROR is not None, reason=f"MJX unavailable: {MJX_IMPORT_ERROR}")
def test_loss_function_smoke_with_external_force_rollout():
    loss, aux = _run_loss_smoke(
        hz_enabled=False,
        tau_map_sample_no=8,
        rollout_cmd_noise_std=0.0,
        external_force_scale=12.0,
    )

    assert np.isfinite(loss)
    assert np.isfinite(aux["tau_cmd_rollout_step0_loss"])
    assert np.isfinite(aux["tau_cmd_rollout_loss"])
    assert aux["external_force_absmean"] > 0.0
    assert aux["tau_eff_ext_ideal_absmean"] > 0.0
    np.testing.assert_allclose(aux["tau_cmd_recon_loss"], aux["tau_cmd_rollout_step0_loss"])


@pytest.mark.skipif(MJX_IMPORT_ERROR is not None, reason=f"MJX unavailable: {MJX_IMPORT_ERROR}")
def test_multi_sample_rollout_matches_single_sample_with_force_when_noise_is_zero():
    loss_single, aux_single = _run_loss_smoke(
        hz_enabled=False,
        tau_map_sample_no=1,
        rollout_cmd_noise_std=0.0,
        external_force_scale=12.0,
    )
    loss_multi, aux_multi = _run_loss_smoke(
        hz_enabled=False,
        tau_map_sample_no=8,
        rollout_cmd_noise_std=0.0,
        external_force_scale=12.0,
    )

    np.testing.assert_allclose(loss_multi, loss_single, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        aux_multi["tau_cmd_rollout_loss"],
        aux_single["tau_cmd_rollout_loss"],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        aux_multi["tau_cmd_rollout_step0_loss"],
        aux_single["tau_cmd_rollout_step0_loss"],
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.skipif(MJX_IMPORT_ERROR is not None, reason=f"MJX unavailable: {MJX_IMPORT_ERROR}")
def test_step0_only_rollout_loss_is_finite():
    loss, aux = _run_loss_smoke(
        hz_enabled=False,
        tau_map_sample_no=1,
        rollout_cmd_noise_std=0.0,
        training_seq_length=0,
    )

    assert np.isfinite(loss)
    assert np.isfinite(aux["tau_cmd_rollout_step0_loss"])
    assert np.isfinite(aux["tau_cmd_rollout_loss"])
    assert aux["rollout_steps"] == 0.0
    np.testing.assert_allclose(aux["tau_cmd_rollout_loss"], aux["tau_cmd_rollout_step0_loss"])


@pytest.mark.skipif(MJX_IMPORT_ERROR is not None, reason=f"MJX unavailable: {MJX_IMPORT_ERROR}")
def test_loss_function_requires_full_rollout_horizon():
    cfg = _make_smoke_cfg(hz_enabled=False)
    cfg.training_seq_length = 3
    mjx_model, hist_model, adaptor_model, params, datasets, dof, _ = _make_smoke_models_and_data(
        cfg,
        time_steps=cfg.adaptor_seq_length + cfg.training_seq_length + 1,
    )

    with pytest.raises(ValueError, match="adaptor_seq_length \\+ training_seq_length \\+ 2"):
        train_v3.loss_function(
            params,
            jax.random.PRNGKey(123),
            cfg,
            hist_model,
            adaptor_model,
            mjx_model,
            datasets=datasets,
            external_force_body_id=-1,
            rollout_cmd_noise_std=0.0,
            norm_stats=train_v3.init_norm_stats(dof),
            is_eval=False,
        )
