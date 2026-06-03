from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

from tests.repo_paths import REPO_ROOT as ROOT

SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from simadaptor.deploy import inf_util  # noqa: E402


def _dense(fin: int, fout: int, *, bias: bool = True) -> dict:
    params = {"kernel": np.zeros((fin, fout), dtype=np.float32)}
    if bias:
        params["bias"] = np.zeros((fout,), dtype=np.float32)
    return params


def _layernorm(dim: int) -> dict:
    return {
        "scale": np.ones((dim,), dtype=np.float32),
        "bias": np.zeros((dim,), dtype=np.float32),
    }


def _block(hidden: int, out_dim: int, cond_dim: int) -> dict:
    return {
        "AdaLN_0": {
            "LayerNorm_0": _layernorm(hidden),
            "Dense_0": _dense(cond_dim, 2 * hidden),
        },
        "Dense_0": _dense(hidden, hidden),
        "AdaLN_1": {
            "LayerNorm_0": _layernorm(hidden),
            "Dense_0": _dense(cond_dim, 2 * hidden),
        },
        "Dense_1": _dense(hidden, out_dim),
    }


def _block_param_count(hidden: int, out_dim: int, cond_dim: int) -> int:
    return (
        2 * hidden
        + cond_dim * (2 * hidden)
        + (2 * hidden)
        + hidden * hidden
        + hidden
        + 2 * hidden
        + cond_dim * (2 * hidden)
        + (2 * hidden)
        + hidden * out_dim
        + out_dim
    )


def test_export_jointwise_direct_residual_head_sets_binary_flag(tmp_path: Path):
    dof = 7
    emb_dim = 5
    hidden = 4
    depth = 3
    history = 2
    proj_dim = 16

    params = {
        "q_stem": _dense(history, hidden, bias=False),
        "qd_stem": _dense(history, hidden, bias=False),
        "tau_stem": _dense(history, hidden, bias=False),
        "q_stem_ln": _layernorm(hidden),
        "qd_stem_ln": _layernorm(hidden),
        "tau_stem_ln": _layernorm(hidden),
        "joint_direct_tau_hyper": _dense(hidden, 2 * proj_dim),
        "joint_direct_projected2": _dense(2 * proj_dim, proj_dim),
        "joint_direct_projected_out": _dense(proj_dim, 1),
    }
    for idx in range(depth - 1):
        params[f"SimAdaptorBlock_{idx}"] = _block(hidden, hidden, emb_dim)
        params[f"joint_global_proj_{idx}"] = _dense(hidden, hidden)

    fake = object.__new__(inf_util.SimAdaptorInference)
    fake._ensure_checkpoint_loaded = lambda: None
    fake._simadaptor_params = {"adaptor": params}
    fake._cfg = SimpleNamespace(
        ablation_mode="tam",
        adaptor_seq_length=history,
        emb_dim=emb_dim,
        adaptor_hidden=hidden,
        adaptor_depth=depth,
    )
    fake._arm_joint_ids = list(range(dof))
    fake._norm_stats = None

    out_path = fake.export_simadaptor_weights_cpp(tmp_path / "direct.bin")
    raw = out_path.read_bytes()
    header = np.frombuffer(raw[: 6 * 4], dtype=np.int32)

    assert header.tolist() == [
        dof,
        emb_dim,
        hidden,
        depth,
        history,
        (1 << 1) | (1 << 3) | (1 << 4),
    ]

    exported_float_count = (len(raw) - 6 * 4) // 4
    expected_float_count = (
        3 * history * hidden
        + 6 * hidden
        + (depth - 1) * _block_param_count(hidden, hidden, emb_dim)
        + (depth - 1) * (hidden * hidden + hidden)
        + hidden * (2 * proj_dim)
        + (2 * proj_dim)
        + (2 * proj_dim) * proj_dim
        + proj_dim
        + proj_dim
        + 1
    )
    assert exported_float_count == expected_float_count
