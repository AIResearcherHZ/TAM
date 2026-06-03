from dataclasses import dataclass
from pathlib import Path
import sys

from tests.repo_paths import REPO_ROOT as ROOT
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from simadaptor.cli import normalize_tyro_argv, parse_tyro_config
from simadaptor.config import TrainConfig


@dataclass
class _Cfg:
    hz_filter: tuple[int, ...] = ()
    robot_key: tuple[str, ...] = ()


def test_normalize_tyro_argv_hyphenates_flag_names_only():
    argv = ["--hz_filter", "500", "1000", "--robot_key", "panda_pandagripper"]
    assert normalize_tyro_argv(argv) == [
        "--hz-filter",
        "500",
        "1000",
        "--robot-key",
        "panda_pandagripper",
    ]


def test_parse_tyro_config_handles_variadic_option_followed_by_underscore_flag():
    cfg = parse_tyro_config(
        _Cfg,
        args=["--hz_filter", "500", "1000", "--robot_key", "panda_pandagripper"],
    )
    assert cfg.hz_filter == (500, 1000)
    assert cfg.robot_key == ("panda_pandagripper",)


def test_train_config_checkpoint_retention_defaults_to_keep_all_and_can_be_limited():
    assert TrainConfig().ckpt.max_to_keep == 0

    cfg = parse_tyro_config(TrainConfig, args=["--ckpt.max-to-keep", "3"])

    assert cfg.ckpt.max_to_keep == 3


def test_train_config_uses_full_rollout_chunk_by_default_and_can_limit():
    assert TrainConfig().max_rollout_time_chunk is None

    cfg = parse_tyro_config(
        TrainConfig,
        args=["--max-rollout-time-chunk", "512"],
    )

    assert cfg.max_rollout_time_chunk == 512
