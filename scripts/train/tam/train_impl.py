# training_simadaptor.py
from __future__ import annotations

import os
import copy
import glob
import logging
import multiprocessing as mp
import sys
import warnings
os.environ["MUJOCO_GL"] = "egl"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
logging.getLogger("absl").setLevel(logging.ERROR)
warnings.filterwarnings(
    "ignore",
    message=r"The symbol `warp\.jax\.jax_kernel` will soon be removed from the public API\..*",
    category=DeprecationWarning,
)

from typing import Dict, Any, Optional, List, NamedTuple
from collections.abc import Mapping
import numpy as np

import mujoco
from mujoco import mjx

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core import FrozenDict, freeze, unfreeze
from flax.training import train_state, checkpoints
import optax
from functools import partial
import time
import json
try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
import tqdm
import pickle
import einops
import glob
import zarr

import wandb
import shutil
from pathlib import Path

import datetime
from absl import logging as absl_logging

from simadaptor.cli import parse_tyro_config
import simadaptor.core.structs as structs
from simadaptor.core.structs import NormStats
import simadaptor.models.transformer as models_transformer
import simadaptor.models.adaptor as models
from simadaptor.data.datagen_profiles import derive_robot_key, load_datagen_profile
try:
    import simadaptor.data.dataloader as dataloader
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    dataloader = None  # type: ignore[assignment]
import simadaptor.physics.actuator as actuator_util
import simadaptor.physics.dynamics as dynamics
from simadaptor.eval.gt_tau_cmd_validation import compute_gt_tau_cmd as _compute_gt_tau_cmd_shared
from simadaptor.eval.tau_reconstruction import TauReconstructionTester
from simadaptor.training.hz_randomization import (
    apply_time_keep_mask as _apply_time_keep_mask,
    build_exact_step_keep_mask as _build_exact_step_keep_mask,
    build_history_bernoulli_keep_mask as _build_history_bernoulli_keep_mask,
    compute_history_token_time as _compute_history_token_time,
    get_hz_randomization_settings as _get_hz_randomization_settings,
    mix_time_series_by_cut as _mix_time_series_by_cut,
    sample_hz_per_batch as _sample_hz_per_batch,
)

CUR_TIME_STR = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

from simadaptor.config import TrainConfig
import dataclasses

TAM_MODE = "tam"
SUPPORTED_ABLATION_MODES = (TAM_MODE,)
_LEGACY_CFG_FIELD_DEFAULTS: dict[str, Any] = {
    "ablation_mode": TAM_MODE,
    "random_input_delay_enable": True,
}


def _ablation_mode(cfg: TrainConfig) -> str:
    raw_mode = str(getattr(cfg, "ablation_mode", TAM_MODE) or TAM_MODE).strip()
    if raw_mode in {"", "tam"}:
        return TAM_MODE
    return raw_mode


def _validate_ablation_config(cfg: TrainConfig, *, is_multi_robot: bool = False) -> None:
    del is_multi_robot
    mode = _ablation_mode(cfg)
    if mode not in SUPPORTED_ABLATION_MODES:
        raise ValueError(
            "Public TAM training supports only ablation_mode='tam'. "
            f"Got {mode!r}."
        )
    if int(getattr(cfg, "adaptor_seq_length", 1) or 1) < 1:
        raise ValueError(f"adaptor_seq_length must be >= 1, got {cfg.adaptor_seq_length}.")


def _checkpoint_keep_limit(cfg: TrainConfig) -> int:
    raw_keep = getattr(getattr(cfg, "ckpt", None), "max_to_keep", 0)
    keep = int(raw_keep or 0)
    if keep < 0:
        raise ValueError(f"ckpt.max_to_keep must be >= 0, got {keep}.")
    return keep


def init_norm_stats(dof: int) -> NormStats:
    zeros = jnp.zeros((dof,), dtype=jnp.float32)
    ones = jnp.ones((dof,), dtype=jnp.float32)
    return NormStats(
        mean_q=zeros,
        var_q=ones,
        mean_dq=zeros,
        var_dq=ones,
        mean_u=zeros,
        var_u=ones,
    )


def update_norm_stats(stats: NormStats, q, dq, u, momentum: float = 0.99) -> NormStats:
    """Update EMA stats with batch q/dq/u shaped [B, T, DoF]."""
    def _ema(old_mean, old_var, x):
        x_mean = jnp.mean(x, axis=(0, 1))
        x_var = jnp.mean((x - x_mean) ** 2, axis=(0, 1))
        new_mean = momentum * old_mean + (1.0 - momentum) * x_mean
        new_var = momentum * old_var + (1.0 - momentum) * x_var
        return new_mean, new_var

    mean_q, var_q = _ema(stats.mean_q, stats.var_q, q)
    mean_dq, var_dq = _ema(stats.mean_dq, stats.var_dq, dq)
    mean_u, var_u = _ema(stats.mean_u, stats.var_u, u)
    return NormStats(mean_q=mean_q, var_q=var_q, mean_dq=mean_dq, var_dq=var_dq, mean_u=mean_u, var_u=var_u)


def split_shard_paths(base_path: str, split: str, num_data_limit: Optional[int], train_fraction: float, seed: int):
    """Deterministically split shard paths into non-overlapping train/eval sets."""
    paths = sorted(glob.glob(os.path.join(base_path, split, "*.zarr")))
    paths += sorted(glob.glob(os.path.join(base_path, split, "*.zarr.zip")))
    if not paths:
        raise FileNotFoundError(f"No .zarr files under {base_path}/{split}")
    if num_data_limit is not None:
        paths = paths[:num_data_limit]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(paths))
    cutoff = int(len(paths) * train_fraction)
    if cutoff == 0 and len(paths) > 1:
        cutoff = len(paths) - 1  # leave at least one for eval
    train_idx = perm[:cutoff]
    eval_idx = perm[cutoff:]
    # ensure eval non-empty
    if len(eval_idx) == 0 and len(train_idx) > 0:
        eval_idx = train_idx[-1:]
        train_idx = train_idx[:-1]
    train_paths = [paths[i] for i in train_idx]
    eval_paths = [paths[i] for i in eval_idx]
    return train_paths, eval_paths


def _build_perturbed_dataloader(
    *,
    base_path: str,
    fields: list[str],
    paths: list[str],
    time_chunk: Optional[int],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    drop_last: bool = True,
) -> torch.utils.data.DataLoader:
    if torch is None or dataloader is None:
        raise RuntimeError(
            "torch is required to build training dataloaders; install torch or use "
            "an image that already provides it."
        )
    dataset = dataloader.PandaRolloutShardDataset(
        base_path=base_path,
        split="perturbed",
        fields=fields,
        time_chunk=time_chunk,
        num_data_limit=None,
        paths=paths,
    )
    loader_kwargs: Dict[str, Any] = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=dataloader.concat_shard_collate,
        drop_last=drop_last,
        num_workers=int(max(0, num_workers)),
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["worker_init_fn"] = dataloader.seed_dataloader_worker
        if sys.platform.startswith("linux"):
            loader_kwargs["multiprocessing_context"] = mp.get_context("forkserver")
    return torch.utils.data.DataLoader(dataset, **loader_kwargs)


def _open_zarr_group(path: str, mode: str = "r"):
    if path.endswith(".zip"):
        store = zarr.storage.ZipStore(path, mode=mode)
        group = zarr.open_group(store=store, mode=mode)
        return group, store.close
    return zarr.open_group(path, mode=mode), (lambda: None)


def _read_root_attrs(path: str) -> Dict[str, Any]:
    group, closer = _open_zarr_group(path, mode="r")
    try:
        return dict(group.attrs)
    finally:
        closer()


def _rollout_field_exists(path: str, field: str) -> bool:
    group, closer = _open_zarr_group(path, mode="r")
    try:
        return ("rollout" in group) and (field in group["rollout"])
    finally:
        closer()


def _infer_rollout_time_length(path: str, preferred_fields: tuple[str, ...] = ("q", "qd", "u")) -> Optional[int]:
    group, closer = _open_zarr_group(path, mode="r")
    try:
        if "rollout" not in group:
            return None
        g_rollout = group["rollout"]
        for name in preferred_fields:
            if name in g_rollout:
                arr = g_rollout[name]
                if len(arr.shape) >= 2:
                    return int(arr.shape[1])
        for _, arr in g_rollout.arrays():
            if len(arr.shape) >= 2:
                return int(arr.shape[1])
        return None
    finally:
        closer()


@dataclasses.dataclass
class RobotTrainContext:
    robot_key: str
    dataset_dir: Path
    train_enabled: bool
    manifest_path: Optional[Path]
    manifest: Dict[str, Any]
    ds_cfg: Dict[str, Any]
    xml_path: Path
    mjx_model: mjx.Model
    ideal_model_has_gravity: bool
    external_force_body_id: int
    has_external_force_field: bool
    train_paths: List[str]
    eval_paths: List[str]
    first_pert_path: Optional[str]
    rollout_fields: List[str]
    time_chunk: Optional[int]
    data_loader_train: torch.utils.data.DataLoader
    data_loader_eval: Optional[torch.utils.data.DataLoader]
    dof: int
    resolved_profile_key: str
    datagen_profile_kwargs: Dict[str, Any]
    rollout_cmd_noise_std: float
    history_enc_model: Optional[nn.Module] = None
    train_step_jit: Any = None
    eval_func_jit: Any = None


def _is_robot_dataset_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "data_generation_config.json").exists()
        and (path / "perturbed").exists()
    )


def _discover_robot_dataset_dirs(dataset_root: Path) -> List[Path]:
    if _is_robot_dataset_dir(dataset_root):
        return [dataset_root]
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_root}")
    robot_dirs = [d for d in sorted(dataset_root.iterdir()) if _is_robot_dataset_dir(d)]
    if not robot_dirs:
        raise FileNotFoundError(
            f"No robot datasets found under {dataset_root}. "
            "Expected either a dataset dir with 'perturbed/' + data_generation_config.json "
            "or a root containing such robot subdirectories."
        )
    return robot_dirs


def _dataset_dir_robot_keys(dataset_dir: Path) -> tuple[str, ...]:
    keys: List[str] = [dataset_dir.name]
    manifest_path = dataset_dir / "robot_model" / "manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            manifest_robot_key = str(manifest.get("robot_key", "")).strip()
            if manifest_robot_key:
                keys.append(manifest_robot_key)
        except Exception as e:
            print(f"Warning: failed to read manifest {manifest_path} while resolving robot keys: {e}")
    return tuple(k for i, k in enumerate(keys) if k and k not in keys[:i])


def _filter_robot_dataset_dirs(
    dataset_dirs: List[Path],
    requested_robot_keys: Optional[tuple[str, ...]],
) -> List[Path]:
    if requested_robot_keys is None or len(requested_robot_keys) == 0:
        return dataset_dirs

    requested = tuple(
        key.strip() for key in requested_robot_keys
        if isinstance(key, str) and key.strip()
    )
    if not requested:
        return dataset_dirs

    keys_by_dir = {dataset_dir: _dataset_dir_robot_keys(dataset_dir) for dataset_dir in dataset_dirs}
    available_keys = sorted({key for keys in keys_by_dir.values() for key in keys})
    matched_dirs: List[Path] = []
    missing: List[str] = []

    for requested_key in requested:
        matched_dir = next(
            (dataset_dir for dataset_dir, keys in keys_by_dir.items() if requested_key in keys),
            None,
        )
        if matched_dir is None:
            missing.append(requested_key)
            continue
        if matched_dir not in matched_dirs:
            matched_dirs.append(matched_dir)

    if missing:
        raise ValueError(
            f"Unknown robot_key value(s) {missing}. "
            f"Available robot keys: {available_keys}"
        )

    return matched_dirs


def _resolve_effective_hz_choices(cfg: TrainConfig) -> tuple[int, tuple[int, ...], bool]:
    hz_base, default_hz_choices = _get_hz_randomization_settings(cfg)
    hz_filter_raw = getattr(cfg, "hz_filter", None)
    if hz_filter_raw is None or len(hz_filter_raw) == 0:
        return hz_base, default_hz_choices, bool(getattr(cfg, "hz_randomization_enable", False))

    hz_filter = tuple(int(hz) for hz in hz_filter_raw)
    if not hz_filter:
        raise ValueError("hz_filter must be non-empty when provided.")
    for hz in hz_filter:
        if hz <= 0:
            raise ValueError(f"hz_filter must contain positive values, got {hz_filter}.")
        if hz_base % hz != 0:
            raise ValueError(
                f"hz_randomization_base_hz={hz_base} must be divisible by each hz_filter value, got {hz_filter}."
            )
    return hz_base, hz_filter, True


def _resolve_dataset_robot_xml(
    dataset_dir: Path,
    fallback_xml: str | Path,
) -> tuple[Path, Optional[Path], Dict[str, Any]]:
    manifest_path = dataset_dir / "robot_model" / "manifest.json"
    manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            xml_rel = manifest.get("robot_xml")
            if xml_rel:
                cand = dataset_dir / str(xml_rel)
                if cand.exists():
                    return cand, manifest_path, manifest
        except Exception as e:
            print(f"Warning: failed to read manifest {manifest_path}: {e}")

    dataset_robot_xml = dataset_dir / "robot_model" / "robot.xml"
    if dataset_robot_xml.exists():
        return dataset_robot_xml, (manifest_path if manifest_path.exists() else None), manifest

    fallback_xml = Path(fallback_xml).expanduser()
    fallback_candidates = [fallback_xml, Path.cwd() / fallback_xml]
    for cand in fallback_candidates:
        if cand.exists():
            return cand, (manifest_path if manifest_path.exists() else None), manifest

    raise FileNotFoundError(
        f"Failed to resolve robot XML for dataset {dataset_dir}. "
        f"Tried manifest, robot_model/robot.xml, and fallback {fallback_xml}."
    )


def _resolve_profile_table_path(dataset_dir: Path, table_path: str | Path) -> Path:
    raw = Path(table_path).expanduser()
    candidates: List[Path] = []
    if raw.is_absolute():
        candidates = [raw]
    else:
        candidates = [raw, dataset_dir / raw, Path.cwd() / raw]
    checked: List[str] = []
    for cand in candidates:
        cand_resolved = cand.resolve()
        checked.append(str(cand_resolved))
        if cand_resolved.exists():
            return cand_resolved
    raise FileNotFoundError(
        f"Datagen profile table not found. raw='{table_path}', checked={checked}"
    )


def _copy_robot_bundle_to_dir(
    *,
    xml_path: Path,
    manifest: Dict[str, Any],
    dataset_dir: Path,
    robot_key: str,
    dst_robot_dir: Path,
) -> Dict[str, Any]:
    dst_robot_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(xml_path, dst_robot_dir / "robot.xml")
    src_assets = xml_path.parent / "assets"
    if src_assets.exists():
        shutil.copytree(src_assets, dst_robot_dir / "assets", dirs_exist_ok=True)

    out_manifest = dict(manifest) if manifest else {}
    out_manifest["robot_key"] = robot_key
    out_manifest["robot_xml"] = "robot.xml"
    out_manifest.setdefault("source_xml", str(xml_path))
    out_manifest["dataset_dir"] = str(dataset_dir)
    with open(dst_robot_dir / "manifest.json", "w") as f:
        json.dump(out_manifest, f, indent=2)
    return out_manifest


def _resolve_external_force_body(
    xml_path: str,
    ds_cfg: Dict[str, Any],
    root_attrs: Optional[Dict[str, Any]] = None,
) -> tuple[int, str]:
    root_attrs = root_attrs or {}
    body_id_attr = root_attrs.get("external_force_body_id", None)
    if body_id_attr is not None:
        try:
            body_id = int(body_id_attr)
            if body_id >= 0:
                body_name = root_attrs.get("external_force_body_name", f"body_{body_id}")
                return body_id, str(body_name)
        except Exception:
            pass

    mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
    body_name = root_attrs.get("external_force_body_name", ds_cfg.get("external_force_body_name"))

    if body_name:
        body_id = int(mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, str(body_name)))
        if body_id >= 0:
            return body_id, str(body_name)

    raise ValueError(
        "Failed to resolve external-force body from attrs/cfg. "
        f"attrs_body_id={body_id_attr!r}, cfg_body_name={body_name!r}"
    )

def _get_attr_path(obj, dotted: str):
    cur = obj
    for part in dotted.split("."):
        cur = getattr(cur, part)
    return cur

def _set_attr_path(obj, dotted: str, value):
    parts = dotted.split(".")
    cur = obj
    for part in parts[:-1]:
        cur = getattr(cur, part)
    setattr(cur, parts[-1], value)


def _to_mutable_tree(tree):
    if isinstance(tree, FrozenDict):
        return unfreeze(tree)
    if isinstance(tree, Mapping):
        return {k: _to_mutable_tree(v) for k, v in tree.items()}
    return tree


def _merge_missing_tree(loaded, template):
    if isinstance(template, (FrozenDict, Mapping)):
        template_map = _to_mutable_tree(template)
        loaded_map = _to_mutable_tree(loaded) if isinstance(loaded, (FrozenDict, Mapping)) else {}
        merged = {}
        for key, tmpl_val in template_map.items():
            if key in loaded_map:
                merged[key] = _merge_missing_tree(loaded_map[key], tmpl_val)
            else:
                merged[key] = copy.deepcopy(tmpl_val)
        for key, loaded_val in loaded_map.items():
            if key not in merged:
                merged[key] = loaded_val
        if isinstance(template, FrozenDict) or isinstance(loaded, FrozenDict):
            return freeze(merged)
        return merged
    return loaded if loaded is not None else copy.deepcopy(template)

def _backfill_missing_cfg_fields(cfg_obj: object, defaults: object) -> None:
    """Backfill missing attributes in (possibly pickled) cfg objects.

    Pickle restores objects without calling __init__, so newly-added dataclass
    fields may be absent when loading old checkpoints.
    """
    if cfg_obj is None or defaults is None:
        return
    for field in dataclasses.fields(defaults):
        name = field.name
        if not hasattr(cfg_obj, name):
            if name in _LEGACY_CFG_FIELD_DEFAULTS:
                setattr(cfg_obj, name, copy.deepcopy(_LEGACY_CFG_FIELD_DEFAULTS[name]))
            else:
                setattr(cfg_obj, name, copy.deepcopy(getattr(defaults, name)))
            continue
        cur_val = getattr(cfg_obj, name)
        default_val = getattr(defaults, name)
        if dataclasses.is_dataclass(default_val) and cur_val is not None:
            _backfill_missing_cfg_fields(cur_val, default_val)

def maybe_restore_cfg_from_ckpt(cfg: TrainConfig, ckpt_dir: str, cli_cfg: Optional[TrainConfig] = None) -> TrainConfig:
    """Optionally load the saved config from the checkpoint directory."""
    if not cfg.restore_cfg_from_ckpt:
        return cfg
    cfg_pickle = os.path.join(ckpt_dir, "save_dict.pkl")
    if not os.path.isfile(cfg_pickle):
        return cfg
    try:
        with open(cfg_pickle, "rb") as f:
            saved = pickle.load(f)
        saved_cfg = saved.get("cfg")
        if saved_cfg is None:
            return cfg
        print(f"Restoring cfg from checkpoint: {cfg_pickle}")
        print("Pass --restore_cfg_from_ckpt=False to keep CLI overrides instead.")
        _backfill_missing_cfg_fields(saved_cfg, TrainConfig())
        override_fields = getattr(cli_cfg, "override_from_cli", ()) if cli_cfg is not None else ()
        if not override_fields:
            return saved_cfg
        for field_path in override_fields:
            try:
                val = _get_attr_path(cli_cfg, field_path)
                _set_attr_path(saved_cfg, field_path, val)
                print(f"Override cfg field '{field_path}' with CLI value '{val}'")
            except AttributeError:
                print(f"Warning: override_from_cli path '{field_path}' not found; skipping.")
        return saved_cfg
    except Exception as e:
        print(f"Failed to load cfg from checkpoint ({e}); using CLI cfg instead.")
        return cfg

def restore_checkpoint_with_report(ckpt_dir: str, state: train_state.TrainState):
    """Restore a checkpoint if present and report whether resume happened.

    Prefers Flax checkpoints (full state incl. opt_state). Falls back to save_dict.pkl
    only if no Flax checkpoint is found.
    """
    ckpt_dir_path = Path(ckpt_dir)
    latest = checkpoints.latest_checkpoint(ckpt_dir)
    restored = state
    resumed = False

    if latest is not None:
        restored = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=state)
        if hasattr(restored, "params"):
            try:
                restored = restored.replace(params=_merge_missing_tree(restored.params, state.params))
            except Exception as exc:
                print(f"Warning: failed to merge missing params after Flax restore: {exc}")
        resumed = True
        print(f"Restored checkpoint from {latest}")

    if not resumed:
        save_dict_path = ckpt_dir_path / "save_dict.pkl"
        if save_dict_path.exists():
            try:
                with open(save_dict_path, "rb") as f:
                    saved = pickle.load(f)
                saved_step = saved.get("step")
                saved_opt_step = saved.get("opt_step")
                saved_epoch = saved.get("epoch")
                saved_params = saved.get("params")
                saved_norm_stats = saved.get("norm_stats")
                if saved_params is not None:
                    restored = restored.replace(params=_merge_missing_tree(saved_params, restored.params))
                step_value_raw = saved_opt_step if saved_opt_step is not None else saved_step
                if step_value_raw is not None:
                    step_value = int(step_value_raw)
                    try:
                        restored = restored.replace(step=jnp.array(step_value, dtype=restored.step.dtype))
                    except Exception:
                        restored = restored.replace(step=step_value)
                if saved_epoch is not None:
                    restored = restored.replace(epoch=int(saved_epoch))
                elif saved_step is not None:
                    # Legacy save_dict.pkl files stored the last completed loop label in
                    # `step`, so resume from the next label instead of replaying it.
                    restored = restored.replace(epoch=int(saved_step) + 1)
                if saved_norm_stats is not None and hasattr(restored, "norm_stats"):
                    restored = restored.replace(norm_stats=saved_norm_stats)
                resumed = True
                print(
                    "Loaded save_dict.pkl "
                    f"(step={saved_step}, opt_step={saved_opt_step}, epoch={saved_epoch}) "
                    f"from {save_dict_path} (opt_state not available)."
                )
            except Exception as e:
                print(f"Warning: failed to load save_dict.pkl at {save_dict_path}: {e}")

    if not resumed:
        print(f"No checkpoint found in {ckpt_dir}; starting fresh.")

    return restored, resumed


def _safe_wandb_artifact_name(name: str) -> str:
    return str(name).strip().replace("/", "_").replace(":", "_")


def _default_ckpt_artifact_name(run_name: str) -> str:
    return f"tam_ckpt_{_safe_wandb_artifact_name(run_name)}"


def log_latest_checkpoint_artifact_to_wandb(
    *,
    wandb_run,
    ckpt_dir: str | Path,
    step: int,
    run_name: str,
    artifact_name: Optional[str] = None,
    artifact_type: str = "model",
) -> bool:
    """Upload latest checkpoint files (and pointer/save_dict) as a W&B artifact."""
    import tempfile
    import tarfile

    ckpt_dir = Path(ckpt_dir)
    latest = checkpoints.latest_checkpoint(str(ckpt_dir))
    if latest is None:
        print(f"Warning: no checkpoint found in {ckpt_dir}; skipping W&B artifact upload.")
        return False

    latest_path = Path(latest)
    ckpt_parent = latest_path.parent
    prefix = latest_path.name
    resolved_artifact_name = _safe_wandb_artifact_name(
        artifact_name if artifact_name else _default_ckpt_artifact_name(run_name)
    )
    artifact = wandb.Artifact(
        name=resolved_artifact_name,
        type=str(artifact_type),
        metadata={
            "step": int(step),
            "run_name": str(run_name),
            "checkpoint_prefix": str(prefix),
        },
    )

    def _add_path(path: Path):
        if path.is_file():
            artifact.add_file(str(path), name=path.name)
            return
        if path.is_dir():
            add_dir = getattr(artifact, "add_dir", None)
            if callable(add_dir):
                add_dir(str(path), name=path.name)
                return
            with tempfile.TemporaryDirectory() as tmpdir:
                archive_path = Path(tmpdir) / f"{path.name}.tar.gz"
                with tarfile.open(str(archive_path), "w:gz") as tar:
                    tar.add(str(path), arcname=path.name)
                artifact.add_file(str(archive_path), name=archive_path.name)
            return
        print(f"Warning: artifact path is neither file nor directory: {path}")

    ckpt_entries = sorted(ckpt_parent.glob(f"{prefix}*"))
    if ckpt_entries:
        for entry in ckpt_entries:
            _add_path(entry)
    else:
        _add_path(latest_path)

    pointer_file = ckpt_parent / "checkpoint"
    if pointer_file.exists():
        _add_path(pointer_file)

    save_dict_file = ckpt_parent / "save_dict.pkl"
    if save_dict_file.exists():
        _add_path(save_dict_file)

    wandb_run.log_artifact(artifact, aliases=[f"step_{int(step)}", "latest"])
    return True

def add_noise_to_obs(
    q, dq, rng,
    q_std=3e-4, dq_std=3e-3,
    has_time_dim=False,
):
    """
    Adds Gaussian noise to q, dq.
    If has_time_dim=True and cutoff is provided (>0), uses a low-pass (Hann-window FIR) to create
    temporally correlated noise with that cutoff frequency.
    Shapes: q, dq are [..., DoF] or [..., T, DoF]
    Returns: noisy_q, noisy_dq
    """

    rng, sub = jax.random.split(rng)
    q = q + jax.random.normal(sub, q.shape) * q_std
    rng, sub = jax.random.split(rng)
    dq = dq + jax.random.normal(sub, dq.shape) * dq_std

    return q, dq


def _push_window(window: jnp.ndarray, new_elem: jnp.ndarray) -> jnp.ndarray:
    """Push one element to the end of a time window, dropping the oldest."""
    if int(window.shape[-2]) == 0:
        return window
    return jnp.concatenate([window[..., 1:, :], new_elem[..., None, :]], axis=-2)


def _fit_to_model_dim(x: jax.Array, target_dim: int) -> jax.Array:
    """Pad/truncate trailing DoF dim to match model nq/nv/nu size."""
    common = min(int(x.shape[-1]), int(target_dim))
    out = jnp.zeros(x.shape[:-1] + (int(target_dim),), dtype=x.dtype)
    return out.at[..., :common].set(x[..., :common])


def _tau_ref_noise_std(rollout_cmd_noise_std: float, dtype: jnp.dtype) -> jax.Array:
    return jnp.asarray(0.5 * float(rollout_cmd_noise_std), dtype=dtype)


def _build_shared_rollout_keep_mask(
    rng: jax.Array,
    batch_size: int,
    window_len: int,
    *,
    dtype: jnp.dtype,
) -> jax.Array:
    if int(window_len) <= 0:
        return jnp.ones((int(batch_size), 0, 1), dtype=dtype)
    keep_start = jax.random.randint(
        rng,
        shape=(int(batch_size), 1),
        minval=0,
        maxval=int(window_len),
    )
    keep = jnp.arange(int(window_len), dtype=jnp.int32)[None, :] >= keep_start
    return keep[..., None].astype(dtype)


def compute_gt_tau_cmd(
    mjx_model_ideal: mjx.Model,
    rollout_params: structs.RolloutParams,
    q_cur: jax.Array,
    qd_cur: jax.Array,
    tau_ref: jax.Array,
    *,
    external_force_ee: jax.Array | None = None,
    external_force_body_id: int | jax.Array = -1,
) -> jax.Array:
    return _compute_gt_tau_cmd_shared(
        mjx_model_ideal,
        rollout_params,
        q_cur,
        qd_cur,
        tau_ref,
        external_force_ee=external_force_ee,
        external_force_body_id=external_force_body_id,
    )


def _resolve_rollout_external_force_body_ids(
    rollout_inputs_ptb: Dict[str, jax.Array],
    batch_size: int,
    fallback_body_id: int,
) -> jax.Array:
    body_id = rollout_inputs_ptb.get("external_force_body_id", None)
    if body_id is None:
        return jnp.full((int(batch_size),), int(fallback_body_id), dtype=jnp.int32)

    body_id = jnp.asarray(body_id, dtype=jnp.int32)
    if body_id.ndim == 1 and int(body_id.shape[0]) == int(batch_size):
        return body_id
    if body_id.ndim == 2 and body_id.shape == (int(batch_size), 1):
        return body_id[:, 0]
    raise ValueError(
        f"rollout external_force_body_id must have shape ({batch_size},) or ({batch_size}, 1), "
        f"got {body_id.shape}."
    )


def _replace_data_external_force(
    data: mjx.Data,
    external_force_t: jax.Array | None,
    external_force_body_id: int | jax.Array,
) -> mjx.Data:
    xfrc_applied = jnp.zeros_like(data.xfrc_applied)
    if external_force_t is None:
        return data.replace(xfrc_applied=xfrc_applied)

    force_wrench = jnp.asarray(external_force_t, dtype=xfrc_applied.dtype)
    body_id = jnp.asarray(
        -1 if external_force_body_id is None else external_force_body_id,
        dtype=jnp.int32,
    )
    if force_wrench.shape[-1] not in (3, 6):
        raise ValueError(f"external_force_t must have trailing size 3 or 6; got {force_wrench.shape}")

    def _set_single_body_wrench(
        xfrc_single: jax.Array,
        force_single: jax.Array,
        body_id_single: jax.Array,
    ) -> jax.Array:
        def _apply(xfrc: jax.Array) -> jax.Array:
            if force_single.shape[-1] == 3:
                return xfrc.at[body_id_single, :3].set(force_single)
            return xfrc.at[body_id_single, :].set(force_single)

        return jax.lax.cond(
            (body_id_single >= 0) & (body_id_single < int(xfrc_single.shape[-2])),
            _apply,
            lambda xfrc: xfrc,
            xfrc_single,
        )

    if xfrc_applied.ndim == 2:
        if body_id.ndim != 0:
            raise ValueError(
                f"Expected scalar external_force_body_id for unbatched mjx.Data, got {body_id.shape}."
            )
        xfrc_applied = _set_single_body_wrench(xfrc_applied, force_wrench, body_id)
        return data.replace(xfrc_applied=xfrc_applied)

    if xfrc_applied.ndim == 3:
        batch_size = int(xfrc_applied.shape[0])
        if force_wrench.ndim != 2 or int(force_wrench.shape[0]) != batch_size:
            raise ValueError(
                f"Expected external_force_t shape ({batch_size}, 3|6) for batched mjx.Data, got {force_wrench.shape}."
            )
        if body_id.ndim == 0:
            body_id = jnp.broadcast_to(body_id, (batch_size,))
        elif body_id.ndim == 1 and int(body_id.shape[0]) == batch_size:
            pass
        else:
            raise ValueError(
                f"Expected scalar or shape ({batch_size},) external_force_body_id for batched mjx.Data, "
                f"got {body_id.shape}."
            )
        xfrc_applied = jax.vmap(_set_single_body_wrench)(xfrc_applied, force_wrench, body_id)
        return data.replace(xfrc_applied=xfrc_applied)

    raise ValueError(f"Unsupported xfrc_applied rank in mjx.Data: {xfrc_applied.shape}")


def _gather_rollout_segments_from_source(
    x: jax.Array,
    source_batch_idx: jax.Array,
    seg_idx: jax.Array,
) -> jax.Array:
    """Gather rollout sub-sequences from selected source batches.

    Args:
      x: [B, T, ...]
      source_batch_idx: [B, N_tok]
      seg_idx: [B, N_tok, seg_len]

    Returns:
      Gathered segments shaped [B, N_tok, seg_len, ...].
    """
    x_source = x[source_batch_idx]  # [B, N_tok, T, ...]
    take_idx = seg_idx
    while take_idx.ndim < x_source.ndim:
        take_idx = take_idx[..., None]
    take_idx = jnp.broadcast_to(
        take_idx,
        x_source.shape[:2] + seg_idx.shape[2:] + x_source.shape[3:],
    )
    return jnp.take_along_axis(x_source, take_idx, axis=2)


def _gather_rollout_params_from_source(
    rollout_params: structs.RolloutParams,
    source_batch_idx_flat: jax.Array,
) -> structs.RolloutParams:
    """Gather per-trajectory rollout params for flattened history-token sources."""
    gathered = jax.tree.map(
        lambda x: x[source_batch_idx_flat] if x is not None else None,
        rollout_params,
    )
    if isinstance(gathered, structs.RolloutParams):
        return gathered
    return structs.RolloutParams(**gathered)


def _delay_range_ms_to_step_bounds(
    delay_range_ms: tuple[float, float],
    *,
    dt_s: float,
) -> tuple[int, int]:
    if len(delay_range_ms) != 2:
        raise ValueError(f"delay range must contain exactly two values, got {delay_range_ms!r}")
    lo_ms = float(delay_range_ms[0])
    hi_ms = float(delay_range_ms[1])
    if not np.isfinite(lo_ms) or not np.isfinite(hi_ms):
        raise ValueError(f"delay range must be finite, got {delay_range_ms!r}")
    if lo_ms < 0.0 or hi_ms < 0.0:
        raise ValueError(f"delay range must be non-negative, got {delay_range_ms!r}")
    if hi_ms < lo_ms:
        raise ValueError(f"delay range max must be >= min, got {delay_range_ms!r}")
    dt_ms = float(dt_s) * 1000.0
    lo_steps = int(np.ceil(lo_ms / dt_ms - 1e-9))
    hi_steps = int(np.floor(hi_ms / dt_ms + 1e-9))
    if hi_steps < lo_steps:
        quantized = int(np.round((0.5 * (lo_ms + hi_ms)) / dt_ms))
        lo_steps = quantized
        hi_steps = quantized
    return max(0, lo_steps), max(0, hi_steps)


def _sample_delay_steps(
    key: jax.Array,
    batch_size: int,
    *,
    min_steps: int,
    max_steps: int,
) -> jax.Array:
    if max_steps < min_steps:
        raise ValueError(f"max_steps must be >= min_steps, got {min_steps}, {max_steps}")
    if min_steps == max_steps:
        return jnp.full((batch_size,), min_steps, dtype=jnp.int32)
    return jax.random.randint(
        key,
        shape=(batch_size,),
        minval=min_steps,
        maxval=max_steps + 1,
        dtype=jnp.int32,
    )


def _apply_batch_time_delay(
    x: jax.Array,
    delay_steps: jax.Array,
) -> jax.Array:
    if x.ndim < 3:
        raise ValueError(f"Expected rank >= 3 tensor for delayed time series, got {x.shape}")
    batch_size = int(x.shape[0])
    time_len = int(x.shape[1])
    delay_steps = jnp.asarray(delay_steps, dtype=jnp.int32)
    if delay_steps.ndim == 0:
        delay_steps = jnp.broadcast_to(delay_steps, (batch_size,))
    elif delay_steps.ndim != 1 or int(delay_steps.shape[0]) != batch_size:
        raise ValueError(
            f"delay_steps must be scalar or shape ({batch_size},), got {delay_steps.shape}"
        )
    take_idx = jnp.arange(time_len, dtype=jnp.int32)[None, :] - delay_steps[:, None]
    take_idx = jnp.clip(take_idx, 0, time_len - 1)
    while take_idx.ndim < x.ndim:
        take_idx = take_idx[..., None]
    take_idx = jnp.broadcast_to(take_idx, x.shape)
    return jnp.take_along_axis(x, take_idx, axis=1)


def _gather_rollout_context_from_source(
    x: jax.Array,
    source_batch_idx: jax.Array,
    end_idx: jax.Array,
    total_len: int,
) -> jax.Array:
    if total_len < 1:
        raise ValueError(f"total_len must be >= 1, got {total_len}")
    time_len = int(x.shape[1])
    gather_idx = (
        end_idx[..., None] - (total_len - 1) + jnp.arange(total_len, dtype=jnp.int32)[None, None, :]
    )
    gather_idx = jnp.clip(gather_idx, 0, time_len - 1)
    return _gather_rollout_segments_from_source(x, source_batch_idx, gather_idx)


def _extract_delayed_window_from_buffer(
    buffer: jax.Array,
    delay_steps: jax.Array,
    *,
    window_len: int,
) -> jax.Array:
    if window_len < 0:
        raise ValueError(f"window_len must be >= 0, got {window_len}")
    if window_len == 0:
        return jnp.zeros(buffer.shape[:-2] + (0, buffer.shape[-1]), dtype=buffer.dtype)
    if buffer.ndim < 3:
        raise ValueError(f"Expected rank >= 3 buffer, got {buffer.shape}")
    buffer_len = int(buffer.shape[-2])
    max_delay = buffer_len - window_len
    if max_delay < 0:
        raise ValueError(
            f"Buffer too short for delayed window: buffer_len={buffer_len}, window_len={window_len}"
        )
    batch_size = int(buffer.shape[0])
    delay_steps = jnp.asarray(delay_steps, dtype=jnp.int32)
    if delay_steps.ndim == 0:
        delay_steps = jnp.broadcast_to(delay_steps, (batch_size,))
    elif delay_steps.ndim != 1 or int(delay_steps.shape[0]) != batch_size:
        raise ValueError(
            f"delay_steps must be scalar or shape ({batch_size},), got {delay_steps.shape}"
        )
    start_idx = jnp.clip(max_delay - delay_steps, 0, max_delay)
    take_idx = start_idx[:, None] + jnp.arange(window_len, dtype=jnp.int32)[None, :]
    while take_idx.ndim < buffer.ndim:
        take_idx = take_idx[..., None]
    take_idx = jnp.broadcast_to(take_idx, buffer.shape[:-2] + (window_len, buffer.shape[-1]))
    return jnp.take_along_axis(buffer, take_idx, axis=-2)


class TauStepResult(NamedTuple):
    tau_pred: jax.Array
    tau_gt: jax.Array
    loss_mean: jax.Array
    mae_mean: jax.Array


def _sample_tau_ref_candidates(
    tau_center: jax.Array,
    tau_noise_key: jax.Array,
    *,
    tau_map_sample_no: int,
    rollout_cmd_noise_std: float,
) -> jax.Array:
    tau_center = jax.lax.stop_gradient(tau_center)
    sample_no = int(tau_map_sample_no)
    if sample_no == 1:
        tau_ref = tau_center[:, None, :]
    else:
        tau_ref_noise = jax.random.normal(
            tau_noise_key,
            shape=(int(tau_center.shape[0]), sample_no - 1, int(tau_center.shape[-1])),
            dtype=tau_center.dtype,
        ) * _tau_ref_noise_std(rollout_cmd_noise_std, tau_center.dtype)
        tau_ref = jnp.concatenate(
            [tau_center[:, None, :], tau_center[:, None, :] + tau_ref_noise],
            axis=1,
        )
    return tau_ref


def _select_tau_sample(
    tau_pred: jax.Array,
    sample_key: jax.Array,
    *,
    tau_map_sample_no: int,
) -> jax.Array:
    sample_no = int(tau_map_sample_no)
    if sample_no == 1:
        return tau_pred[:, 0, :]
    selected_sample = jax.random.randint(
        sample_key,
        shape=(int(tau_pred.shape[0]),),
        minval=0,
        maxval=sample_no,
    )
    return jnp.take_along_axis(
        tau_pred,
        selected_sample[:, None, None],
        axis=1,
    )[:, 0, :]


def _huber_abs_error_stats(
    err: jax.Array,
    huber_delta: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    abs_err = jnp.abs(err)
    huber_elem = jnp.where(
        abs_err <= huber_delta,
        0.5 * (err * err) / huber_delta,
        abs_err - 0.5 * huber_delta,
    )
    return huber_elem, jnp.mean(huber_elem), jnp.mean(abs_err)


def _predict_and_score_tau_step(
    *,
    params: Dict[str, Any],
    adaptor_model: nn.Module,
    mjx_model_ideal: mjx.Model,
    rollout_params_flat: structs.RolloutParams,
    history_emb: jax.Array,
    norm_stats: Optional[NormStats],
    tau_map_sample_no: int,
    huber_delta: jax.Array,
    dof_cmd: int,
    external_force_body_id_flat: jax.Array,
    train: bool,
    q_win_cur: jax.Array,
    qd_win_cur: jax.Array,
    tau_hist_cur: jax.Array,
    tau_ref_center: jax.Array,
    tau_ref_samples: jax.Array,
    q_cur: jax.Array,
    qd_cur: jax.Array,
    external_force_t: jax.Array,
    input_keep_mask: Optional[jax.Array],
    drop_key_step: jax.Array,
) -> TauStepResult:
    tau_input = jnp.concatenate([tau_hist_cur, tau_ref_center[:, None, :]], axis=1)
    q_in, qd_in, tau_in = jax.tree.map(
        lambda x: jax.lax.stop_gradient(x),
        (q_win_cur, qd_win_cur, tau_input),
    )
    if input_keep_mask is not None:
        q_in = _apply_time_keep_mask(q_in, input_keep_mask)
        qd_in = _apply_time_keep_mask(qd_in, input_keep_mask)
        tau_in = _apply_time_keep_mask(tau_in, input_keep_mask)

    adaptor_kwargs = dict(
        train=train,
        rngs={"dropout": drop_key_step},
        norm_stats=norm_stats,
        input_keep_mask=input_keep_mask,
    )
    tau_override = tau_ref_samples[:, 0, :] if int(tau_map_sample_no) == 1 else tau_ref_samples
    tau_residual_pred, _ = adaptor_model.apply(
        params["adaptor"],
        q_in,
        qd_in,
        tau_in,
        history_emb,
        tau_des_override=tau_override,
        **adaptor_kwargs,
    )
    if tau_residual_pred.ndim == 2:
        tau_residual_pred = tau_residual_pred[:, None, :]
    tau_pred = tau_ref_samples + tau_residual_pred

    tau_ref_arg = tau_ref_samples[:, 0, :] if int(tau_map_sample_no) == 1 else tau_ref_samples
    tau_gt = compute_gt_tau_cmd(
        mjx_model_ideal,
        rollout_params_flat,
        q_cur,
        qd_cur,
        tau_ref_arg,
        external_force_ee=external_force_t,
        external_force_body_id=external_force_body_id_flat,
    )
    if tau_gt.ndim == 2:
        tau_gt = tau_gt[:, None, :]
    tau_gt = jax.lax.stop_gradient(tau_gt)
    _, tau_loss_mean, tau_mae_mean = _huber_abs_error_stats(
        tau_pred - tau_gt,
        huber_delta,
    )

    return TauStepResult(
        tau_pred,
        tau_gt,
        tau_loss_mean,
        tau_mae_mean,
    )


# =========================
# Loss function
# =========================
def loss_function(params: Dict[str, Any],
                  rng_key,
                  cfg: TrainConfig,
                  history_enc_model: nn.Module,
                  adaptor_model: nn.Module,
                  mjx_model_ideal: mjx.Model,
                  datasets,
                  external_force_body_id: int = -1,
                  rollout_cmd_noise_std: float = 0.0,
                  norm_stats: Optional[NormStats] = None,
                  is_eval: bool = False):
    """Compute the training loss for one logged batch.

    Major shapes:
      B: logged batch size
      T_hist: logged rollout length
      D: command DoF
      N_tok: encoder output token count used by v3 loss
      n_flat = B * N_tok
      S = tau_map_sample_no
      win = cfg.adaptor_seq_length

      q_hist_ptb / qd_hist_ptb / tau_cmd_logged_hist_ptb: [B, T_hist, D]
      history_emb_all: [B, N_tok, H] or [B, N_tok, P, H]
      starts / source_batch_idx: [B, N_tok]
      q_seg / qd_seg_aligned / tau_cmd_logged_seg_aligned_ptb / force_seg_ptb: [B, N_tok, seg_len, D_or_F]
      history_emb (flattened): [n_flat, H] or [n_flat, P, H]
      q_window_seed / qd_window_seed: [n_flat, win, D]
      tau_hist_seed: [n_flat, win - 1, D]
      force_roll: [n_flat, rollout_steps + 1, 3 or 6]
      tau_ref_* / tau_pred / tau_gt: [n_flat, S, D]
    """
    # Torque convention:
    # - tau_cmd_*: commanded torque (pre-actuator), the reconstruction target space.
    # - tau_eff_*: effective/applied torque (post-actuator / inverse-dynamics equivalent).
    # - ptb/ideal suffixes indicate perturbed dataset dynamics vs ideal model quantities.
    rng = rng_key
    rollout_cmd_noise_std = float(rollout_cmd_noise_std)
    if not np.isfinite(rollout_cmd_noise_std):
        raise ValueError(f"rollout_cmd_noise_std must be finite, got {rollout_cmd_noise_std}.")

    if datasets is None:
        raise ValueError(
            "TAM training requires datasets from the dataloader; online generation is disabled."
        )
    rollout_inputs_ptb, rollout_params_ptb = datasets

    if not isinstance(rollout_params_ptb, structs.RolloutParams):
        rollout_params_ptb = structs.RolloutParams(**rollout_params_ptb)

    q_roll_ptb = rollout_inputs_ptb["q"]
    qd_roll_ptb_aligned = rollout_inputs_ptb["qd"]
    tau_cmd_logged_roll_ptb_aligned = rollout_inputs_ptb["u"]
    q_hist_dtype = q_roll_ptb.dtype

    hz_base, hz_choices, hz_train_enable = _resolve_effective_hz_choices(cfg)
    hz_rand_active = bool(hz_train_enable) and (not is_eval)
    history_keep_mask = None
    history_keep_ratio = jnp.asarray(1.0, dtype=q_hist_dtype)
    batch_size = int(q_roll_ptb.shape[0])
    batch_idx = jnp.arange(batch_size, dtype=jnp.int32)

    delay_dt_s = 1e-3
    dq_delay_lo_steps, dq_delay_hi_steps = _delay_range_ms_to_step_bounds(
        tuple(getattr(cfg, "dq_delay_range_ms", (0.0, 2.0)) or (0.0, 2.0)),
        dt_s=delay_dt_s,
    )
    tau_delay_lo_steps, tau_delay_hi_steps = _delay_range_ms_to_step_bounds(
        tuple(getattr(cfg, "torque_delay_range_ms", (0.0, 4.0)) or (0.0, 4.0)),
        dt_s=delay_dt_s,
    )
    delay_active = bool(getattr(cfg, "random_input_delay_enable", True)) and (not is_eval)
    if delay_active:
        rng, dq_delay_key, tau_delay_key = jax.random.split(rng, 3)
        dq_delay_steps = _sample_delay_steps(
            dq_delay_key,
            batch_size,
            min_steps=dq_delay_lo_steps,
            max_steps=dq_delay_hi_steps,
        )
        tau_delay_steps = _sample_delay_steps(
            tau_delay_key,
            batch_size,
            min_steps=tau_delay_lo_steps,
            max_steps=tau_delay_hi_steps,
        )
    else:
        dq_delay_steps = jnp.zeros((batch_size,), dtype=jnp.int32)
        tau_delay_steps = jnp.zeros((batch_size,), dtype=jnp.int32)

    qd_roll_ptb_obs = _apply_batch_time_delay(qd_roll_ptb_aligned, dq_delay_steps)
    tau_cmd_logged_roll_ptb_obs = _apply_batch_time_delay(
        tau_cmd_logged_roll_ptb_aligned,
        tau_delay_steps,
    )
    q_hist_ptb = q_roll_ptb
    qd_hist_ptb = qd_roll_ptb_obs
    tau_cmd_logged_hist_ptb = tau_cmd_logged_roll_ptb_obs
    external_force_roll_ptb = rollout_inputs_ptb.get("external_force_ee", None)
    if external_force_roll_ptb is None:
        external_force_roll_ptb = jnp.zeros(
            q_roll_ptb.shape[:-1] + (3,),
            dtype=q_roll_ptb.dtype,
        )
    external_force_body_id_roll_ptb = _resolve_rollout_external_force_body_ids(
        rollout_inputs_ptb,
        int(q_roll_ptb.shape[0]),
        external_force_body_id,
    )
    external_force_raw_absmean = jnp.mean(jnp.abs(external_force_roll_ptb))
    rollout_steps = max(int(getattr(cfg, "training_seq_length", 0) or 0), 0)
    rollout_loss_weight = float(getattr(cfg, "rollout_loss_weight", 1.0) or 0.0)
    tau_map_sample_no = max(int(getattr(cfg, "tau_map_sample_no", 256) or 0), 0)
    huber_delta = jnp.asarray(float(getattr(cfg, "tau_recon_huber_delta", 0.01)), dtype=q_roll_ptb.dtype)
    if tau_map_sample_no < 1:
        raise ValueError(f"tau_map_sample_no must be >= 1, got {tau_map_sample_no}.")
    if not is_eval and rollout_loss_weight == 0.0:
        raise ValueError("At least one training loss weight must be non-zero.")
    traj_mix_enabled = bool(getattr(cfg, "traj_mix_enable", False)) and (not is_eval)
    traj_mix_active = bool(traj_mix_enabled and (int(q_roll_ptb.shape[1]) > 1))
    rng, mix_perm_key, mix_cut_key, seg_key, drop_key = jax.random.split(rng, 5)
    perm = batch_idx
    cut_t = jnp.zeros((batch_size,), dtype=jnp.int32)
    tau_cmd_logged_hist_encoder = tau_cmd_logged_hist_ptb

    if traj_mix_active:
        perm = jax.random.permutation(mix_perm_key, batch_size)
        cut_t = jax.random.randint(
            mix_cut_key,
            shape=(batch_size,),
            minval=1,
            maxval=int(q_roll_ptb.shape[1]),
        )

        q_hist_ptb = _mix_time_series_by_cut(q_roll_ptb, q_roll_ptb[perm], cut_t)
        qd_hist_ptb = _mix_time_series_by_cut(qd_roll_ptb_obs, qd_roll_ptb_obs[perm], cut_t)
        tau_cmd_logged_hist_encoder = _mix_time_series_by_cut(
            tau_cmd_logged_hist_ptb, tau_cmd_logged_roll_ptb_obs[perm], cut_t
        )

    rng_q_std, rng = jax.random.split(rng)
    q_std_random = jax.random.uniform(
        rng_q_std,
        rollout_inputs_ptb["q"][..., 0, :].shape,
        minval=1e-4,
        maxval=1e-3,
    )
    rng_dq_std, rng = jax.random.split(rng)
    dq_std_random = jax.random.uniform(
        rng_dq_std,
        rollout_inputs_ptb["qd"][..., 0, :].shape,
        minval=1e-3,
        maxval=1e-2,
    )
    rng_noise, rng = jax.random.split(rng)
    q_hist_ptb, qd_hist_ptb = add_noise_to_obs(
        q_hist_ptb,
        qd_hist_ptb,
        rng_noise,
        has_time_dim=True,
        q_std=q_std_random[..., None, :],
        dq_std=dq_std_random[..., None, :],
    )
    hz_sample_key, rng = jax.random.split(rng)
    sampled_hz_mixed = _sample_hz_per_batch(
        hz_sample_key,
        int(q_hist_ptb.shape[0]),
        base_hz=hz_base,
        choices=hz_choices,
        enabled=hz_rand_active,
    )

    if hz_rand_active:
        hist_mask_key, rng = jax.random.split(rng)
        history_keep_mask = _build_history_bernoulli_keep_mask(
            hist_mask_key,
            sampled_hz_mixed,
            int(q_hist_ptb.shape[1]),
            base_hz=hz_base,
            dtype=q_hist_ptb.dtype,
        )
        q_hist_ptb = _apply_time_keep_mask(q_hist_ptb, history_keep_mask)
        qd_hist_ptb = _apply_time_keep_mask(qd_hist_ptb, history_keep_mask)
        tau_cmd_logged_hist_encoder = _apply_time_keep_mask(tau_cmd_logged_hist_encoder, history_keep_mask)
        history_keep_ratio = jnp.mean(history_keep_mask)

    rollout_params_ptb = rollout_params_ptb.replace(
        q_noise_std=q_std_random, dq_noise_std=dq_std_random
    )

    dropout_rng, rng = jax.random.split(rng)
    history_emb = history_enc_model.apply(
        params["hist"],
        q_hist_ptb,
        qd_hist_ptb,
        tau_cmd_logged_hist_encoder,
        rngs={"dropout": dropout_rng},
        deterministic=is_eval,
        norm_stats=norm_stats,
        input_keep_mask=history_keep_mask,
    )

    if history_emb.ndim == 2:
        history_emb = history_emb[:, None, :]
    if history_emb.ndim not in (3, 4):
        raise ValueError(f"Unsupported history_emb shape: {history_emb.shape}")
    history_emb_all = history_emb  # [B, N_tok, H] or [B, N_tok, DoF, H]
    B, n_tokens = history_emb_all.shape[:2]
    token_idx = jnp.broadcast_to(jnp.arange(n_tokens, dtype=jnp.int32)[None, :], (B, n_tokens))

    win = cfg.adaptor_seq_length
    seg_len = int(win + rollout_steps + 2)
    B_hist = int(q_roll_ptb.shape[0])
    T_hist = int(q_roll_ptb.shape[1])
    if B != B_hist:
        raise ValueError(f"Batch mismatch between history tokens and rollout batch: B={B}, B_hist={B_hist}")
    start_max = T_hist - seg_len
    if start_max < 0:
        raise ValueError(
            "History trajectory too short for rollout window: "
            f"T_hist={T_hist}, required>={seg_len} (adaptor_seq_length + training_seq_length + 2)"
        )

    starts = jax.random.randint(seg_key, (B_hist, n_tokens), minval=0, maxval=start_max + 1)
    seg_idx = starts[..., None] + jnp.arange(seg_len, dtype=jnp.int32)[None, None, :]  # [B, N_tok, seg_len]
    if traj_mix_active:
        hist_token_time = _compute_history_token_time(
            token_idx,
            n_tokens=n_tokens,
            total_steps=T_hist,
            patch_size=int(cfg.enc.patch_size),
            patch_stride=int(cfg.enc.patch_stride),
        )
        use_original_source = hist_token_time < cut_t[:, None]  # [B, N_tok]
        source_batch_idx = jnp.where(use_original_source, batch_idx[:, None], perm[:, None])
        traj_mix_cut_t_mean = jnp.mean(cut_t.astype(q_roll_ptb.dtype))
        traj_mix_use_original_ratio = jnp.mean(use_original_source.astype(q_roll_ptb.dtype))
    else:
        source_batch_idx = jnp.broadcast_to(batch_idx[:, None], (B, n_tokens))
        traj_mix_cut_t_mean = jnp.asarray(0.0, dtype=q_roll_ptb.dtype)
        traj_mix_use_original_ratio = jnp.asarray(1.0, dtype=q_roll_ptb.dtype)

    q_seg = _gather_rollout_segments_from_source(q_roll_ptb, source_batch_idx, seg_idx)
    qd_seg_aligned = _gather_rollout_segments_from_source(
        qd_roll_ptb_aligned,
        source_batch_idx,
        seg_idx,
    )
    tau_cmd_logged_seg_aligned_ptb = _gather_rollout_segments_from_source(
        tau_cmd_logged_roll_ptb_aligned,
        source_batch_idx,
        seg_idx,
    )
    force_seg_ptb = _gather_rollout_segments_from_source(
        external_force_roll_ptb, source_batch_idx, seg_idx
    )
    dq_buffer_len = int(win + dq_delay_hi_steps)
    tau_hist_len = max(int(win) - 1, 0)
    tau_buffer_len = max(tau_hist_len + tau_delay_hi_steps, 1)
    qd_actual_buffer = _gather_rollout_context_from_source(
        qd_roll_ptb_aligned,
        source_batch_idx,
        starts + int(win),
        dq_buffer_len,
    )
    tau_actual_buffer = _gather_rollout_context_from_source(
        tau_cmd_logged_roll_ptb_aligned,
        source_batch_idx,
        starts + int(win) - 1,
        tau_buffer_len,
    )

    # Flatten the full history-token axis so adaptor rollout runs over n_flat=B*N_tok windows.
    history_emb = history_emb_all.reshape((-1,) + history_emb_all.shape[2:])  # [n_flat, ...]
    flat_batch_idx = jnp.broadcast_to(batch_idx[:, None], (B, n_tokens)).reshape((-1,))
    source_batch_idx_flat = source_batch_idx.reshape((-1,))
    sampled_hz_flat = sampled_hz_mixed[flat_batch_idx]
    external_force_body_id_flat = external_force_body_id_roll_ptb[source_batch_idx_flat]

    (
        q_seg,
        qd_seg_aligned,
        tau_cmd_logged_seg_aligned_ptb,
        force_seg_ptb,
        qd_actual_buffer,
        tau_actual_buffer,
    ) = jax.tree.map(
        lambda x: x.reshape((-1, x.shape[-2], x.shape[-1])),
        (
            q_seg,
            qd_seg_aligned,
            tau_cmd_logged_seg_aligned_ptb,
            force_seg_ptb,
            qd_actual_buffer,
            tau_actual_buffer,
        ),
    )  # each [n_flat, seg_len_or_buffer, D_or_F]
    rollout_params_flat = _gather_rollout_params_from_source(
        rollout_params_ptb,
        source_batch_idx_flat,
    )
    dq_delay_steps_flat = dq_delay_steps[source_batch_idx_flat]
    tau_delay_steps_flat = tau_delay_steps[source_batch_idx_flat]

    q_seg_seed = q_seg[:, : win + 2, :]
    qd_seg_seed_aligned = qd_seg_aligned[:, : win + 2, :]
    q_last = q_seg_seed[:, -2, :]  # [n_flat, D]
    qd_last = qd_seg_seed_aligned[:, -2, :]  # [n_flat, D]
    force_roll = force_seg_ptb[:, win : win + rollout_steps + 1, :]  # [n_flat, rollout_steps + 1, 3 or 6]
    force_ee_last = force_roll[:, 0, :]  # [n_flat, 3 or 6]
    dt = jnp.asarray(float(getattr(cfg, "tau_recon_dt", 1e-3)), dtype=q_last.dtype)
    qacc_last = (qd_seg_seed_aligned[:, -1, :] - qd_seg_seed_aligned[:, -2, :]) / dt  # [n_flat, D]

    tau_eff_id_ideal_from_qacc = jax.vmap(partial(dynamics.mjx_inverse_dynamics_rne, mjx_model_ideal))(
        q_last, qd_last, qacc_last
    )  # [n_flat, D]
    tau_eff_ext_ideal = dynamics.compute_external_tau_equivalent(
        mjx_model_ideal,
        q_last,
        qd_last,
        force_ee_last,
        external_force_body_id=external_force_body_id_flat,
    )  # [n_flat, D]
    tau_cmd_required_ideal = tau_eff_id_ideal_from_qacc - tau_eff_ext_ideal  # [n_flat, D]

    q_window_seed = q_seg_seed[:, 1:-1, :]  # [n_flat, win, D]
    qd_window_seed = _extract_delayed_window_from_buffer(
        qd_actual_buffer,
        dq_delay_steps_flat,
        window_len=int(win),
    )  # [n_flat, win, D]
    tau_hist_seed = _extract_delayed_window_from_buffer(
        tau_actual_buffer,
        tau_delay_steps_flat,
        window_len=tau_hist_len,
    )  # [n_flat, win - 1, D]
    adaptor_keep_mask_flat = None
    adaptor_keep_ratio = jnp.asarray(1.0, dtype=q_roll_ptb.dtype)
    if hz_rand_active:
        adaptor_keep_mask_flat = _build_exact_step_keep_mask(
            sampled_hz_flat,
            int(win),
            base_hz=hz_base,
            dtype=q_window_seed.dtype,
        )
        adaptor_keep_ratio = jnp.mean(adaptor_keep_mask_flat)

    _, step0_drop_key = jax.random.split(drop_key)
    history_emb_for_adaptor = history_emb

    dof_cmd = int(q_window_seed.shape[-1])
    n_flat = int(q_window_seed.shape[0])

    step_context = dict(
        params=params,
        adaptor_model=adaptor_model,
        mjx_model_ideal=mjx_model_ideal,
        rollout_params_flat=rollout_params_flat,
        history_emb=history_emb_for_adaptor,
        norm_stats=norm_stats,
        tau_map_sample_no=tau_map_sample_no,
        huber_delta=huber_delta,
        dof_cmd=dof_cmd,
        external_force_body_id_flat=external_force_body_id_flat,
        train=not is_eval,
    )

    sample_context = dict(
        tau_map_sample_no=tau_map_sample_no,
        rollout_cmd_noise_std=rollout_cmd_noise_std,
    )

    step0_noise_key, step0_sample_key, rollout_rng, rng = jax.random.split(rng, 4)
    tau_ref_step0 = _sample_tau_ref_candidates(
        tau_cmd_required_ideal,
        step0_noise_key,
        **sample_context,
    )
    (
        tau_pred_step0,
        _,
        tau_cmd_rollout_step0_loss,
        tau_cmd_rollout_step0_mae,
    ) = _predict_and_score_tau_step(
        **step_context,
        q_win_cur=q_window_seed,
        qd_win_cur=qd_window_seed,
        tau_hist_cur=tau_hist_seed,
        tau_ref_center=tau_cmd_required_ideal,
        tau_ref_samples=tau_ref_step0,
        q_cur=q_last,
        qd_cur=qd_last,
        external_force_t=force_ee_last,
        input_keep_mask=adaptor_keep_mask_flat,
        drop_key_step=step0_drop_key,
    )
    tau_cmd_step0_selected = _select_tau_sample(
        tau_pred_step0,
        step0_sample_key,
        tau_map_sample_no=tau_map_sample_no,
    )
    data0 = mjx.make_data(mjx_model_ideal)
    data0 = jax.tree.map(lambda x: jnp.repeat(x[None], n_flat, axis=0), data0)
    data0 = data0.replace(
        qpos=_fit_to_model_dim(q_last, mjx_model_ideal.nq),
        qvel=_fit_to_model_dim(qd_last, mjx_model_ideal.nv),
        ctrl=_fit_to_model_dim(tau_cmd_step0_selected, mjx_model_ideal.nu),
    )
    data0 = _replace_data_external_force(data0, force_ee_last, external_force_body_id_flat)
    data_boot = jax.vmap(partial(actuator_util.mjx_step_with_actuator, mjx_model_ideal))(
        data0, rollout_params_flat
    )
    q_boot = data_boot.qpos[..., :dof_cmd]  # [n_flat, D]
    qd_boot = data_boot.qvel[..., :dof_cmd]  # [n_flat, D]

    q_window_roll = _push_window(q_window_seed, q_boot)  # [n_flat, win, D]
    qd_actual_buffer_roll = _push_window(qd_actual_buffer, qd_boot)  # [n_flat, win + max_dq_delay, D]
    qd_window_roll = _extract_delayed_window_from_buffer(
        qd_actual_buffer_roll,
        dq_delay_steps_flat,
        window_len=int(win),
    )  # [n_flat, win, D]
    tau_actual_buffer_roll = _push_window(
        tau_actual_buffer,
        tau_cmd_step0_selected,
    )  # [n_flat, win - 1 + max_tau_delay, D]
    tau_hist_roll = _extract_delayed_window_from_buffer(
        tau_actual_buffer_roll,
        tau_delay_steps_flat,
        window_len=tau_hist_len,
    )  # [n_flat, win - 1, D]
    grav_ff = jax.vmap(partial(dynamics.mjx_inverse_dynamics_rne, mjx_model_ideal))(
        q_boot,
        jnp.zeros_like(qd_boot),
        jnp.zeros_like(qd_boot),
    )[..., :dof_cmd]  # [n_flat, D]
    grav_ff = jax.lax.stop_gradient(grav_ff)
    rollout_hz_keep_mask = adaptor_keep_mask_flat if hz_rand_active else None

    def rollout_step(carry, force_t):
        (
            data_cur,
            q_win_cur,
            qd_win_cur,
            tau_hist_cur,
            qd_actual_buf_cur,
            tau_actual_buf_cur,
            rng_cur,
        ) = carry
        rng_cur, tau_noise_key, keep_mask_key, sample_key, drop_key_step, center_choice_key = jax.random.split(
            rng_cur,
            6,
        )

        q_cur = data_cur.qpos[..., :dof_cmd]
        qd_cur = data_cur.qvel[..., :dof_cmd]
        tau_eff_ext_ideal_cur = dynamics.compute_external_tau_equivalent(
            mjx_model_ideal,
            q_cur,
            qd_cur,
            force_t,
            external_force_body_id=external_force_body_id_flat,
        )
        tau_ref_center_ff = grav_ff - tau_eff_ext_ideal_cur  # [n_flat, D]
        if tau_hist_len > 0:
            tau_ref_center_window = jnp.mean(tau_hist_cur, axis=-2)  # [n_flat, D]
        else:
            tau_ref_center_window = tau_ref_center_ff
        use_window_center = jax.random.bernoulli(
            center_choice_key,
            p=0.5,
            shape=(int(tau_ref_center_ff.shape[0]), 1),
        )
        tau_ref_center = jnp.where(use_window_center, tau_ref_center_window, tau_ref_center_ff)  # [n_flat, D]
        tau_ref_center_window_ratio = jnp.mean(use_window_center.astype(q_cur.dtype))
        tau_ref = _sample_tau_ref_candidates(
            tau_ref_center,
            tau_noise_key,
            **sample_context,
        )  # [n_flat, S, D]

        rollout_keep_mask = rollout_hz_keep_mask
        if rollout_keep_mask is None and not is_eval:
            rollout_keep_mask = _build_shared_rollout_keep_mask(
                keep_mask_key,
                n_flat,
                int(q_win_cur.shape[-2]),
                dtype=q_win_cur.dtype,
            )

        (
            tau_pred_roll,
            _,
            step_loss_mean,
            step_mae_mean,
        ) = _predict_and_score_tau_step(
            **step_context,
            q_win_cur=q_win_cur,
            qd_win_cur=qd_win_cur,
            tau_hist_cur=tau_hist_cur,
            tau_ref_center=tau_ref_center,
            tau_ref_samples=tau_ref,
            q_cur=q_cur,
            qd_cur=qd_cur,
            external_force_t=force_t,
            input_keep_mask=rollout_keep_mask,
            drop_key_step=drop_key_step,
        )
        tau_pred_roll_selected = _select_tau_sample(
            tau_pred_roll,
            sample_key,
            tau_map_sample_no=tau_map_sample_no,
        )

        data_next = data_cur.replace(ctrl=_fit_to_model_dim(tau_pred_roll_selected, mjx_model_ideal.nu))
        data_next = _replace_data_external_force(data_next, force_t, external_force_body_id_flat)
        data_next = jax.vmap(partial(actuator_util.mjx_step_with_actuator, mjx_model_ideal))(
            data_next, rollout_params_flat
        )
        q_next = data_next.qpos[..., :dof_cmd]  # [n_flat, D]
        qd_next = data_next.qvel[..., :dof_cmd]  # [n_flat, D]

        qd_actual_buf_next = _push_window(qd_actual_buf_cur, qd_next)
        tau_actual_buf_next = _push_window(tau_actual_buf_cur, tau_pred_roll_selected)

        carry_next = (
            data_next,
            _push_window(q_win_cur, q_next),
            _extract_delayed_window_from_buffer(
                qd_actual_buf_next,
                dq_delay_steps_flat,
                window_len=int(win),
            ),
            _extract_delayed_window_from_buffer(
                tau_actual_buf_next,
                tau_delay_steps_flat,
                window_len=tau_hist_len,
            ),
            qd_actual_buf_next,
            tau_actual_buf_next,
            rng_cur,
        )
        return carry_next, (
            step_loss_mean,
            step_mae_mean,
            tau_ref_center_window_ratio,
            q_next,
            qd_next,
        )

    rollout_force_scan = jnp.swapaxes(force_roll[:, 1:, :], 0, 1)  # [rollout_steps, n_flat, 3 or 6]
    rollout_carry0 = (
        data_boot,
        q_window_roll,
        qd_window_roll,
        tau_hist_roll,
        qd_actual_buffer_roll,
        tau_actual_buffer_roll,
        rollout_rng,
    )
    _, (
        roll_step_loss_seq,
        roll_step_mae_seq,
        roll_tau_center_window_ratio_seq,
        roll_q_next_seq,
        roll_qd_next_seq,
    ) = jax.lax.scan(
        rollout_step,
        rollout_carry0,
        xs=rollout_force_scan,
        length=rollout_steps,
    )
    total_step_loss_seq = jnp.concatenate([tau_cmd_rollout_step0_loss[None], roll_step_loss_seq], axis=0)
    total_step_mae_seq = jnp.concatenate([tau_cmd_rollout_step0_mae[None], roll_step_mae_seq], axis=0)
    # Per-step rollout scalars: [rollout_steps + 1].
    tau_cmd_rollout_loss = jnp.mean(total_step_loss_seq)
    tau_cmd_rollout_mae = jnp.mean(total_step_mae_seq)
    tau_cmd_recon_loss = tau_cmd_rollout_step0_loss
    tau_center_window_ratio = (
        jnp.mean(roll_tau_center_window_ratio_seq)
        if rollout_steps > 0
        else jnp.asarray(0.0, dtype=q_roll_ptb.dtype)
    )
    q_pred_seq = jnp.concatenate([q_boot[None, ...], roll_q_next_seq], axis=0)
    qd_pred_seq = jnp.concatenate([qd_boot[None, ...], roll_qd_next_seq], axis=0)
    q_target_seq = jnp.swapaxes(q_seg[:, win + 1 : win + rollout_steps + 2, :], 0, 1)
    qd_target_seq = jnp.swapaxes(qd_seg_aligned[:, win + 1 : win + rollout_steps + 2, :], 0, 1)
    q_rollout_err = q_pred_seq - q_target_seq
    qd_rollout_err = qd_pred_seq - qd_target_seq
    q_rollout_rmse = jnp.sqrt(jnp.mean(jnp.square(q_rollout_err)))
    qd_rollout_rmse = jnp.sqrt(jnp.mean(jnp.square(qd_rollout_err)))
    q_rollout_joint_rmse = jnp.sqrt(jnp.mean(jnp.square(q_rollout_err), axis=(0, 1)))
    qd_rollout_joint_rmse = jnp.sqrt(jnp.mean(jnp.square(qd_rollout_err), axis=(0, 1)))
    q_final_err = q_pred_seq[-1] - q_target_seq[-1]
    q_final_rmse = jnp.sqrt(jnp.mean(jnp.square(q_final_err)))
    q_final_joint_rmse = jnp.sqrt(jnp.mean(jnp.square(q_final_err), axis=0))

    training_loss = rollout_loss_weight * tau_cmd_rollout_loss
    sampled_hz_float = sampled_hz_mixed.astype(q_roll_ptb.dtype)

    aux = {
        "loss": training_loss,
        "tau_cmd_recon_loss": tau_cmd_recon_loss,
        "tau_cmd_rollout_loss": tau_cmd_rollout_loss,
        "tau_cmd_rollout_mae": tau_cmd_rollout_mae,
        "tau_cmd_rollout_step0_loss": tau_cmd_rollout_step0_loss,
        "tau_cmd_rollout_step0_mae": tau_cmd_rollout_step0_mae,
        "tau_center_window_ratio": tau_center_window_ratio,
        "rollout_loss_weight": jnp.asarray(rollout_loss_weight, dtype=q_roll_ptb.dtype),
        "rollout_steps": jnp.asarray(float(rollout_steps), dtype=q_roll_ptb.dtype),
        "tau_map_sample_no": jnp.asarray(float(tau_map_sample_no), dtype=q_roll_ptb.dtype),
        "tau_cmd_required_ideal_absmean": jnp.mean(jnp.abs(tau_cmd_required_ideal)),
        "tau_eff_ext_ideal_absmean": jnp.mean(jnp.abs(tau_eff_ext_ideal)),
        "external_force_absmean": jnp.mean(jnp.abs(force_roll)),
        "external_force_raw_absmean": external_force_raw_absmean,
        "rollout_cmd_noise_std": jnp.asarray(rollout_cmd_noise_std, dtype=q_roll_ptb.dtype),
        "tau_ref_noise_std": _tau_ref_noise_std(rollout_cmd_noise_std, q_roll_ptb.dtype),
        "input_delay_enabled": jnp.asarray(float(delay_active), dtype=q_roll_ptb.dtype),
        "dq_delay_steps_mean": jnp.mean(dq_delay_steps.astype(q_roll_ptb.dtype)),
        "tau_delay_steps_mean": jnp.mean(tau_delay_steps.astype(q_roll_ptb.dtype)),
        "dq_delay_ms_mean": jnp.mean(dq_delay_steps.astype(q_roll_ptb.dtype)) * 1.0,
        "tau_delay_ms_mean": jnp.mean(tau_delay_steps.astype(q_roll_ptb.dtype)) * 1.0,
        "traj_mix_enabled": jnp.asarray(float(traj_mix_active), dtype=q_roll_ptb.dtype),
        "traj_mix_cut_t_mean": traj_mix_cut_t_mean,
        "traj_mix_use_original_ratio": traj_mix_use_original_ratio,
        "hz_rand_enabled": jnp.asarray(float(hz_rand_active), dtype=q_roll_ptb.dtype),
        "hz_rand_mean": jnp.mean(sampled_hz_float),
        "hz_rand_frac_200": jnp.mean((sampled_hz_mixed == 200).astype(q_roll_ptb.dtype)),
        "hz_rand_frac_500": jnp.mean((sampled_hz_mixed == 500).astype(q_roll_ptb.dtype)),
        "hz_rand_frac_1000": jnp.mean((sampled_hz_mixed == 1000).astype(q_roll_ptb.dtype)),
        "history_keep_ratio": history_keep_ratio,
        "adaptor_keep_ratio": adaptor_keep_ratio,
    }
    if is_eval and bool(getattr(cfg.sim_eval, "dataset_eval_jointwise_rmse", True)):
        aux.update(
            {
                "dataset_rollout/q_rmse_rad": q_rollout_rmse,
                "dataset_rollout/qd_rmse_radps": qd_rollout_rmse,
                "dataset_rollout/final_q_rmse_rad": q_final_rmse,
            }
        )
        for joint_idx in range(dof_cmd):
            aux[f"dataset_rollout/q_rmse_joint_{joint_idx:02d}_rad"] = q_rollout_joint_rmse[joint_idx]
            aux[f"dataset_rollout/qd_rmse_joint_{joint_idx:02d}_radps"] = qd_rollout_joint_rmse[joint_idx]
            aux[f"dataset_rollout/final_q_rmse_joint_{joint_idx:02d}_rad"] = q_final_joint_rmse[joint_idx]
    return training_loss, aux


# =========================
# Train state & step
# =========================
class DualModelState(train_state.TrainState):
    epoch: int = 1
    norm_stats: Optional[NormStats] = None


def _count_nans(tree):
    counts = jax.tree_util.tree_map(lambda x: jnp.sum(jnp.isnan(x)), tree)
    return jax.tree_util.tree_reduce(lambda a, b: a + b, counts, initializer=jnp.array(0))

def _any_nan(tree):
    return _count_nans(tree) > 0

def _tree_where(mask, a, b):
    # Select per-leaf with a scalar boolean mask
    return jax.tree_util.tree_map(lambda x, y: jnp.where(mask, x, y), a, b)

def _tree_normal_like(key, tree):
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    keys = jax.random.split(key, len(leaves))
    noise_leaves = [jax.random.normal(k, shape=l.shape, dtype=l.dtype) for k, l in zip(keys, leaves)]
    return jax.tree_util.tree_unflatten(treedef, noise_leaves)


def _accumulate_metrics(
    accum: Optional[Dict[str, jax.Array]],
    metrics: Dict[str, Any],
) -> Dict[str, jax.Array]:
    if accum is None:
        return {k: jnp.asarray(v) for k, v in metrics.items()}
    out = dict(accum)
    for k, v in metrics.items():
        out[k] = out[k] + jnp.asarray(v)
    return out


def _periodic_interval(cfg_obj) -> int:
    if cfg_obj is None:
        return 0
    return int(getattr(cfg_obj, "dataset_eval_interval", 0) or 0)


def _should_run_periodic_interval(cfg_obj, step: int) -> bool:
    interval = _periodic_interval(cfg_obj)
    return interval > 0 and step % interval == 0


def _sim_eval_value(
    sim_eval_cfg,
    cfg,
    name: str,
    default,
    *,
    legacy_name: Optional[str] = None,
):
    if sim_eval_cfg is not None and hasattr(sim_eval_cfg, name):
        return getattr(sim_eval_cfg, name)
    if legacy_name is not None and cfg is not None and hasattr(cfg, legacy_name):
        return getattr(cfg, legacy_name)
    return default


# @jax.jit
def train_step(state: DualModelState,
               rng_key,
               cfg: TrainConfig,
               history_enc_model: nn.Module,
               adaptor_model: nn.Module,
               mjx_model_ideal: mjx.Model,
               external_force_body_id: int = -1,
               rollout_cmd_noise_std: float = 0.0,
               datasets=None):

    # RNG splits (keep deterministic)
    rng_key, loss_rng, noise_rng = jax.random.split(rng_key, 3)

    # (1) Use preloaded batch only (no online generation in v3)
    if datasets is None:
        raise ValueError(
            "TAM training requires datasets from the dataloader; online generation is disabled."
        )
    rollout_inputs_ptb, rollout_params_ptb = datasets

    # (1b) Update normalization stats (EMA) using raw logged commands.
    use_norm_stats = bool(getattr(cfg, "use_norm_stats", True))
    if use_norm_stats:
        tau_cmd_for_stats = rollout_inputs_ptb["u"]
        if state.norm_stats is None:
            dof = rollout_inputs_ptb["q"].shape[-1]
            norm_stats_updated = init_norm_stats(dof)
        else:
            norm_stats_updated = update_norm_stats(
                state.norm_stats,
                rollout_inputs_ptb["q"],
                rollout_inputs_ptb["qd"],
                tau_cmd_for_stats,
            )
        norm_stats_for_loss = norm_stats_updated
    else:
        norm_stats_updated = None
        norm_stats_for_loss = None

    # (2) Loss + grads
    loss_fn = partial(
        loss_function,
        cfg=cfg,
        history_enc_model=history_enc_model,
        adaptor_model=adaptor_model,
        mjx_model_ideal=mjx_model_ideal,
        external_force_body_id=external_force_body_id,
        rollout_cmd_noise_std=rollout_cmd_noise_std,
    )
    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, loss_rng, datasets=datasets, norm_stats=norm_stats_for_loss)

    # (3) Metrics (computed regardless of NaN handling)
    grads_nan = _count_nans(grads)
    grads_nan_hist = _count_nans(grads["hist"]) if "hist" in grads else jnp.array(0)
    grads_nan_adaptor = _count_nans(grads["adaptor"]) if "adaptor" in grads else jnp.array(0)
    grad_norm = optax.global_norm(grads)
    hist_grad_norm = optax.global_norm(grads["hist"]) if "hist" in grads else jnp.array(0.0)
    adaptor_grad_norm = optax.global_norm(grads["adaptor"]) if "adaptor" in grads else jnp.array(0.0)

    # (4) Compute BOTH paths:

    # 4a) Normal optimizer step
    new_state_normal = state.apply_gradients(grads=grads)

    # 4b) Skip update, keep optimizer state/step, add tiny noise to params
    #     (so learning doesn’t get stuck and we avoid propagating NaNs)
    noise_std = cfg.nan_noise_std
    noise_tree = _tree_normal_like(noise_rng, state.params)
    params_noisy = jax.tree_util.tree_map(lambda p, n: p + (noise_std * n), state.params, noise_tree)

    # Keep opt_state and step unchanged on skip path
    new_state_skip = state.replace(params=params_noisy)

    # (5) Mask: if loss or any grad has NaN → choose skip path
    has_nan = jnp.isnan(loss) | _any_nan(grads)

    # (6) Select fields with a scalar mask (keeps function jittable)
    final_params   = _tree_where(has_nan, new_state_skip.params,   new_state_normal.params)
    final_optstate = _tree_where(has_nan, state.opt_state,         new_state_normal.opt_state)
    final_step     = jnp.where(has_nan, state.step,                new_state_normal.step)

    new_state = new_state_normal.replace(params=final_params, opt_state=final_optstate, step=final_step, norm_stats=norm_stats_updated)

    metrics = {
        "loss": loss,
        "grad_norm": grad_norm,
        "hist_grad_norm": hist_grad_norm,
        "adaptor_grad_norm": adaptor_grad_norm,
        "grads_nan": grads_nan,
        "grads_nan_hist": grads_nan_hist,
        "grads_nan_adaptor": grads_nan_adaptor,
        "nan_triggered": has_nan.astype(jnp.int32),
        **aux,
    }
    return new_state, metrics

# =========================
# Main
# =========================
def main(cfg: TrainConfig):
    logging.getLogger("absl").setLevel(logging.ERROR)
    absl_logging.set_verbosity(absl_logging.ERROR)

    if cfg.platform is not None:
        os.environ["JAX_PLATFORM_NAME"] = cfg.platform
    if cfg.ckpt.run_name is None:
        cfg.ckpt.run_name = f"{CUR_TIME_STR}"
    requested_run_name = cfg.ckpt.run_name
    requested_workdir = cfg.ckpt.workdir
    requested_resume_step = getattr(cfg.ckpt, "resume_step", None)
    requested_max_to_keep = getattr(cfg.ckpt, "max_to_keep", 0)

    CKPT_DIR = os.path.abspath(os.path.join(requested_workdir, requested_run_name)) # make it absolute

    # If resuming, optionally pull cfg from checkpoint to avoid silent mismatches.
    cfg = maybe_restore_cfg_from_ckpt(cfg, CKPT_DIR, cli_cfg=copy.deepcopy(cfg))
    cfg.ckpt.run_name = requested_run_name
    cfg.ckpt.workdir = requested_workdir
    cfg.ckpt.resume_step = requested_resume_step
    cfg.ckpt.max_to_keep = requested_max_to_keep

    # Derive fields after potential cfg restoration.
    cfg.data.history_batch = cfg.history_batch  # sync history batch size
    if int(cfg.enc.rope_max_len) < 512:
        old_rope_len = int(cfg.enc.rope_max_len)
        cfg.enc.rope_max_len = 1024
        print(
            "Warning: public TAM uses jointwise history tokens and needs longer RoPE context; "
            f"overriding enc.rope_max_len {old_rope_len} -> {cfg.enc.rope_max_len}."
        )

    CKPT_DIR = os.path.abspath(os.path.join(cfg.ckpt.workdir, cfg.ckpt.run_name))
    os.makedirs(CKPT_DIR, exist_ok=True)
    checkpoint_keep = _checkpoint_keep_limit(cfg)
    if checkpoint_keep == 0:
        print("Checkpoint retention: keeping every checkpoint step (ckpt.max_to_keep=0).")
    else:
        print(f"Checkpoint retention: keeping the latest {checkpoint_keep} checkpoint step(s).")

    sim_eval_cfg = getattr(cfg, "sim_eval", None)
    if sim_eval_cfg is None:
        raise RuntimeError("cfg.sim_eval must be configured.")

    dataset_root = Path(cfg.data.dataset_base_path).expanduser()
    all_robot_dataset_dirs = _discover_robot_dataset_dirs(dataset_root)
    train_robot_dataset_dirs = _filter_robot_dataset_dirs(all_robot_dataset_dirs, cfg.robot_key)
    dataset_eval_robot_keys = tuple(
        key.strip()
        for key in getattr(sim_eval_cfg, "dataset_eval_robot_key", ())
        if isinstance(key, str) and key.strip()
    )
    extra_eval_robot_dataset_dirs = (
        _filter_robot_dataset_dirs(all_robot_dataset_dirs, dataset_eval_robot_keys)
        if dataset_eval_robot_keys
        else []
    )
    robot_dataset_dirs: List[Path] = []
    for dataset_dir in [*train_robot_dataset_dirs, *extra_eval_robot_dataset_dirs]:
        if dataset_dir not in robot_dataset_dirs:
            robot_dataset_dirs.append(dataset_dir)
    train_dataset_dir_set = {dataset_dir.resolve() for dataset_dir in train_robot_dataset_dirs}
    selected_robot_keys = {
        key
        for dataset_dir in robot_dataset_dirs
        for key in _dataset_dir_robot_keys(dataset_dir)
    }
    print(
        f"Selected robot datasets: {[str(d) for d in robot_dataset_dirs]} "
        f"(robot_keys={sorted(selected_robot_keys)})"
    )
    if extra_eval_robot_dataset_dirs:
        print(
            "Held-out dataset eval robots: "
            f"{[str(d) for d in extra_eval_robot_dataset_dirs]}"
        )
    hz_base_preview, hz_choices_preview, hz_train_enable_preview = _resolve_effective_hz_choices(cfg)
    print(
        f"Active Hz settings: base_hz={hz_base_preview} choices={hz_choices_preview} "
        f"masking={'on' if hz_train_enable_preview else 'off'}"
    )
    is_multi_robot = len(robot_dataset_dirs) > 1
    _validate_ablation_config(cfg, is_multi_robot=is_multi_robot)
    print(f"Active ablation mode: {_ablation_mode(cfg)}")
    if is_multi_robot and bool(cfg.use_norm_stats):
        print("Multi-robot mode detected. Disabling norm stats for mixed-DoF training.")
        cfg.use_norm_stats = False

    dataset_chunk_size = 16
    batch_size_per_loader = cfg.history_batch // dataset_chunk_size
    if batch_size_per_loader < 1:
        raise ValueError(
            f"history_batch={cfg.history_batch} must be >= dataset_chunk_size={dataset_chunk_size}."
        )
    dataset_eval_batch_size_cfg = getattr(sim_eval_cfg, "dataset_eval_batch_size", None)
    if dataset_eval_batch_size_cfg is None:
        dataset_eval_batch_size = int(cfg.history_batch)
    else:
        dataset_eval_batch_size = int(dataset_eval_batch_size_cfg)
    if dataset_eval_batch_size > 0:
        if dataset_eval_batch_size < dataset_chunk_size:
            raise ValueError(
                f"sim_eval.dataset_eval_batch_size={dataset_eval_batch_size} must be >= "
                f"dataset_chunk_size={dataset_chunk_size}, or <= 0 to disable dataset eval loss."
            )
        if dataset_eval_batch_size % dataset_chunk_size != 0:
            raise ValueError(
                f"sim_eval.dataset_eval_batch_size={dataset_eval_batch_size} must be a multiple of "
                f"dataset_chunk_size={dataset_chunk_size}."
            )
        eval_batch_size_per_loader = dataset_eval_batch_size // dataset_chunk_size
    else:
        eval_batch_size_per_loader = 0
    if eval_batch_size_per_loader > 0:
        print(
            f"Dataset eval loss enabled: effective_batch={dataset_eval_batch_size} "
            f"(loader_batch={eval_batch_size_per_loader} shards x {dataset_chunk_size})."
        )
    else:
        print("Dataset eval loss disabled: sim_eval.dataset_eval_batch_size <= 0.")

    split_seed = cfg.seed
    train_fraction = 0.9
    robot_contexts: List[RobotTrainContext] = []
    used_robot_keys: set[str] = set()
    train_loader_num_workers = int(max(0, getattr(cfg, "num_workers", 0) or 0))
    if train_loader_num_workers > 0:
        if sys.platform.startswith("linux"):
            print(
                "[dataloader] Training loaders enabled with persistent workers "
                f"(num_workers={train_loader_num_workers}, context='forkserver')."
            )
        else:
            print(
                "[dataloader] Training loaders enabled with persistent workers "
                f"(num_workers={train_loader_num_workers}, default multiprocessing context)."
            )

    for ridx, dataset_dir in enumerate(robot_dataset_dirs):
        train_enabled = dataset_dir.resolve() in train_dataset_dir_set
        pert_train_paths, pert_eval_paths = split_shard_paths(
            str(dataset_dir),
            "perturbed",
            cfg.num_data_limit,
            train_fraction,
            split_seed + ridx,
        )
        if not pert_train_paths:
            raise ValueError(
                f"No training shards for robot dataset {dataset_dir}. "
                "Increase data size or adjust split settings."
            )
        if not pert_eval_paths:
            raise ValueError(
                f"No eval shards for robot dataset {dataset_dir}. "
                "Increase data size or adjust split settings."
            )
        first_pert_path = (
            pert_train_paths[0] if pert_train_paths else (pert_eval_paths[0] if pert_eval_paths else None)
        )

        has_external_force_field = False
        if first_pert_path is not None:
            try:
                has_external_force_field = _rollout_field_exists(first_pert_path, "external_force_ee")
            except Exception as e:
                print(f"Warning: failed to inspect external_force_ee from {first_pert_path}: {e}")
        perturbed_rollout_fields = ["q", "qd", "u"] + (["external_force_ee"] if has_external_force_field else [])

        perturbed_time_chunk = None
        rollout_time_cap = getattr(cfg, "max_rollout_time_chunk", None)
        if rollout_time_cap is not None:
            rollout_time_cap = int(rollout_time_cap)
            if rollout_time_cap < 1:
                raise ValueError(f"max_rollout_time_chunk must be >= 1, got {rollout_time_cap}")
        if first_pert_path is not None:
            try:
                inferred_time_chunk = _infer_rollout_time_length(first_pert_path)
                if inferred_time_chunk is None:
                    perturbed_time_chunk = rollout_time_cap
                elif rollout_time_cap is None:
                    perturbed_time_chunk = inferred_time_chunk
                else:
                    perturbed_time_chunk = min(inferred_time_chunk, rollout_time_cap)
            except Exception as e:
                print(f"Warning: failed to infer rollout length from {first_pert_path}: {e}")
                perturbed_time_chunk = rollout_time_cap
        elif rollout_time_cap is not None:
            perturbed_time_chunk = rollout_time_cap

        if perturbed_time_chunk is not None:
            min_required_time = max(
                int(cfg.enc.patch_size),
                int(cfg.adaptor_seq_length) + int(cfg.training_seq_length) + 2,
            )
            if perturbed_time_chunk < min_required_time:
                raise ValueError(
                    "Configured rollout time chunk is too short for training: "
                    f"time_chunk={perturbed_time_chunk}, required>={min_required_time} "
                    "(max(enc.patch_size, adaptor_seq_length + training_seq_length + 2))."
                )

        data_loader_train = _build_perturbed_dataloader(
            base_path=str(dataset_dir),
            fields=perturbed_rollout_fields,
            paths=pert_train_paths,
            time_chunk=perturbed_time_chunk,
            batch_size=batch_size_per_loader,
            shuffle=True,
            num_workers=train_loader_num_workers,
        )
        data_loader_eval = None
        if eval_batch_size_per_loader > 0:
            data_loader_eval = _build_perturbed_dataloader(
                base_path=str(dataset_dir),
                fields=perturbed_rollout_fields,
                paths=pert_eval_paths,
                time_chunk=perturbed_time_chunk,
                batch_size=eval_batch_size_per_loader,
                shuffle=True,
                num_workers=0,
                drop_last=False,
            )

        dataset_cfg_path = dataset_dir / "data_generation_config.json"
        with open(dataset_cfg_path, "r") as f:
            ds_cfg = json.load(f)
        if bool(ds_cfg.get("ideal_model_has_gravity", True)) is not True:
            raise ValueError(
                "Public TAM training supports only real-gravity datasets "
                f"(ideal_model_has_gravity=True). Please regenerate {dataset_cfg_path}."
            )
        ideal_model_has_gravity = True
        root_attrs_preview = _read_root_attrs(first_pert_path) if first_pert_path is not None else {}
        ee_body_name = root_attrs_preview.get("ee_payload_body_name", ds_cfg.get("ee_payload_body_name", None))
        ee_body_id = root_attrs_preview.get("ee_payload_body_id", ds_cfg.get("ee_payload_body_id", None))
        ee_payload_offset_min = root_attrs_preview.get(
            "ee_payload_com_offset_min_local_m",
            ds_cfg.get("ee_payload_com_offset_min_local_m", None),
        )
        ee_payload_offset_max = root_attrs_preview.get(
            "ee_payload_com_offset_max_local_m",
            ds_cfg.get("ee_payload_com_offset_max_local_m", None),
        )
        ee_mass_delta_range = root_attrs_preview.get(
            "ee_payload_mass_delta_range",
            ds_cfg.get("ee_payload_mass_delta_range", None),
        )
        joint_model_major_ee_scale = root_attrs_preview.get(
            "joint_model_major_ee_scale",
            ds_cfg.get("joint_model_major_ee_scale", None),
        )
        if (
            ee_body_name is not None
            and ee_body_id is not None
            and ee_payload_offset_min is not None
            and ee_payload_offset_max is not None
        ):
            print(
                f"[{dataset_dir.name}] EE COM randomization: "
                f"body='{ee_body_name}' (id={ee_body_id}), "
                f"payload_com_offset_box={ee_payload_offset_min}->{ee_payload_offset_max}, "
                f"payload_mass_delta_range={ee_mass_delta_range}, "
                f"joint_model_major_ee_scale={joint_model_major_ee_scale}"
            )

        xml_path_to_use, manifest_path, manifest = _resolve_dataset_robot_xml(dataset_dir, cfg.data.xml_path)
        mjx_model = dynamics.load_mjx_model_from_path(str(xml_path_to_use), remove_constraints=True)
        mjx_model = mjx_model.replace(body_gravcomp=jnp.zeros_like(mjx_model.body_gravcomp))

        profile_table_path_raw = str(ds_cfg.get("datagen_profile_table_path", "") or "")
        if not profile_table_path_raw:
            raise ValueError(
                f"Dataset config {dataset_cfg_path} missing required 'datagen_profile_table_path' "
                "for rollout_cmd_noise_std profile resolution."
            )
        profile_key_override_raw = ds_cfg.get("datagen_profile_key", None)
        profile_key_override = (
            str(profile_key_override_raw).strip() if profile_key_override_raw is not None else ""
        )
        if profile_key_override == "":
            profile_key_override = None
        try:
            profile_table_path = _resolve_profile_table_path(dataset_dir, profile_table_path_raw)
        except Exception as e:
            raise ValueError(
                f"[{dataset_dir}] failed to resolve datagen profile table from "
                f"datagen_profile_table_path='{profile_table_path_raw}': {e}"
            ) from e

        # Resolve profile key robustly for bundled robot.xml ("robot" stem) datasets.
        profile_candidates: List[str] = []
        if profile_key_override is not None:
            profile_candidates.append(profile_key_override)
        else:
            manifest_robot_key = str(manifest.get("robot_key", "")).strip() if manifest else ""
            if manifest_robot_key:
                profile_candidates.append(manifest_robot_key)
            manifest_source_xml = str(manifest.get("source_xml", "")).strip() if manifest else ""
            if manifest_source_xml:
                profile_candidates.append(derive_robot_key(Path(manifest_source_xml)))
            ds_cfg_xml = str(ds_cfg.get("xml_path", "") or "").strip()
            if ds_cfg_xml:
                profile_candidates.append(derive_robot_key(Path(ds_cfg_xml)))
            profile_candidates.append(dataset_dir.name)
            profile_candidates.append(derive_robot_key(xml_path_to_use))
        # stable dedupe preserving order
        profile_candidates = [k for i, k in enumerate(profile_candidates) if k and k not in profile_candidates[:i]]

        resolved_profile_key = None
        datagen_profile_kwargs = None
        candidate_errors: List[str] = []
        for profile_candidate in profile_candidates:
            try:
                resolved_profile_key, datagen_profile_kwargs = load_datagen_profile(
                    table_path=profile_table_path,
                    robot_key=profile_candidate,
                    profile_key=None,
                )
                break
            except KeyError as e:
                candidate_errors.append(f"{profile_candidate}: {e}")
                continue
            except Exception as e:
                raise ValueError(
                    f"[{dataset_dir}] failed to load datagen profile for candidate "
                    f"'{profile_candidate}'. table='{profile_table_path}'."
                ) from e
        if resolved_profile_key is None or datagen_profile_kwargs is None:
            raise ValueError(
                f"[{dataset_dir}] failed to resolve datagen profile for rollout_cmd_noise_std. "
                f"table='{profile_table_path}', profile_key_override={profile_key_override!r}, "
                f"tried_candidates={profile_candidates}. errors={candidate_errors}"
            )
        rollout_cmd_noise_std = datagen_profile_kwargs.get("rollout_cmd_noise_std", None)
        if rollout_cmd_noise_std is None:
            raise ValueError(
                f"[{dataset_dir}] datagen profile '{resolved_profile_key}' is missing required "
                "'rollout_cmd_noise_std'."
            )
        rollout_cmd_noise_std = float(rollout_cmd_noise_std)
        if not np.isfinite(rollout_cmd_noise_std):
            raise ValueError(
                f"[{dataset_dir}] datagen profile '{resolved_profile_key}' has non-finite "
                f"rollout_cmd_noise_std={rollout_cmd_noise_std}."
            )

        external_force_body_id = -1
        if has_external_force_field:
            root_attrs = _read_root_attrs(first_pert_path) if first_pert_path is not None else {}
            try:
                external_force_body_id, external_force_body_name = _resolve_external_force_body(
                    str(xml_path_to_use),
                    ds_cfg,
                    root_attrs=root_attrs,
                )
                print(
                    f"[{dataset_dir.name}] external-force body_id={external_force_body_id}, "
                    f"body_name='{external_force_body_name}'"
                )
            except Exception as e:
                print(
                    f"Warning: failed to resolve external-force body id for {dataset_dir} ({e}); "
                    "disabling force compensation."
                )
                external_force_body_id = -1

        robot_key_raw = str(manifest.get("robot_key", "")).strip() if manifest else ""
        robot_key = robot_key_raw or dataset_dir.name
        suffix = 1
        while robot_key in used_robot_keys:
            robot_key = f"{robot_key_raw or dataset_dir.name}_{suffix:02d}"
            suffix += 1
        used_robot_keys.add(robot_key)

        ctx = RobotTrainContext(
            robot_key=robot_key,
            dataset_dir=dataset_dir,
            train_enabled=bool(train_enabled),
            manifest_path=manifest_path,
            manifest=manifest,
            ds_cfg=ds_cfg,
            xml_path=xml_path_to_use,
            mjx_model=mjx_model,
            ideal_model_has_gravity=ideal_model_has_gravity,
            external_force_body_id=external_force_body_id,
            has_external_force_field=has_external_force_field,
            train_paths=pert_train_paths,
            eval_paths=pert_eval_paths,
            first_pert_path=first_pert_path,
            rollout_fields=perturbed_rollout_fields,
            time_chunk=perturbed_time_chunk,
            data_loader_train=data_loader_train,
            data_loader_eval=data_loader_eval,
            dof=int(mjx_model.nv),
            resolved_profile_key=str(resolved_profile_key),
            datagen_profile_kwargs=dict(datagen_profile_kwargs),
            rollout_cmd_noise_std=rollout_cmd_noise_std,
        )
        robot_contexts.append(ctx)

        print(
            f"[{ctx.robot_key}] dataset={ctx.dataset_dir} "
            f"shards(train/eval)={len(ctx.train_paths)}/{len(ctx.eval_paths)} "
            f"dof={ctx.dof} ext_force={ctx.has_external_force_field} "
            f"profile={resolved_profile_key} rollout_cmd_noise_std={ctx.rollout_cmd_noise_std:.4f} "
            f"train_enabled={ctx.train_enabled}"
        )

    if not robot_contexts:
        raise RuntimeError("No robot training context was created.")
    train_robot_contexts = [ctx for ctx in robot_contexts if ctx.train_enabled]
    if not train_robot_contexts:
        raise RuntimeError("No train-enabled robot context was created.")
    print(f"Training robot order: {[ctx.robot_key for ctx in train_robot_contexts]}")

    max_dof = max(ctx.dof for ctx in robot_contexts)
    if int(cfg.enc.max_dof_tokens) < int(max_dof):
        raise ValueError(
            f"enc.max_dof_tokens={cfg.enc.max_dof_tokens} is smaller than max discovered DoF={max_dof}. "
            "Increase enc.max_dof_tokens."
        )

    robot_context_by_key = {ctx.robot_key: ctx for ctx in robot_contexts}
    primary_ctx = robot_contexts[0]
    cfg.data.xml_path = str(primary_ctx.xml_path)
    print(f"Primary training robot: {primary_ctx.robot_key}")
    dataset_eval_contexts: List[RobotTrainContext] = []
    if bool(getattr(sim_eval_cfg, "dataset_eval_include_primary", True)):
        dataset_eval_contexts.append(primary_ctx)
    for requested_key in dataset_eval_robot_keys:
        ctx = robot_context_by_key.get(requested_key)
        if ctx is None:
            avail = ", ".join(robot_context_by_key.keys())
            raise ValueError(
                f"Unknown sim_eval.dataset_eval_robot_key='{requested_key}'. Available: {avail}"
            )
        if ctx not in dataset_eval_contexts:
            dataset_eval_contexts.append(ctx)
    if dataset_eval_contexts:
        print(f"Dataset eval robot order: {[ctx.robot_key for ctx in dataset_eval_contexts]}")
    else:
        print("Dataset eval robot order: []")

    # Save robot model bundles into checkpoint.
    robot_models_root = Path(CKPT_DIR) / "robot_models"
    robot_models_root.mkdir(parents=True, exist_ok=True)
    robot_index: Dict[str, Any] = {}
    for ctx in robot_contexts:
        manifest_out = _copy_robot_bundle_to_dir(
            xml_path=ctx.xml_path,
            manifest=ctx.manifest,
            dataset_dir=ctx.dataset_dir,
            robot_key=ctx.robot_key,
            dst_robot_dir=robot_models_root / ctx.robot_key,
        )
        robot_index[ctx.robot_key] = {
            "robot_key": ctx.robot_key,
            "dataset_dir": str(ctx.dataset_dir),
            "xml_path": str(ctx.xml_path),
            "xml_sha1": manifest_out.get("xml_sha1"),
            "ideal_model_has_gravity": bool(ctx.ideal_model_has_gravity),
            "dof": int(ctx.dof),
        }
    with open(robot_models_root / "index.json", "w") as f:
        json.dump(robot_index, f, indent=2, sort_keys=True)

    # Backward-compatibility: keep primary robot at CKPT_DIR/robot_model.
    _copy_robot_bundle_to_dir(
        xml_path=primary_ctx.xml_path,
        manifest=primary_ctx.manifest,
        dataset_dir=primary_ctx.dataset_dir,
        robot_key=primary_ctx.robot_key,
        dst_robot_dir=Path(CKPT_DIR) / "robot_model",
    )

    # ---- WandB ----
    try:
        import subprocess
        commit_msg = subprocess.check_output(
            ["git", "log", "-1", "--pretty=format:%h %s"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        commit_msg = "unknown"
    cfg_dict = json.loads(json.dumps(cfg, default=lambda o: o.__dict__))
    cfg_dict["git_commit"] = commit_msg
    cfg_dict["robot_keys"] = [ctx.robot_key for ctx in robot_contexts]
    cfg_dict["train_robot_keys"] = [ctx.robot_key for ctx in train_robot_contexts]
    cfg_dict["dataset_eval_robot_keys_resolved"] = [ctx.robot_key for ctx in dataset_eval_contexts]
    wandb_run_id = cfg.ckpt.run_name
    wandb_run = wandb.init(
        project=cfg.wandb.project,
        name=cfg.ckpt.run_name,
        mode=cfg.wandb.mode,
        group=cfg.wandb.group,
        notes=cfg.wandb.notes,
        tags=list(cfg.wandb.tags),
        config=cfg_dict,
        id=wandb_run_id,
        resume="allow",
    )
    artifact_interval = int(getattr(cfg.wandb, "artifact_interval", 0) or 0)
    artifact_type = str(getattr(cfg.wandb, "artifact_type", "model") or "model")
    artifact_name = getattr(cfg.wandb, "artifact_name", None)
    last_artifact_upload_step = -1
    if artifact_interval > 0:
        resolved_name = artifact_name or _default_ckpt_artifact_name(cfg.ckpt.run_name)
        print(
            "W&B artifact upload enabled: "
            f"interval={artifact_interval}, name='{resolved_name}', type='{artifact_type}'."
        )

    def build_dataset(rollout_data_perturbed):
        rollout_inputs = rollout_data_perturbed["rollout"]
        perturbed_rollout_params = rollout_data_perturbed["params"]
        rollout_inputs, perturbed_rollout_params = jax.tree.map(
            lambda x: jnp.array(x), (rollout_inputs, perturbed_rollout_params)
        )
        return rollout_inputs, perturbed_rollout_params

    # ---- RNGs ----
    rng = jax.random.PRNGKey(cfg.seed)
    rng, init_hist_key, init_adapt_key = jax.random.split(rng, 3)

    # ---- Instantiate modules ----
    adaptor = models.SimAdaptorJointwiseFlat(
        emb_dim=cfg.emb_dim,
        hidden=cfg.adaptor_hidden,
        depth=cfg.adaptor_depth,
    )
    for ctx in robot_contexts:
        ctx.history_enc_model = models_transformer.JointwiseFlatARTransformerDecoder(
            cfg=cfg.enc,
            emb_dim=cfg.emb_dim,
            ideal_mjx_model=ctx.mjx_model,
        )

    primary_hist_enc = primary_ctx.history_enc_model
    if primary_hist_enc is None:
        raise RuntimeError("Primary history encoder is not initialized.")

    # Jitted apply fns for quick eval helpers (primary robot only).
    history_enc_apply = jax.jit(
        lambda params_hist, q, qd, u, input_keep_mask, rng, deterministic, norm_stats: primary_hist_enc.apply(
            params_hist,
            q,
            qd,
            u,
            rngs={"dropout": rng},
            deterministic=deterministic,
            norm_stats=norm_stats,
            input_keep_mask=input_keep_mask,
        ),
        static_argnums=6,  # deterministic
    )
    patch_size = int(cfg.enc.patch_size)
    patch_stride = int(cfg.enc.patch_stride)

    def _pad_history_to_patch(x: jax.Array) -> jax.Array:
        pad_len = max(0, patch_size - int(x.shape[1]))
        if pad_len <= 0:
            return x
        pad = jnp.repeat(x[:, :1, :], pad_len, axis=1)
        return jnp.concatenate([pad, x], axis=1)

    def _pad_history_keep_mask(mask: Optional[jax.Array], batch_size: int, time_len: int) -> jax.Array:
        if mask is None:
            return jnp.ones((batch_size, time_len, 1), dtype=jnp.float32)
        mask = jnp.asarray(mask, dtype=jnp.float32)
        if mask.ndim == 2:
            mask = mask[..., None]
        elif mask.ndim != 3:
            raise ValueError(f"input_keep_mask must be rank 2 or 3, got {mask.shape}")
        if mask.shape[:2] != (batch_size, time_len):
            raise ValueError(
                f"input_keep_mask must match [B,T]={((batch_size), time_len)}, got {mask.shape}"
            )
        pad_len = max(0, patch_size - int(mask.shape[1]))
        if pad_len <= 0:
            return mask
        pad = jnp.repeat(mask[:, :1, :], pad_len, axis=1)
        return jnp.concatenate([pad, mask], axis=1)

    def _last_history_from_autoregressive(
        params_hist,
        q,
        qd,
        u,
        input_keep_mask,
        rng,
        deterministic,
        norm_stats,
    ):
        del deterministic, rng
        batch_size = int(q.shape[0])
        time_len = int(q.shape[1])
        q = _pad_history_to_patch(q)
        qd = _pad_history_to_patch(qd)
        u = _pad_history_to_patch(u)
        keep_mask = _pad_history_keep_mask(input_keep_mask, batch_size, time_len)
        q_tok = models.chunk_time(q, patch_size, patch_stride)
        qd_tok = models.chunk_time(qd, patch_size, patch_stride)
        u_tok = models.chunk_time(u, patch_size, patch_stride)
        keep_tok = models.chunk_time(keep_mask, patch_size, patch_stride)
        hist_seq, _ = models_transformer.online_history_update(
            params_hist,
            primary_hist_enc,
            q_tok,
            qd_tok,
            u_tok,
            cache=None,
            valid_mask=None,
            key=None,
            norm_stats=norm_stats,
            input_keep_mask=keep_tok,
        )
        if hist_seq.ndim == 4:
            return hist_seq[:, -1, :, :]
        return hist_seq[:, -1, :]

    history_last_apply = jax.jit(
        _last_history_from_autoregressive,
        static_argnums=6,  # deterministic
    )
    adaptor_apply = jax.jit(
        lambda params_adaptor, q, qd, tau, hist, rng, train, norm_stats: adaptor.apply(
            params_adaptor,
            q,
            qd,
            tau,
            hist,
            train=train,
            rngs={"dropout": rng},
            norm_stats=norm_stats,
        ),
        static_argnums=6,  # train
    )

    # Tau reconstruction setup (primary robot only).
    tau_recon_data = None
    tau_recon_path_str = str(
        _sim_eval_value(
            sim_eval_cfg,
            cfg,
            "real_recon_log_path",
            "",
            legacy_name="tau_recon_log_path",
        )
        or ""
    )
    tau_recon_path = Path(tau_recon_path_str) if tau_recon_path_str else None
    try:
        tau_recon_npz = np.load(tau_recon_path, allow_pickle=True)
        tau_recon_data = {k: tau_recon_npz[k] for k in tau_recon_npz.files}
        print(f"[tau-recon] Loaded {tau_recon_path} keys={tau_recon_npz.files}")
    except Exception as e:
        print(f"[tau-recon] Failed to load {tau_recon_path}: {e}")
        tau_recon_data = None

    tau_recon_tester = TauReconstructionTester(
        ideal_mjx_model=primary_ctx.mjx_model,
        history_apply_fn=history_enc_apply,
        history_last_apply_fn=history_last_apply,
        adaptor_apply_fn=adaptor_apply,
        external_force_body_id=primary_ctx.external_force_body_id,
        window_size=int(getattr(cfg, "adaptor_seq_length", 1) or 1),
        dt=float(
            _sim_eval_value(
                sim_eval_cfg,
                cfg,
                "real_recon_dt",
                1e-3,
                legacy_name="tau_recon_dt",
            )
            or 1e-3
        ),
        apply_zero_torque_mask=bool(
            _sim_eval_value(
                sim_eval_cfg,
                cfg,
                "real_recon_apply_zero_torque_mask",
                True,
            )
        ),
        zero_torque_threshold=float(
            _sim_eval_value(
                sim_eval_cfg,
                cfg,
                "real_recon_zero_torque_threshold",
                1e-5,
            )
            or 1e-5
        ),
        masked_fit_max_neighbors_each_side=int(
            _sim_eval_value(
                sim_eval_cfg,
                cfg,
                "real_recon_masked_fit_max_neighbors_each_side",
                getattr(cfg.enc, "masked_fit_max_neighbors_each_side", 50),
            )
            or getattr(cfg.enc, "masked_fit_max_neighbors_each_side", 50)
        ),
        masked_fit_q_weight=float(
            _sim_eval_value(
                sim_eval_cfg,
                cfg,
                "real_recon_masked_fit_q_weight",
                getattr(cfg.enc, "masked_fit_q_weight", 2.0),
            )
            or getattr(cfg.enc, "masked_fit_q_weight", 2.0)
        ),
        masked_fit_qd_weight=float(
            _sim_eval_value(
                sim_eval_cfg,
                cfg,
                "real_recon_masked_fit_qd_weight",
                getattr(cfg.enc, "masked_fit_qd_weight", 1.0),
            )
            or getattr(cfg.enc, "masked_fit_qd_weight", 1.0)
        ),
    )
    tau_recon_inputs = tau_recon_tester.prepare(tau_recon_data) if tau_recon_data is not None else None

    datasets_init = None
    for rollout_data_perturbed in tqdm.tqdm(
        primary_ctx.data_loader_train,
        total=len(primary_ctx.data_loader_train),
        desc=f"init-batch:{primary_ctx.robot_key}",
    ):
        datasets_init = build_dataset(rollout_data_perturbed)
        break
    if datasets_init is None:
        raise RuntimeError(f"Failed to fetch init batch from {primary_ctx.robot_key}.")

    # ---- Param init ----
    dof_init = primary_ctx.dof
    B = 2
    T_init = 1000
    q0 = jnp.zeros((B, T_init, dof_init))
    qd0 = jnp.zeros((B, T_init, dof_init))
    u0 = jnp.zeros((B, T_init, dof_init))

    hist_params = primary_hist_enc.init(
        {"params": init_hist_key, "dropout": init_hist_key},
        q0,
        qd0,
        u0,
    )

    q_cur = jnp.zeros((B, cfg.adaptor_seq_length, dof_init))
    qd_cur = jnp.zeros((B, cfg.adaptor_seq_length, dof_init))
    tau_cmd_init_window = jnp.zeros((B, cfg.adaptor_seq_length, dof_init))
    hist_emb0 = jnp.zeros((B, dof_init, cfg.emb_dim))
    adaptor_params = adaptor.init(init_adapt_key, q_cur, qd_cur, tau_cmd_init_window, hist_emb0)

    params = {"hist": hist_params, "adaptor": adaptor_params}

    # ---- Optimizer / Train state ----
    warmup_lr = 1e-5
    base_lr = cfg.lr
    warmup_steps = 5000
    lr_schedule = optax.join_schedules(
        schedules=[
            optax.constant_schedule(warmup_lr),
            optax.constant_schedule(base_lr),
        ],
        boundaries=[warmup_steps],
    )
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(lr_schedule, b1=0.9, b2=0.95, eps=1e-8, weight_decay=0.01),
    )
    use_norm_stats = bool(getattr(cfg, "use_norm_stats", True))
    state = DualModelState.create(
        apply_fn=None,
        params=params,
        tx=tx,
        norm_stats=init_norm_stats(dof_init) if use_norm_stats else None,
    )

    # ---- Checkpoint manager + resume ----
    state, resumed = restore_checkpoint_with_report(CKPT_DIR, state)
    if resumed:
        print(f"Resuming from epoch {int(state.epoch)} (opt step {int(state.step)}).")
        print("[resume] Best-effort data resume only: rebuilt dataloaders from scratch, so batch order and random crops may differ.")
    if not use_norm_stats and getattr(state, "norm_stats", None) is not None:
        state = state.replace(norm_stats=None)

    # Build per-robot JIT functions.
    for ctx in robot_contexts:
        if ctx.history_enc_model is None:
            raise RuntimeError(f"History encoder missing for robot {ctx.robot_key}")
        ctx.eval_func_jit = jax.jit(
            partial(
                loss_function,
                cfg=cfg,
                history_enc_model=ctx.history_enc_model,
                adaptor_model=adaptor,
                mjx_model_ideal=ctx.mjx_model,
                external_force_body_id=ctx.external_force_body_id,
                rollout_cmd_noise_std=ctx.rollout_cmd_noise_std,
                is_eval=True,
            )
        )
        ctx.train_step_jit = jax.jit(
            partial(
                train_step,
                cfg=cfg,
                history_enc_model=ctx.history_enc_model,
                adaptor_model=adaptor,
                mjx_model_ideal=ctx.mjx_model,
                external_force_body_id=ctx.external_force_body_id,
                rollout_cmd_noise_std=ctx.rollout_cmd_noise_std,
            )
        )

    if tau_recon_inputs is not None and bool(
        _sim_eval_value(
            sim_eval_cfg,
            cfg,
            "real_recon_warmup",
            True,
            legacy_name="tau_recon_warmup",
        )
    ):
        try:
            rng, warm_key = jax.random.split(rng)
            tau_recon_tester.warmup(
                warm_key,
                params=state.params,
                norm_stats=state.norm_stats,
                inputs=tau_recon_inputs,
                stride=int(
                    _sim_eval_value(
                        sim_eval_cfg,
                        cfg,
                        "real_recon_stride",
                        1,
                        legacy_name="tau_recon_stride",
                    )
                    or 1
                ),
                history_max_steps=_sim_eval_value(
                    sim_eval_cfg,
                    cfg,
                    "real_recon_hist_max_steps",
                    None,
                    legacy_name="tau_recon_hist_max_steps",
                ),
            )
        except Exception as e:
            print(f"[tau-recon] Warmup failed (continuing): {e}")

    # ---- Global step training loop with round-robin robot schedule ----
    robot_order = [ctx.robot_key for ctx in train_robot_contexts]
    train_iters = {ctx.robot_key: iter(ctx.data_loader_train) for ctx in train_robot_contexts}
    metric_accum_by_robot: Dict[str, Optional[Dict[str, jax.Array]]] = {k: None for k in robot_order}
    metric_count_by_robot: Dict[str, int] = {k: 0 for k in robot_order}
    t0 = time.time()
    internal_cnt = int(state.epoch)

    while internal_cnt <= cfg.max_steps:
        robot_key = robot_order[internal_cnt % len(robot_order)]
        ctx = robot_context_by_key[robot_key]

        try:
            rollout_data_perturbed = next(train_iters[robot_key])
        except StopIteration:
            train_iters[robot_key] = iter(ctx.data_loader_train)
            rollout_data_perturbed = next(train_iters[robot_key])

        datagen_time_start = time.time()
        datasets = build_dataset(rollout_data_perturbed)
        datasets = jax.block_until_ready(datasets)
        datagen_time_end = time.time()

        rng, step_key = jax.random.split(rng)
        trainstep_time_start = time.time()
        state, metrics = ctx.train_step_jit(state, step_key, datasets=datasets)
        metrics = jax.block_until_ready(metrics)
        trainstep_time_end = time.time()

        metrics = dict(metrics)
        metrics["data_gen_time"] = jnp.asarray(
            datagen_time_end - datagen_time_start if internal_cnt > 0 else 0.0,
            dtype=jnp.float32,
        )
        metrics["train_step_time"] = jnp.asarray(
            trainstep_time_end - trainstep_time_start if internal_cnt > 0 else 0.0,
            dtype=jnp.float32,
        )
        hist_input_T = int(datasets[0]["q"].shape[1])
        patch_size = int(cfg.enc.patch_size)
        patch_stride = max(1, int(cfg.enc.patch_stride))
        hist_tokens = max(1, 1 + (hist_input_T - patch_size) // patch_stride)
        metrics["hist_tokens"] = jnp.asarray(float(hist_tokens), dtype=jnp.float32)
        metrics["hist_joint_tokens"] = jnp.asarray(
            float(hist_tokens * int(datasets[0]["q"].shape[-1])),
            dtype=jnp.float32,
        )

        metric_accum_by_robot[robot_key] = _accumulate_metrics(metric_accum_by_robot[robot_key], metrics)
        metric_count_by_robot[robot_key] += 1

        if cfg.log_interval != 0 and internal_cnt % cfg.log_interval == 0:
            elapsed = time.time() - t0
            log: Dict[str, float] = {
                "step": float(internal_cnt),
                "sec_per_step": elapsed / float(cfg.log_interval) if internal_cnt > 0 else 0.0,
            }
            for key in robot_order:
                count = metric_count_by_robot[key]
                if count <= 0:
                    continue
                accum = metric_accum_by_robot[key]
                if accum is None:
                    continue
                for mk, mv in accum.items():
                    log[f"train/{key}/{mk}"] = float(np.asarray(mv / count))
            wandb.log(log, step=internal_cnt)
            t0 = time.time()
            metric_accum_by_robot = {k: None for k in robot_order}
            metric_count_by_robot = {k: 0 for k in robot_order}

        state = state.replace(epoch=int(state.epoch) + 1)

        if _should_run_periodic_interval(sim_eval_cfg, internal_cnt):
            if bool(getattr(sim_eval_cfg, "dataset_eval_enabled", True)):
                dataset_eval_num_batches = int(getattr(sim_eval_cfg, "dataset_eval_num_batches", 50))
                if dataset_eval_num_batches <= 0:
                    dataset_eval_num_batches = 0
                for eval_ctx in dataset_eval_contexts:
                    if eval_ctx.data_loader_eval is None:
                        continue
                    eval_log = None
                    log_cnt = 0
                    for rollout_data_eval in eval_ctx.data_loader_eval:
                        if log_cnt >= dataset_eval_num_batches:
                            break
                        datasets_eval = build_dataset(rollout_data_eval)
                        rng, eval_key = jax.random.split(rng)
                        _, aux = eval_ctx.eval_func_jit(
                            state.params,
                            eval_key,
                            datasets=datasets_eval,
                            norm_stats=state.norm_stats,
                        )

                        eval_log_ = {
                            f"eval/{eval_ctx.robot_key}/{k}": float(np.asarray(v))
                            for k, v in aux.items()
                        }
                        eval_log = eval_log_ if eval_log is None else {
                            k: eval_log[k] + eval_log_[k] for k in eval_log_
                        }
                        log_cnt += 1

                    if eval_log is not None and log_cnt > 0:
                        eval_log = {k: float(v / log_cnt) for k, v in eval_log.items()}
                        wandb.log(eval_log, step=internal_cnt)

            if bool(_sim_eval_value(sim_eval_cfg, cfg, "real_recon_enabled", True)):
                rng, recon_key = jax.random.split(rng)
                recon_metrics, recon_plot_path = tau_recon_tester.run(
                    recon_key,
                    tau_recon_inputs if tau_recon_inputs is not None else tau_recon_data,
                    params=state.params,
                    norm_stats=state.norm_stats,
                    cfg=sim_eval_cfg if sim_eval_cfg is not None else cfg,
                    out_dir=os.path.join(CKPT_DIR, "rollout_vis"),
                    prefix=f"tau_recon_{primary_ctx.robot_key}_{internal_cnt}",
                )
                if recon_metrics:
                    wandb.log(recon_metrics, step=internal_cnt)
                if recon_plot_path is not None:
                    wandb.log({"eval_plots/tau_recon": wandb.Image(recon_plot_path)}, step=internal_cnt)

                if bool(
                    _sim_eval_value(
                        sim_eval_cfg,
                        cfg,
                        "real_recon_rest_sweep_plot",
                        False,
                        legacy_name="rest_sweep_plot",
                    )
                ):
                    rest_plot_path = tau_recon_tester.plot_joint_sweep_from_real_logs(
                        params=state.params,
                        norm_stats=state.norm_stats,
                        out_dir=os.path.join(CKPT_DIR, "rollout_vis"),
                        act_log_path=str(
                            _sim_eval_value(
                                sim_eval_cfg,
                                cfg,
                                "real_recon_log_path",
                                "assets/test_traj/tau_verification_log.npz",
                                legacy_name="tau_recon_log_path",
                            )
                        ),
                        rest_log_path=str(
                            _sim_eval_value(
                                sim_eval_cfg,
                                cfg,
                                "real_recon_rest_log_path",
                                "assets/test_traj/rest_log.npz",
                                legacy_name="rest_log_path",
                            )
                        ),
                        prefix=f"rest_sweep_{primary_ctx.robot_key}_{internal_cnt}",
                        seed=int(getattr(cfg, "seed", 0) or 0),
                        tau_min=float(
                            _sim_eval_value(
                                sim_eval_cfg,
                                cfg,
                                "real_recon_rest_sweep_tau_min",
                                -1.5,
                                legacy_name="rest_sweep_tau_min",
                            )
                            or -1.5
                        ),
                        tau_max=float(
                            _sim_eval_value(
                                sim_eval_cfg,
                                cfg,
                                "real_recon_rest_sweep_tau_max",
                                1.5,
                                legacy_name="rest_sweep_tau_max",
                            )
                            or 1.5
                        ),
                        num_tau_samples=int(
                            _sim_eval_value(
                                sim_eval_cfg,
                                cfg,
                                "real_recon_rest_sweep_num_samples",
                                21,
                                legacy_name="rest_sweep_num_samples",
                            )
                            or 21
                        ),
                    )
                    if rest_plot_path is not None:
                        wandb.log({"eval_plots/rest_sweep": wandb.Image(rest_plot_path)}, step=internal_cnt)

        if cfg.ckpt_interval != 0 and internal_cnt % cfg.ckpt_interval == 0:
            opt_step = int(state.step)
            save_diction = {
                "step": opt_step,
                "opt_step": opt_step,
                "epoch": int(state.epoch),
                "train_loop_step": int(internal_cnt),
                "cfg": cfg,
                "params": state.params,
                "norm_stats": state.norm_stats,
            }
            checkpoints.save_checkpoint(CKPT_DIR, state, state.step, keep=checkpoint_keep)
            with open(os.path.join(CKPT_DIR, "save_dict.pkl"), "wb") as f:
                pickle.dump(save_diction, f)
            wandb.log({"ckpt/saved": internal_cnt}, step=internal_cnt)
            should_upload_artifact = (
                artifact_interval > 0
                and (
                    last_artifact_upload_step < 0
                    or (internal_cnt - last_artifact_upload_step) >= artifact_interval
                )
            )
            if should_upload_artifact:
                try:
                    uploaded = log_latest_checkpoint_artifact_to_wandb(
                        wandb_run=wandb_run,
                        ckpt_dir=CKPT_DIR,
                        step=internal_cnt,
                        run_name=cfg.ckpt.run_name,
                        artifact_name=artifact_name,
                        artifact_type=artifact_type,
                    )
                    if uploaded:
                        last_artifact_upload_step = internal_cnt
                        wandb.log({"ckpt/artifact_uploaded": internal_cnt}, step=internal_cnt)
                except Exception as e:
                    print(f"Warning: failed to upload W&B artifact at step {internal_cnt}: {e}")

        internal_cnt += 1

    checkpoints.save_checkpoint(CKPT_DIR, state, state.step, keep=checkpoint_keep)
    final_opt_step = int(state.step)
    save_diction = {
        "step": final_opt_step,
        "opt_step": final_opt_step,
        "epoch": int(state.epoch),
        "train_loop_step": int(internal_cnt),
        "cfg": cfg,
        "params": state.params,
        "norm_stats": state.norm_stats,
    }
    with open(os.path.join(CKPT_DIR, "save_dict.pkl"), "wb") as f:
        pickle.dump(save_diction, f)
    wandb.finish()


if __name__ == "__main__":
    cfg = parse_tyro_config(TrainConfig)
    main(cfg)
