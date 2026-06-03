import os, glob
import random
import zipfile
from typing import Dict, Any, Optional, Iterable, List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
import zarr


def _open_zarr_group(path: str, mode: str = "r"):
    """Open a Zarr group from directory or ZipStore and return (group, closer)."""
    if path.endswith(".zip"):
        store = zarr.storage.ZipStore(path, mode=mode)
        g = zarr.open_group(store=store, mode=mode)
        return g, store.close
    return zarr.open_group(path, mode=mode), (lambda: None)


def _is_permanent_shard_error(exc: BaseException) -> bool:
    if isinstance(exc, zipfile.BadZipFile):
        return True
    msg = str(exc).lower()
    return (
        "file is not a zip file" in msg
        or "not a zip file" in msg
        or "bad magic number" in msg
        or "not a zarr" in msg
    )

def _infer_ref_T(g_rollout: zarr.Group, fields: Optional[List[str]], Bshard: int) -> Optional[int]:
    """Pick an actions-like time length T (min over (B, Ti, ...) arrays)."""
    names = fields or [name for name, _ in g_rollout.arrays()]
    time_lens = []
    for name in names:
        arr = g_rollout[name]
        shp = arr.shape
        if len(shp) >= 2 and shp[0] == Bshard:
            time_lens.append(shp[1])
    return min(time_lens) if time_lens else None


class PandaRolloutShardDataset(Dataset):
    """
    One sample == one Zarr shard (file). Each __getitem__ loads all Bshard items (e.g., 16).
    Use DataLoader(batch_size = num_files_per_batch, collate_fn=concat_shard_collate) to
    concatenate across shards into a big batch.
    """
    _MAX_GETITEM_RETRIES = 4

    def __init__(
        self,
        base_path: str,
        split: str = "perturbed",
        time_chunk: Optional[int] = None,
        fields: Optional[Iterable[str]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        num_data_limit: Optional[int] = None,
        paths: Optional[List[str]] = None,
    ):
        super().__init__()
        self.base_path = base_path
        self.split = split
        self.time_chunk = time_chunk
        self.fields = None if fields is None else list(fields)
        self.device = device
        self.dtype = dtype
        self.num_data_limit = num_data_limit

        if paths is not None:
            self.paths = list(paths)
        else:
            zips = glob.glob(os.path.join(base_path, split, "*.zarr.zip"))
            dirs = glob.glob(os.path.join(base_path, split, "*.zarr"))
            self.paths = sorted(zips + dirs)
        if not self.paths:
            raise FileNotFoundError(f"No .zarr files under {base_path}/{split}")
        if self.num_data_limit is not None:
            self.paths = self.paths[:self.num_data_limit]
        self._bad_paths: set[str] = set()

        # just remember shard paths; no open or cache here
        mjx_params_files = ['dof_frictionloss', 'dof_armature', 'dof_damping', 'body_mass', 'body_inertia', 'body_ipos']
        if split == "perturbed":
            self.params_fields = [
                'kp',
                'kd',
                'deadzone',
                'torque_range',
                'torque_bias',
                'damping',
                'friction_params',
                'torque_scale',
                *mjx_params_files,
            ]
        elif split == "original":
            self.params_fields = ['kp', 'kd', 'torque_range', *mjx_params_files]

    def __len__(self) -> int:
        return len(self.paths)

    def _choose_retry_index(self, current_idx: int) -> Optional[int]:
        candidates = [
            i for i, path in enumerate(self.paths)
            if path not in self._bad_paths and i != int(current_idx)
        ]
        if not candidates:
            candidates = [
                i for i, path in enumerate(self.paths)
                if path not in self._bad_paths
            ]
        if not candidates:
            return None
        return int(candidates[int(np.random.randint(0, len(candidates)))])

    def __getitem__(self, file_idx: int):
        original_idx = int(file_idx)
        current_idx = original_idx
        failures: List[str] = []

        for attempt in range(self._MAX_GETITEM_RETRIES):
            closer = lambda: None
            if self.paths[current_idx] in self._bad_paths:
                retry_idx = self._choose_retry_index(current_idx)
                if retry_idx is None:
                    break
                current_idx = retry_idx
            path = self.paths[current_idx]
            try:
                g, closer = _open_zarr_group(path, mode="r")  # open on demand

                # --- rollout ---
                g_rollout = g["rollout"]
                fields = self.fields or [name for name, _ in g_rollout.arrays()]

                # infer shard batch size & reference T
                _, arr0 = next(iter(g_rollout.arrays()))
                Bshard = arr0.shape[0]
                ref_T = _infer_ref_T(g_rollout, fields, Bshard)

                assert Bshard > 0, f"Invalid shard batch size in file {path}"
                assert ref_T is None or ref_T > 0, f"Invalid reference time length in file {path}"

                # choose time crop
                if self.time_chunk is None or ref_T is None:
                    t0 = 0
                    L = None
                else:
                    L = min(int(self.time_chunk), int(ref_T))
                    max_start = max(0, ref_T - L)
                    t0 = 0 if max_start == 0 else np.random.randint(0, max_start)

                rollout: Dict[str, torch.Tensor] = {}
                for name in fields:
                    arr: zarr.Array = g_rollout[name]
                    shp = arr.shape

                    if len(shp) >= 2 and shp[0] == Bshard:
                        Ti = shp[1]
                        if L is None:
                            np_item = arr[:]
                        else:
                            desired = L
                            if ref_T is not None and Ti == ref_T + 1:
                                desired = L + 1
                            desired = min(desired, Ti)
                            t0_this = min(t0, max(0, Ti - desired))
                            t1_this = t0_this + desired
                            np_item = arr[:, t0_this:t1_this]
                        t = torch.from_numpy(np.asarray(np_item))
                    else:
                        raise ValueError(f"Rollout field '{name}' has invalid shape {shp} in file {path}")

                    if self.dtype is not None and t.dtype.is_floating_point:
                        t = t.to(self.dtype)
                    if self.device is not None:
                        t = t.to(self.device, non_blocking=True)

                    if name == "qd":
                        qd_abs_max = torch.max(torch.abs(t))
                        qd_threshold = 10.0
                        if torch.isnan(qd_abs_max) or qd_abs_max > qd_threshold:
                            raise ValueError(
                                f"qd contains invalid values: max |qd|={float(qd_abs_max):.3f} exceeds threshold {qd_threshold}"
                            )

                    rollout[name] = t

                assert all(rollout[k].shape[:2] == rollout[list(rollout.keys())[0]].shape[:2] for k in rollout.keys())

                # --- params ---
                g_params = g["params"]
                params: Dict[str, torch.Tensor] = {}
                for name in self.params_fields:
                    if name in g_params:
                        arr = g_params[name]
                        t = torch.from_numpy(np.asarray(arr[:]))
                        if self.dtype is not None and t.dtype.is_floating_point:
                            t = t.to(self.dtype)
                        if self.device is not None:
                            t = t.to(self.device, non_blocking=True)
                        params[name] = t
                    else:
                        raise KeyError(f"Param field '{name}' not found in file {path}")

                assert all((v is None) or (v.shape[0] == Bshard) for v in params.values())

                for k, v in rollout.items():
                    if torch.is_floating_point(v) and not torch.all(torch.isfinite(v)):
                        raise ValueError(f"Non-finite values in rollout field '{k}' from file {path}")
                for k, v in params.items():
                    if v is not None and torch.is_floating_point(v) and not torch.all(torch.isfinite(v)):
                        raise ValueError(f"Non-finite values in params field '{k}' from file {path}")

                body_id_attr = g.attrs.get("external_force_body_id", None)
                if body_id_attr is not None:
                    try:
                        body_id = int(body_id_attr)
                        rollout["external_force_body_id"] = torch.full(
                            (int(Bshard),),
                            body_id,
                            dtype=torch.int32,
                        )
                        if self.device is not None:
                            rollout["external_force_body_id"] = rollout["external_force_body_id"].to(
                                self.device,
                                non_blocking=True,
                            )
                    except Exception as e:
                        raise ValueError(
                            f"Invalid external_force_body_id attr {body_id_attr!r} in file {path}: {e}"
                        ) from e

                return {"rollout": rollout, "params": params}
            except Exception as e:
                failures.append(f"{path}: {e}")
                if _is_permanent_shard_error(e):
                    self._bad_paths.add(path)
                print(
                    f"Error loading shard dataset file {path} "
                    f"(attempt {attempt + 1}/{self._MAX_GETITEM_RETRIES}): {e}"
                )
                retry_idx = self._choose_retry_index(current_idx)
                if retry_idx is not None:
                    current_idx = retry_idx
            finally:
                try:
                    closer()
                except Exception:
                    pass

        failure_msg = "; ".join(failures)
        raise RuntimeError(
            "Failed to load shard dataset item after bounded retries. "
            f"original_path={self.paths[original_idx]}; failures={failure_msg}"
        )


def _concat_collate_items(items: List[Any]) -> Any:
    first = items[0]
    if isinstance(first, torch.Tensor):
        return torch.cat(items, dim=0)
    if isinstance(first, dict):
        return {key: _concat_collate_items([item[key] for item in items]) for key in first}
    if isinstance(first, tuple):
        return tuple(_concat_collate_items([item[idx] for item in items]) for idx in range(len(first)))
    if isinstance(first, list):
        return [_concat_collate_items([item[idx] for item in items]) for idx in range(len(first))]
    return first


def seed_dataloader_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = int(torch.initial_seed() % (2**32))
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def concat_shard_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    batch: list of N items from PandaRolloutShardDataset (each is a shard of size Bshard).
    Concatenate along batch dim 0 so output has size N*Bshard.
    """
    assert len(batch) > 0

    return _concat_collate_items(batch)
    
    # # rollout keys union (assume same set across shards)
    # rollout_keys = batch[0]["rollout"].keys()
    # params_keys = batch[0]["params"].keys()

    # # concat rollout
    # out_rollout: Dict[str, torch.Tensor] = {}
    # for k in rollout_keys:
    #     tensors = [b["rollout"][k] for b in batch]
    #     # if first dim is batch (common), concat on dim 0; else stack then reshape if needed
    #     out_rollout[k] = torch.cat(tensors, dim=0)

    # # concat params (most are (Bshard, ...); if not, try to stack)
    # out_params: Dict[str, torch.Tensor] = {}
    # for k in params_keys:
    #     tensors = [b["params"][k] for b in batch]
    #     out_params[k] = torch.cat(tensors, dim=0)

    # metas as a list (keep per-shard)
    # metas = [b["meta"] for b in batch]
    # return {"rollout": out_rollout, "params": out_params}


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import torch

    ds = PandaRolloutShardDataset(
        base_path="tmp",
        split="original",            # or "original" or "perturbed"
        time_chunk=1000,              # or None
        fields=['q', 'qd', 'times'],                 # or ["q", "qd", "u", ...]
    )

    # Suppose each shard has 16 samples. To form total batch B=128:
    loader = DataLoader(
        ds,
        batch_size=2,                # 8 shards × 16 = 128 samples
        shuffle=True,                # shuffle shards
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=concat_shard_collate,
        drop_last=True,              # keeps shard count consistent
    )
    from tqdm import tqdm
    for batch in tqdm(loader):
        # batch["rollout"]["q"]: (B=128, L or full T, Dq)
        # batch["params"]["kp"]: (B=128, ...)  (expanded if scalar)
        pass
