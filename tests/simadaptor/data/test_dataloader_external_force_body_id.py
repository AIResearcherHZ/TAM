from pathlib import Path
import importlib.util
import sys

import numpy as np
import pytest
import zarr

from tests.repo_paths import REPO_ROOT as ROOT
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

MODULE_PATH = SRC_ROOT / "simadaptor" / "data" / "dataloader.py"
try:
    MODULE_SPEC = importlib.util.spec_from_file_location("test_dataloader_module", MODULE_PATH)
    dataloader_module = importlib.util.module_from_spec(MODULE_SPEC)
    assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
    MODULE_SPEC.loader.exec_module(dataloader_module)
    PandaRolloutShardDataset = dataloader_module.PandaRolloutShardDataset
    concat_shard_collate = dataloader_module.concat_shard_collate
    DATALOADER_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment dependent
    PandaRolloutShardDataset = None
    concat_shard_collate = None
    DATALOADER_IMPORT_ERROR = exc


def _write_array(group: zarr.Group, name: str, arr: np.ndarray) -> None:
    z = group.create_array(
        name=name,
        shape=arr.shape,
        dtype=arr.dtype,
        chunks=arr.shape,
    )
    z[:] = arr


def _write_test_shard(path: Path, *, body_id: int, batch_size: int = 2, time_len: int = 4, dof: int = 3) -> None:
    root = zarr.open_group(str(path), mode="w")
    g_rollout = root.create_group("rollout")
    g_params = root.create_group("params")

    q = np.zeros((batch_size, time_len, dof), dtype=np.float32)
    qd = np.full((batch_size, time_len, dof), 0.1, dtype=np.float32)
    u = np.zeros((batch_size, time_len, dof), dtype=np.float32)
    times = np.linspace(0.0, 0.003, num=time_len, dtype=np.float32)[None, :].repeat(batch_size, axis=0)

    _write_array(g_rollout, "q", q)
    _write_array(g_rollout, "qd", qd)
    _write_array(g_rollout, "u", u)
    _write_array(g_rollout, "times", times)

    _write_array(g_params, "kp", np.ones((batch_size, dof), dtype=np.float32))
    _write_array(g_params, "kd", np.ones((batch_size, dof), dtype=np.float32))
    _write_array(g_params, "deadzone", np.zeros((batch_size, dof, 2), dtype=np.float32))
    _write_array(g_params, "torque_range", np.ones((batch_size, dof, 2), dtype=np.float32))
    _write_array(g_params, "torque_bias", np.zeros((batch_size, dof, 2), dtype=np.float32))
    _write_array(g_params, "damping", np.zeros((batch_size, dof, 2), dtype=np.float32))
    _write_array(g_params, "friction_params", np.zeros((batch_size, dof, 6), dtype=np.float32))
    _write_array(g_params, "torque_scale", np.ones((batch_size, dof, 2), dtype=np.float32))
    _write_array(g_params, "dof_frictionloss", np.zeros((batch_size, dof), dtype=np.float32))
    _write_array(g_params, "dof_armature", np.ones((batch_size, dof), dtype=np.float32))
    _write_array(g_params, "dof_damping", np.zeros((batch_size, dof), dtype=np.float32))
    _write_array(g_params, "body_mass", np.ones((batch_size, 2), dtype=np.float32))
    _write_array(g_params, "body_inertia", np.ones((batch_size, 2, 3), dtype=np.float32))
    _write_array(g_params, "body_ipos", np.zeros((batch_size, 2, 3), dtype=np.float32))

    root.attrs["external_force_body_id"] = int(body_id)


@pytest.mark.skipif(
    DATALOADER_IMPORT_ERROR is not None,
    reason=f"Dataloader dependencies unavailable: {DATALOADER_IMPORT_ERROR}",
)
def test_dataloader_injects_external_force_body_id_from_shard_attrs(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    perturbed_dir = dataset_root / "perturbed"
    perturbed_dir.mkdir(parents=True, exist_ok=True)

    _write_test_shard(perturbed_dir / "shard0000.zarr", body_id=3)
    _write_test_shard(perturbed_dir / "shard0001.zarr", body_id=5)

    ds = PandaRolloutShardDataset(
        base_path=str(dataset_root),
        split="perturbed",
        fields=["q", "qd", "u", "times"],
    )

    shard0 = ds[0]
    shard1 = ds[1]
    np.testing.assert_array_equal(
        np.asarray(shard0["rollout"]["external_force_body_id"]),
        np.asarray([3, 3], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(shard1["rollout"]["external_force_body_id"]),
        np.asarray([5, 5], dtype=np.int32),
    )

    collated = concat_shard_collate([shard0, shard1])
    np.testing.assert_array_equal(
        np.asarray(collated["rollout"]["external_force_body_id"]),
        np.asarray([3, 3, 5, 5], dtype=np.int32),
    )


@pytest.mark.skipif(
    DATALOADER_IMPORT_ERROR is not None,
    reason=f"Dataloader dependencies unavailable: {DATALOADER_IMPORT_ERROR}",
)
def test_dataloader_retires_permanently_bad_zip_shard(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    perturbed_dir = dataset_root / "perturbed"
    perturbed_dir.mkdir(parents=True, exist_ok=True)

    bad_path = perturbed_dir / "bad.zarr.zip"
    bad_path.write_bytes(b"not a valid zip")
    good_path = perturbed_dir / "good.zarr"
    _write_test_shard(good_path, body_id=7)

    ds = PandaRolloutShardDataset(
        base_path=str(dataset_root),
        split="perturbed",
        fields=["q", "qd", "u", "times"],
        paths=[str(bad_path), str(good_path)],
    )

    shard = ds[0]
    assert str(bad_path) in ds._bad_paths
    np.testing.assert_array_equal(
        np.asarray(shard["rollout"]["external_force_body_id"]),
        np.asarray([7, 7], dtype=np.int32),
    )
