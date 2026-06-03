from pathlib import Path
import sys

import numpy as np
import pytest


from tests.repo_paths import REPO_ROOT as ROOT
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from simadaptor.deploy import mapping_server  # noqa: E402


def _valid_window(*, t0: float = 0.0) -> list[dict]:
    q0 = np.linspace(0.1, 0.7, 7, dtype=np.float32)
    q1 = q0 + 0.01
    dq0 = np.linspace(0.2, 0.8, 7, dtype=np.float32)
    dq1 = dq0 + 0.01
    tau0 = np.linspace(0.3, 0.9, 7, dtype=np.float32)
    tau1 = tau0 + 0.01
    gravity = np.linspace(0.4, 1.0, 7, dtype=np.float32)
    return [
        {
            "t": float(t0 + 0.000),
            "q": q0.tolist(),
            "dq": dq0.tolist(),
            "tau_cmd": tau0.tolist(),
            "gravity": gravity.tolist(),
            "valid_for_history": True,
        },
        {
            "t": float(t0 + 0.001),
            "q": q1.tolist(),
            "dq": dq1.tolist(),
            "tau_cmd": tau1.tolist(),
            "gravity": gravity.tolist(),
            "valid_for_history": True,
        },
    ]


class _FakeClient:
    def __init__(self) -> None:
        self.enable_adaptor_best_effort_calls: list[bool] = []
        self.request_enable_adaptor_calls: list[bool] = []
        self.request_load_bin_blob_response_calls: list[tuple[str, str, object]] = []
        self.request_set_embedding_calls: list[np.ndarray] = []
        self.set_embedding_best_effort_calls: list[np.ndarray] = []
        self.request_set_ideal_model_has_gravity_calls: list[bool] = []
        self.request_load_bin_blob_calls: list[tuple[str, str]] = []
        self.send_command_calls: list[dict] = []

    def enable_adaptor_best_effort(self, enabled: bool) -> None:
        self.enable_adaptor_best_effort_calls.append(bool(enabled))

    def request_enable_adaptor(self, enabled: bool) -> bool:
        self.request_enable_adaptor_calls.append(bool(enabled))
        return True

    def request_set_embedding(self, embedding: np.ndarray) -> bool:
        self.request_set_embedding_calls.append(np.asarray(embedding, dtype=np.float32))
        return True

    def set_embedding_best_effort(self, embedding: np.ndarray) -> None:
        self.set_embedding_best_effort_calls.append(np.asarray(embedding, dtype=np.float32))

    def request_set_ideal_model_has_gravity(self, enabled: bool) -> bool:
        self.request_set_ideal_model_has_gravity_calls.append(bool(enabled))
        return True

    def request_load_bin_blob(self, name: str, blob: str) -> bool:
        self.request_load_bin_blob_calls.append((str(name), str(blob)))
        return True

    def request_load_bin_blob_response(
        self,
        name: str,
        blob: str,
        *,
        enable_after_load: object = None,
    ) -> dict:
        self.request_load_bin_blob_response_calls.append((str(name), str(blob), enable_after_load))
        return {"ok": True, "path": f"bin/{name}", "adaptor_enabled": bool(enable_after_load)}

    def send_command(self, **payload) -> None:
        self.send_command_calls.append(dict(payload))


class _FlakyPrepareClient(_FakeClient):
    def __init__(self, *, fail_set_ideal_calls: int = 0) -> None:
        super().__init__()
        self.fail_set_ideal_calls = int(fail_set_ideal_calls)

    def request_set_ideal_model_has_gravity(self, enabled: bool) -> bool:
        if self.fail_set_ideal_calls > 0:
            self.fail_set_ideal_calls -= 1
            raise RuntimeError("NUC request server not ready")
        return super().request_set_ideal_model_has_gravity(enabled)


class _FakeHistoryAdaptor:
    def __init__(self, outputs: list[np.ndarray | None]) -> None:
        self.outputs = list(outputs)
        self.reset_calls = 0
        self.push_calls: list[dict[str, np.ndarray | None]] = []

    def reset(self) -> None:
        self.reset_calls += 1

    def push_window(self, timestamps, q, dq, tau, gravity=None, *, keep_mask=None):
        self.push_calls.append(
            {
                "timestamps": np.asarray(timestamps, dtype=np.float64),
                "q": np.asarray(q, dtype=np.float32),
                "dq": np.asarray(dq, dtype=np.float32),
                "tau": np.asarray(tau, dtype=np.float32),
                "gravity": None if gravity is None else np.asarray(gravity, dtype=np.float32),
                "keep_mask": None if keep_mask is None else np.asarray(keep_mask, dtype=np.float32),
            }
        )
        if not self.outputs:
            return None
        output = self.outputs.pop(0)
        if output is None:
            return None
        return np.asarray(output, dtype=np.float32)


def test_tam_mapping_updater_keeps_uploaded_bin_for_direct_reset():
    client = _FakeClient()
    adaptor = _FakeHistoryAdaptor([np.arange(8, dtype=np.float32), np.ones(8, dtype=np.float32)])
    updater = mapping_server.SimAdaptorMappingUpdater(
        client=client,
        adaptor=adaptor,
        embedding_interval_s=0.0,
        min_patches_before_send=1,
        enable_after_first_embedding=True,
        reset_on_controller_reset=True,
        print_embedding=False,
        ideal_model_has_gravity=True,
        bin_name="tam.bin",
        bin_b64="ZmFrZQ==",
    )

    updater.prepare_remote_controller()
    sent = updater.process_once(window=_valid_window(), now=10.0)
    updater.process_once(reset_event={"reason": "command", "direct": True}, now=11.0)
    updater.process_once(window=_valid_window(t0=20.0), now=12.0)

    assert sent is True
    assert client.request_set_ideal_model_has_gravity_calls == [True, True]
    assert client.request_load_bin_blob_response_calls == [
        ("tam.bin", "ZmFrZQ==", False),
    ]
    assert client.enable_adaptor_best_effort_calls == [False, False]
    assert len(client.request_set_embedding_calls) == 4
    np.testing.assert_allclose(client.request_set_embedding_calls[0], np.zeros(0, dtype=np.float32))
    np.testing.assert_allclose(client.request_set_embedding_calls[1], np.arange(8, dtype=np.float32))
    np.testing.assert_allclose(client.request_set_embedding_calls[2], np.zeros(0, dtype=np.float32))
    np.testing.assert_allclose(client.request_set_embedding_calls[3], np.ones(8, dtype=np.float32))
    assert client.set_embedding_best_effort_calls == []
    assert client.request_enable_adaptor_calls == [False, False, True, True]
    assert adaptor.reset_calls == 1
    assert client.send_command_calls == []


def test_tam_mapping_updater_delays_enable_but_keeps_sending_embeddings():
    client = _FakeClient()
    adaptor = _FakeHistoryAdaptor(
        [
            np.arange(8, dtype=np.float32),
            np.full((8,), 2.0, dtype=np.float32),
        ]
    )
    updater = mapping_server.SimAdaptorMappingUpdater(
        client=client,
        adaptor=adaptor,
        embedding_interval_s=0.0,
        min_patches_before_send=1,
        enable_after_first_embedding=True,
        reset_on_controller_reset=True,
        print_embedding=False,
        ideal_model_has_gravity=True,
    )

    updater.prepare_remote_controller()
    updater.hold_enable_for(1.0, now=10.0)
    assert updater.process_once(window=_valid_window(), now=10.5) is True
    status_held = updater.status_payload(now=10.5)

    assert updater.enabled is False
    assert status_held["health"] == "enable_delayed"
    assert status_held["enable_hold_active"] is True
    assert len(client.request_set_embedding_calls) == 2
    np.testing.assert_allclose(client.request_set_embedding_calls[1], np.arange(8, dtype=np.float32))
    assert client.request_enable_adaptor_calls == [False]

    assert updater.process_once(window=_valid_window(t0=1.0), now=11.1) is True

    assert updater.enabled is True
    assert client.request_enable_adaptor_calls == [False]
    assert client.enable_adaptor_best_effort_calls[-1] is True
    np.testing.assert_allclose(client.set_embedding_best_effort_calls[0], np.full((8,), 2.0, dtype=np.float32))


def test_tam_mapping_updater_reset_requires_fresh_embedding_after_hold():
    client = _FakeClient()
    adaptor = _FakeHistoryAdaptor(
        [
            np.arange(8, dtype=np.float32),
            None,
            np.full((8,), 3.0, dtype=np.float32),
        ]
    )
    updater = mapping_server.SimAdaptorMappingUpdater(
        client=client,
        adaptor=adaptor,
        embedding_interval_s=0.0,
        min_patches_before_send=1,
        enable_after_first_embedding=True,
        reset_on_controller_reset=True,
        print_embedding=False,
        ideal_model_has_gravity=True,
    )

    updater.prepare_remote_controller()
    assert updater.process_once(window=_valid_window(), now=10.0) is True
    true_enables_before_reset = (
        sum(bool(v) for v in client.request_enable_adaptor_calls)
        + sum(bool(v) for v in client.enable_adaptor_best_effort_calls)
    )

    updater.hold_enable_for(1.0, now=11.0)
    updater.process_once(reset_event={"reason": "command", "direct": True}, now=11.0)
    updater.prepare_remote_controller()
    assert updater.current_embedding is None
    assert updater.num_sent == 0

    assert updater.process_once(window=_valid_window(t0=20.0), now=12.2) is False
    true_enables_after_empty_update = (
        sum(bool(v) for v in client.request_enable_adaptor_calls)
        + sum(bool(v) for v in client.enable_adaptor_best_effort_calls)
    )
    assert true_enables_after_empty_update == true_enables_before_reset
    assert updater.enabled is False

    assert updater.process_once(window=_valid_window(t0=21.0), now=12.3) is True
    assert updater.enabled is True
    np.testing.assert_allclose(client.request_set_embedding_calls[-1], np.full((8,), 3.0, dtype=np.float32))


def test_tam_mapping_updater_requires_control_before_enable():
    client = _FakeClient()
    adaptor = _FakeHistoryAdaptor(
        [
            np.arange(8, dtype=np.float32),
            np.full((8,), 2.0, dtype=np.float32),
        ]
    )
    updater = mapping_server.SimAdaptorMappingUpdater(
        client=client,
        adaptor=adaptor,
        embedding_interval_s=0.0,
        min_patches_before_send=1,
        enable_after_first_embedding=True,
        reset_on_controller_reset=True,
        print_embedding=False,
        ideal_model_has_gravity=True,
        require_control_enable=True,
    )

    updater.prepare_remote_controller()
    assert updater.process_once(window=_valid_window(), now=10.0) is True
    status_blocked = updater.status_payload(now=10.0)

    assert updater.enabled is False
    assert updater.control_enable_allowed is False
    assert status_blocked["health"] == "waiting_for_control_enable"
    assert client.request_enable_adaptor_calls == []
    assert client.enable_adaptor_best_effort_calls == [False]

    mapping_server._apply_control_enable_delay(
        updater,
        {"cmd": "reset", "direct": True, "enable_delay_s": 0.0},
        now=11.0,
    )
    updater.process_once(reset_event={"cmd": "reset", "direct": True}, now=11.0)
    updater.prepare_remote_controller()
    assert updater.process_once(window=_valid_window(t0=1.0), now=11.5) is True

    assert updater.control_enable_allowed is True
    assert updater.enabled is True
    assert client.request_enable_adaptor_calls[-1] is True


def test_tam_mapping_updater_reset_can_hold_control_enable():
    client = _FakeClient()
    adaptor = _FakeHistoryAdaptor(
        [
            np.arange(8, dtype=np.float32),
            np.full((8,), 2.0, dtype=np.float32),
        ]
    )
    updater = mapping_server.SimAdaptorMappingUpdater(
        client=client,
        adaptor=adaptor,
        embedding_interval_s=0.0,
        min_patches_before_send=1,
        enable_after_first_embedding=True,
        reset_on_controller_reset=True,
        print_embedding=False,
        ideal_model_has_gravity=True,
        require_control_enable=True,
    )

    updater.allow_control_enable(now=9.0)
    assert updater.process_once(window=_valid_window(), now=10.0) is True
    updater.process_once(reset_event={"cmd": "reset", "direct": True, "allow_enable": False}, now=11.0)
    updater.prepare_remote_controller()
    assert updater.process_once(window=_valid_window(t0=1.0), now=12.0) is True

    assert updater.control_enable_allowed is False
    assert updater.enabled is False
    assert updater.status_payload(now=12.0)["health"] == "waiting_for_control_enable"
    assert client.request_enable_adaptor_calls == [True]

    updater.allow_control_enable(now=13.0)
    assert updater.process_once(window=_valid_window(t0=2.0), now=13.0) is False
    assert updater.enabled is True
    assert client.enable_adaptor_best_effort_calls[-1] is True


def test_mapping_server_backend_choices_are_public_tam_only():
    parser = mapping_server.build_arg_parser()
    tam_args = parser.parse_args(["--backend", "tam"])

    assert tam_args.backend == "tam"
    assert tam_args.min_patches_before_send == 2
    assert mapping_server._mapping_mode_for_backend("tam") == "tam"
    with pytest.raises(SystemExit):
        parser.parse_args(["--backend", "legacy"])


def test_tam_mapping_updater_detects_controller_restart_from_time_rewind():
    client = _FakeClient()
    adaptor = _FakeHistoryAdaptor(
        [
            np.arange(8, dtype=np.float32),
            np.full((8,), 2.0, dtype=np.float32),
        ]
    )
    updater = mapping_server.SimAdaptorMappingUpdater(
        client=client,
        adaptor=adaptor,
        embedding_interval_s=0.0,
        min_patches_before_send=1,
        enable_after_first_embedding=True,
        reset_on_controller_reset=True,
        print_embedding=False,
        ideal_model_has_gravity=True,
        bin_name="tam.bin",
        bin_b64="ZmFrZQ==",
    )

    updater.prepare_remote_controller()
    assert updater.process_once(window=_valid_window(t0=5.0), now=10.0) is True
    assert updater.process_once(window=_valid_window(t0=0.0), now=20.0) is True

    assert adaptor.reset_calls == 1
    assert client.request_load_bin_blob_response_calls == [
        ("tam.bin", "ZmFrZQ==", False),
        ("tam.bin", "ZmFrZQ==", False),
    ]
    assert client.enable_adaptor_best_effort_calls == [False, False]
    assert client.request_enable_adaptor_calls == [False, False, True, False, False, True]
    assert len(client.request_set_embedding_calls) == 4
    np.testing.assert_allclose(client.request_set_embedding_calls[0], np.zeros(0, dtype=np.float32))
    np.testing.assert_allclose(client.request_set_embedding_calls[1], np.arange(8, dtype=np.float32))
    np.testing.assert_allclose(client.request_set_embedding_calls[2], np.zeros(0, dtype=np.float32))
    np.testing.assert_allclose(client.request_set_embedding_calls[3], np.full((8,), 2.0, dtype=np.float32))
    assert client.set_embedding_best_effort_calls == []
