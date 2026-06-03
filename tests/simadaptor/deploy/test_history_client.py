from __future__ import annotations

import numpy as np

from simadaptor.deploy.history_client import HistoryControllerClient


class _FakePush:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def send_json(self, payload: dict) -> None:
        self.payloads.append(dict(payload))


class _FakeFastTransport:
    def __init__(self) -> None:
        self.actions: list[dict] = []

    def get_recent_state(self, **kwargs):
        del kwargs
        return None

    def send_action(self, **kwargs):
        self.actions.append(dict(kwargs))
        return 1


def _bare_client() -> HistoryControllerClient:
    client = HistoryControllerClient.__new__(HistoryControllerClient)
    client.push = _FakePush()
    client.fast_transport = _FakeFastTransport()
    client._command_source = "unit-test"
    client._next_command_id = 0
    client._fast_action_requires_state = False
    client._fast_state_max_age_s = 0.05
    client._fast_poll_timeout_ms = 0
    return client


def test_send_command_can_force_json_for_gain_update() -> None:
    client = _bare_client()
    stiffness = np.arange(7, dtype=np.float64) + 10.0
    damping = np.arange(7, dtype=np.float64) + 1.0

    client.send_command(
        stiffness=stiffness,
        damping=damping,
        prefer_fast_transport=False,
    )

    assert client.fast_transport.actions == []
    assert len(client.push.payloads) == 1
    payload = client.push.payloads[0]
    np.testing.assert_allclose(payload["stiffness"], stiffness)
    np.testing.assert_allclose(payload["damping"], damping)
    assert payload["command_source"] == "unit-test"
    assert payload["command_id"] == 1


def test_send_command_uses_fast_transport_by_default_for_joint_targets() -> None:
    client = _bare_client()
    q = np.arange(7, dtype=np.float64)

    client.send_command(target_q=q, target_dq=np.zeros(7, dtype=np.float64))

    assert client.push.payloads == []
    assert len(client.fast_transport.actions) == 1
    np.testing.assert_allclose(client.fast_transport.actions[0]["target_q"], q)
