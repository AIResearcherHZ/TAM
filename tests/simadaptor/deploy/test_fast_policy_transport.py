from __future__ import annotations

import numpy as np
import pytest

from simadaptor.deploy.fast_policy_transport import (
    ACTION_HAS_DAMPING,
    ACTION_HAS_FEEDFORWARD,
    ACTION_HAS_STIFFNESS,
    ACTION_HAS_TARGET_DQ,
    ACTION_HAS_TARGET_Q,
    ACTION_PROTOCOL_VERSION,
    ACTION_STRUCT,
    LEGACY_STATE_PROTOCOL_VERSION,
    LEGACY_STATE_STRUCT,
    pack_action_message,
    unpack_state_message,
    STATE_MAGIC,
    STATE_HAS_O_T_EE,
    STATE_PROTOCOL_VERSION,
    STATE_STRUCT,
    overlay_fast_state_sample,
)


def test_state_message_unpack_and_overlay() -> None:
    q = np.arange(7, dtype=np.float64)
    dq = q + 10.0
    tau = q + 20.0
    O_T_EE = np.array(
        [
            [1.0, 0.0, 0.0, 0.35],
            [0.0, 0.0, -1.0, -0.05],
            [0.0, 1.0, 0.0, 0.42],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    payload = STATE_STRUCT.pack(
        STATE_MAGIC,
        STATE_PROTOCOL_VERSION,
        STATE_HAS_O_T_EE,
        42,
        1.25,
        123456,
        *np.concatenate([q, dq, tau, O_T_EE.reshape(-1, order="F")]).tolist(),
    )

    state = unpack_state_message(payload, received_wall_time_sec=9.0)
    assert state.seq == 42
    assert state.controller_time_sec == pytest.approx(1.25)
    np.testing.assert_allclose(state.q, q)
    np.testing.assert_allclose(state.dq, dq)
    np.testing.assert_allclose(state.tau_cmd, tau)
    np.testing.assert_allclose(state.O_T_EE, O_T_EE)

    merged = overlay_fast_state_sample({}, state)
    assert merged["state_transport"] == "fast"
    assert merged["fast_state_seq"] == 42
    np.testing.assert_allclose(merged["q"], q)
    np.testing.assert_allclose(merged["O_T_EE"], O_T_EE)


def test_legacy_state_message_unpack_without_pose() -> None:
    q = np.arange(7, dtype=np.float64)
    dq = q + 10.0
    tau = q + 20.0
    payload = LEGACY_STATE_STRUCT.pack(
        STATE_MAGIC,
        LEGACY_STATE_PROTOCOL_VERSION,
        0,
        42,
        1.25,
        123456,
        *np.concatenate([q, dq, tau]).tolist(),
    )

    state = unpack_state_message(payload, received_wall_time_sec=9.0)
    assert state.seq == 42
    np.testing.assert_allclose(state.q, q)
    assert state.O_T_EE is None


def test_action_message_pack_flags_and_arrays() -> None:
    q = np.arange(7, dtype=np.float64)
    dq = q + 1.0
    kp = q + 2.0
    kd = q + 3.0
    tau = q + 4.0

    payload = pack_action_message(
        seq=7,
        observed_state_seq=42,
        target_q=q,
        target_dq=dq,
        stiffness=kp,
        damping=kd,
        feedforward=tau,
        host_send_steady_ns=99,
    )
    unpacked = ACTION_STRUCT.unpack(payload)
    flags = unpacked[2]
    assert flags == (
        ACTION_HAS_TARGET_Q
        | ACTION_HAS_TARGET_DQ
        | ACTION_HAS_STIFFNESS
        | ACTION_HAS_DAMPING
        | ACTION_HAS_FEEDFORWARD
    )
    assert unpacked[3] == 7
    assert unpacked[4] == 42
    assert unpacked[1] == ACTION_PROTOCOL_VERSION
    arrays = np.asarray(unpacked[6:], dtype=np.float64).reshape(5, 7)
    np.testing.assert_allclose(arrays[0], q)
    np.testing.assert_allclose(arrays[1], dq)
    np.testing.assert_allclose(arrays[2], kp)
    np.testing.assert_allclose(arrays[3], kd)
    np.testing.assert_allclose(arrays[4], tau)


def test_action_message_requires_payload() -> None:
    with pytest.raises(ValueError):
        pack_action_message(seq=1)
