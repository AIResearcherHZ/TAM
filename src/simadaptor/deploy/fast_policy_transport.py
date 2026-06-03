"""Low-latency binary transport for latest robot state and joint actions."""

from __future__ import annotations

from dataclasses import dataclass
import struct
import time
from typing import Any, Optional

import numpy as np

try:
    import zmq
except ModuleNotFoundError:  # Keep binary pack/unpack helpers importable in minimal test envs.
    zmq = None


FAST_STATE_ENDPOINT = "tcp://192.168.1.101:5558"
FAST_ACTION_ENDPOINT = "tcp://192.168.1.101:5559"

LEGACY_STATE_PROTOCOL_VERSION = 1
STATE_PROTOCOL_VERSION = 2
ACTION_PROTOCOL_VERSION = 1
PROTOCOL_VERSION = ACTION_PROTOCOL_VERSION
STATE_MAGIC = b"S2FS"
ACTION_MAGIC = b"S2FA"

STATE_HAS_O_T_EE = 1 << 0

ACTION_HAS_TARGET_Q = 1 << 0
ACTION_HAS_TARGET_DQ = 1 << 1
ACTION_HAS_STIFFNESS = 1 << 2
ACTION_HAS_DAMPING = 1 << 3
ACTION_HAS_FEEDFORWARD = 1 << 4

STATE_HEADER_STRUCT = struct.Struct("<4sHHQdQ")
LEGACY_STATE_STRUCT = struct.Struct("<4sHHQdQ21d")
STATE_STRUCT = struct.Struct("<4sHHQdQ37d")
ACTION_STRUCT = struct.Struct("<4sHHQQQ35d")


@dataclass(frozen=True)
class FastStateSample:
    seq: int
    controller_time_sec: float
    nuc_send_steady_ns: int
    q: np.ndarray
    dq: np.ndarray
    tau_cmd: np.ndarray
    received_wall_time_sec: float
    received_monotonic_ns: int
    O_T_EE: Optional[np.ndarray] = None

    @property
    def receive_age_sec(self) -> float:
        return max(0.0, (time.monotonic_ns() - int(self.received_monotonic_ns)) / 1.0e9)


def _as_vector7(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != 7:
        raise ValueError(f"{name} must contain exactly 7 values, got {array.size}.")
    return array.copy()


def unpack_state_message(payload: bytes, *, received_wall_time_sec: Optional[float] = None) -> FastStateSample:
    if len(payload) < STATE_HEADER_STRUCT.size:
        raise ValueError(
            f"fast state payload has {len(payload)} bytes, expected at least {STATE_HEADER_STRUCT.size}."
        )
    magic, version, flags, seq, controller_time_sec, nuc_send_steady_ns = STATE_HEADER_STRUCT.unpack(
        payload[: STATE_HEADER_STRUCT.size]
    )
    if magic != STATE_MAGIC:
        raise ValueError("fast state payload has invalid magic.")
    version_i = int(version)
    if version_i == LEGACY_STATE_PROTOCOL_VERSION:
        if len(payload) != LEGACY_STATE_STRUCT.size:
            raise ValueError(
                f"legacy fast state payload has {len(payload)} bytes, expected {LEGACY_STATE_STRUCT.size}."
            )
        values = LEGACY_STATE_STRUCT.unpack(payload)[6:]
        data = np.asarray(values, dtype=np.float64).reshape(3, 7)
        O_T_EE = None
    elif version_i == STATE_PROTOCOL_VERSION:
        if len(payload) != STATE_STRUCT.size:
            raise ValueError(f"fast state payload has {len(payload)} bytes, expected {STATE_STRUCT.size}.")
        values = STATE_STRUCT.unpack(payload)[6:]
        data = np.asarray(values[:21], dtype=np.float64).reshape(3, 7)
        O_T_EE = None
        if int(flags) & STATE_HAS_O_T_EE:
            O_T_EE = np.asarray(values[21:37], dtype=np.float64).reshape(4, 4, order="F").copy()
    else:
        raise ValueError(f"unsupported fast state protocol version {version}.")
    return FastStateSample(
        seq=int(seq),
        controller_time_sec=float(controller_time_sec),
        nuc_send_steady_ns=int(nuc_send_steady_ns),
        q=data[0].copy(),
        dq=data[1].copy(),
        tau_cmd=data[2].copy(),
        received_wall_time_sec=float(time.perf_counter() if received_wall_time_sec is None else received_wall_time_sec),
        received_monotonic_ns=int(time.monotonic_ns()),
        O_T_EE=O_T_EE,
    )


def pack_action_message(
    *,
    seq: int,
    observed_state_seq: int = 0,
    target_q: Any = None,
    target_dq: Any = None,
    stiffness: Any = None,
    damping: Any = None,
    feedforward: Any = None,
    host_send_steady_ns: Optional[int] = None,
) -> bytes:
    flags = 0
    arrays = []
    for bit, name, value in (
        (ACTION_HAS_TARGET_Q, "target_q", target_q),
        (ACTION_HAS_TARGET_DQ, "target_dq", target_dq),
        (ACTION_HAS_STIFFNESS, "stiffness", stiffness),
        (ACTION_HAS_DAMPING, "damping", damping),
        (ACTION_HAS_FEEDFORWARD, "feedforward", feedforward),
    ):
        if value is None:
            arrays.append(np.zeros(7, dtype=np.float64))
        else:
            flags |= bit
            arrays.append(_as_vector7(value, name=name))
    if flags == 0:
        raise ValueError("fast action message must include at least one action array.")
    packed_arrays = np.concatenate(arrays).astype(np.float64, copy=False)
    return ACTION_STRUCT.pack(
        ACTION_MAGIC,
        ACTION_PROTOCOL_VERSION,
        int(flags),
        int(seq),
        int(observed_state_seq),
        int(time.monotonic_ns() if host_send_steady_ns is None else host_send_steady_ns),
        *[float(x) for x in packed_arrays],
    )


def overlay_fast_state_sample(sample: dict[str, Any], fast_state: FastStateSample) -> dict[str, Any]:
    """Return a history-like sample with latest q/dq/tau fields from fast state."""
    merged = dict(sample)
    merged["t"] = float(fast_state.controller_time_sec)
    merged["q"] = fast_state.q.astype(float).tolist()
    merged["dq"] = fast_state.dq.astype(float).tolist()
    merged["tau_cmd"] = fast_state.tau_cmd.astype(float).tolist()
    if fast_state.O_T_EE is not None:
        merged["O_T_EE"] = np.asarray(fast_state.O_T_EE, dtype=float).reshape(4, 4).tolist()
    merged["fast_state_seq"] = int(fast_state.seq)
    merged["fast_state_received_wall_time_sec"] = float(fast_state.received_wall_time_sec)
    merged["fast_state_receive_age_sec"] = float(fast_state.receive_age_sec)
    merged["state_transport"] = "fast"
    return merged


class FastPolicyTransport:
    """ZMQ SUB/PUSH client for the C++ fast bridge."""

    def __init__(
        self,
        *,
        state_endpoint: str = FAST_STATE_ENDPOINT,
        action_endpoint: str = FAST_ACTION_ENDPOINT,
        conflate: bool = True,
    ):
        if zmq is None:
            raise RuntimeError("pyzmq is required for FastPolicyTransport sockets.")
        self.state_endpoint = str(state_endpoint)
        self.action_endpoint = str(action_endpoint)
        self.ctx = zmq.Context.instance()

        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sub.setsockopt(zmq.RCVHWM, 1)
        self.sub.setsockopt(zmq.LINGER, 0)
        if conflate:
            self.sub.setsockopt(zmq.CONFLATE, 1)
        self.sub.connect(self.state_endpoint)

        self.push = self.ctx.socket(zmq.PUSH)
        self.push.setsockopt(zmq.SNDHWM, 1)
        self.push.setsockopt(zmq.LINGER, 0)
        self.push.connect(self.action_endpoint)

        self._poller = zmq.Poller()
        self._poller.register(self.sub, zmq.POLLIN)
        self._last_state: Optional[FastStateSample] = None
        self._last_seq = -1
        self._next_action_seq = 0
        self.malformed_state_count = 0
        self.state_drop_count = 0
        self.action_send_count = 0
        self.action_eagain_count = 0
        self.action_drop_count = 0

    @property
    def last_state(self) -> Optional[FastStateSample]:
        return self._last_state

    def reset_state_tracking(self) -> None:
        self._last_state = None
        self._last_seq = -1

    def close(self) -> None:
        self.sub.close(linger=0)
        self.push.close(linger=0)

    def poll_state(self, *, timeout_ms: int = 0, latest_only: bool = True) -> Optional[FastStateSample]:
        socks = dict(self._poller.poll(timeout=int(max(0, timeout_ms))))
        if self.sub not in socks:
            return None

        latest: Optional[FastStateSample] = None
        while True:
            try:
                payload = self.sub.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            try:
                state = unpack_state_message(payload)
            except Exception:
                self.malformed_state_count += 1
                continue
            if state.seq <= self._last_seq:
                self.state_drop_count += 1
                continue
            self._last_seq = state.seq
            self._last_state = state
            latest = state
            if not latest_only:
                break
        return latest

    def get_recent_state(self, *, max_age_s: float, poll_timeout_ms: int = 0) -> Optional[FastStateSample]:
        state = self.poll_state(timeout_ms=poll_timeout_ms, latest_only=True)
        if state is None:
            state = self._last_state
        if state is None:
            return None
        if state.receive_age_sec > float(max_age_s):
            return None
        return state

    def send_action(
        self,
        *,
        observed_state_seq: int = 0,
        target_q: Any = None,
        target_dq: Any = None,
        stiffness: Any = None,
        damping: Any = None,
        feedforward: Any = None,
        max_send_wait_s: float = 0.0,
        drop_if_busy: bool = True,
    ) -> Optional[int]:
        # The NUC bridge rejects action sequences that are not strictly
        # increasing. Use a time-derived sequence so independent Python clients
        # in the same process/host do not collide by all starting at 1.
        self._next_action_seq = max(int(self._next_action_seq) + 1, int(time.monotonic_ns()))
        payload = pack_action_message(
            seq=self._next_action_seq,
            observed_state_seq=int(observed_state_seq),
            target_q=target_q,
            target_dq=target_dq,
            stiffness=stiffness,
            damping=damping,
            feedforward=feedforward,
        )
        deadline_ns = time.monotonic_ns() + int(max(0.0, float(max_send_wait_s)) * 1.0e9)
        while True:
            try:
                self.push.send(payload, flags=zmq.NOBLOCK)
                break
            except zmq.Again:
                self.action_eagain_count += 1
                if bool(drop_if_busy):
                    self.action_drop_count += 1
                    return None
                if time.monotonic_ns() >= deadline_ns:
                    raise
                time.sleep(0.0002)
        self.action_send_count += 1
        return self._next_action_seq
