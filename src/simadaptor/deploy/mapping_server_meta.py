from __future__ import annotations

import time
from typing import Any, Optional


DEFAULT_MAPPING_CONTROL_ENDPOINT = "tcp://127.0.0.1:5560"
MAPPING_MODE_NONE = "none"
MAPPING_MODE_SIMADAPTOR = "tam"


def mapping_mode_from_backend(backend: Any) -> str:
    token = str(backend or "").strip().lower().replace("-", "_")
    if token in {"tam", "simadaptor", "sim_adaptor"}:
        return MAPPING_MODE_SIMADAPTOR
    return MAPPING_MODE_NONE


def _optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"", "none", "null"}:
            return None
        if token in {"1", "true", "yes", "y", "on", "enabled"}:
            return True
        if token in {"0", "false", "no", "n", "off", "disabled"}:
            return False
    return bool(value)


def none_mapping_server_meta(
    endpoint: str,
    *,
    source: str,
    error: Optional[str] = None,
    reason: str = "not_running",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": False,
        "server_running": False,
        "mapping_mode": MAPPING_MODE_NONE,
        "backend": None,
        "endpoint": str(endpoint),
        "source": str(source),
        "reason": str(reason),
        "time_unix_s": float(time.time()),
    }
    if error:
        payload["error"] = str(error)
    return payload


def query_mapping_server_meta(
    endpoint: str = DEFAULT_MAPPING_CONTROL_ENDPOINT,
    *,
    source: str,
    timeout_ms: int = 120,
) -> dict[str, Any]:
    endpoint = str(endpoint or "").strip()
    if not endpoint:
        return none_mapping_server_meta(endpoint, source=source, reason="endpoint_disabled")
    try:
        import zmq

        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.REQ)
        sock.SNDTIMEO = int(timeout_ms)
        sock.RCVTIMEO = int(timeout_ms)
        sock.LINGER = 0
        try:
            sock.connect(endpoint)
            sock.send_json(
                {
                    "cmd": "status",
                    "source": str(source),
                    "timestamp": time.time(),
                }
            )
            resp = sock.recv_json()
        finally:
            sock.close()
    except Exception as exc:
        return none_mapping_server_meta(endpoint, source=source, error=str(exc))

    if not isinstance(resp, dict):
        return none_mapping_server_meta(
            endpoint,
            source=source,
            error=f"non-dict status response: {type(resp).__name__}",
        )
    status = resp.get("mapping_server", resp)
    if not isinstance(status, dict):
        status = {}
    meta = dict(status)
    backend = meta.get("backend", resp.get("backend"))
    enabled = _optional_bool(meta.get("enabled", resp.get("enabled")))
    mapping_mode = str(
        meta.get("mapping_mode") or resp.get("mapping_mode") or mapping_mode_from_backend(backend)
    )
    if mapping_mode not in {MAPPING_MODE_SIMADAPTOR, MAPPING_MODE_NONE}:
        mapping_mode = mapping_mode_from_backend(backend)
    if enabled is False:
        mapping_mode = MAPPING_MODE_NONE
    meta.update(
        {
            "ok": bool(resp.get("ok", True)),
            "server_running": True,
            "mapping_mode": mapping_mode,
            "backend": backend,
            "endpoint": endpoint,
            "source": str(source),
            "time_unix_s": float(time.time()),
        }
    )
    return meta
