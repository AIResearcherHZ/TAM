"""Public TAM workflow surface.

TAM stands for Torque Adaptation Module.
"""

from __future__ import annotations

from .presets import PUBLIC_ROBOT_KEYS, RobotPreset, resolve_robot

__all__ = ["PUBLIC_ROBOT_KEYS", "RobotPreset", "resolve_robot"]
