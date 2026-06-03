"""Robot presets for the public TAM workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RobotPreset:
    key: str
    display_name: str
    xml_path: Path
    profile_key: str
    eval_preset: str


PUBLIC_ROBOTS: tuple[RobotPreset, ...] = (
    RobotPreset(
        key="panda_pandagripper",
        display_name="Franka Panda with Panda gripper",
        xml_path=Path("assets/franka_panda/panda_pandagripper.xml"),
        profile_key="panda_pandagripper",
        eval_preset="panda",
    ),
    RobotPreset(
        key="piper_description",
        display_name="AgileX Piper",
        xml_path=Path("assets/piper/piper_description.xml"),
        profile_key="piper_description",
        eval_preset="piper",
    ),
    RobotPreset(
        key="rby1_onearm",
        display_name="RBY-1 one-arm",
        xml_path=Path("assets/rby1a/rby1_onearm.xml"),
        profile_key="rby1_onearm",
        eval_preset="rby1_onearm",
    ),
    RobotPreset(
        key="iiwa14",
        display_name="KUKA iiwa14",
        xml_path=Path("assets/kuka_iiwa_14/iiwa14.xml"),
        profile_key="iiwa14",
        eval_preset="iiwa14",
    ),
    RobotPreset(
        key="google_robot",
        display_name="Google Robot arm",
        xml_path=Path("assets/google_robot/google_robot.xml"),
        profile_key="google_robot",
        eval_preset="google_robot",
    ),
    RobotPreset(
        key="unitree_z1",
        display_name="Unitree Z1",
        xml_path=Path("assets/unitree_z1/unitree_z1.xml"),
        profile_key="unitree_z1",
        eval_preset="unitree_z1",
    ),
    RobotPreset(
        key="flexiv_rizon4",
        display_name="Flexiv Rizon4",
        xml_path=Path("assets/flexiv_rizon4/flexiv_rizon4.xml"),
        profile_key="flexiv_rizon4",
        eval_preset="flexiv_rizon4",
    ),
)


PUBLIC_ROBOT_KEYS = tuple(robot.key for robot in PUBLIC_ROBOTS)

_ALIASES = {
    "panda": "panda_pandagripper",
    "franka": "panda_pandagripper",
    "franka_panda": "panda_pandagripper",
    "panda_pandagripper": "panda_pandagripper",
    "piper": "piper_description",
    "piper_description": "piper_description",
    "rby1": "rby1_onearm",
    "rby-1": "rby1_onearm",
    "rby1_onearm": "rby1_onearm",
    "kuka": "iiwa14",
    "kuka_iiwa14": "iiwa14",
    "iiwa": "iiwa14",
    "iiwa14": "iiwa14",
    "google": "google_robot",
    "google_robot": "google_robot",
    "unitree": "unitree_z1",
    "z1": "unitree_z1",
    "unitree_z1": "unitree_z1",
    "flexiv": "flexiv_rizon4",
    "rizon4": "flexiv_rizon4",
    "flexiv_rizon4": "flexiv_rizon4",
}

_PRESETS_BY_KEY = {robot.key: robot for robot in PUBLIC_ROBOTS}


def resolve_robot(token: str) -> RobotPreset:
    normalized = str(token).strip().lower().replace("-", "_")
    key = _ALIASES.get(normalized, normalized)
    try:
        return _PRESETS_BY_KEY[key]
    except KeyError as exc:
        choices = ", ".join(PUBLIC_ROBOT_KEYS)
        raise ValueError(f"Unknown TAM robot preset {token!r}. Choices: {choices}") from exc


def robot_choices_help() -> str:
    return ", ".join(PUBLIC_ROBOT_KEYS)
