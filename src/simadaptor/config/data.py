from dataclasses import dataclass
from typing import Optional


@dataclass
class DataConfig:
    num_steps: int = 4000
    xml_path: str = "assets/franka_panda/panda_pandagripper.xml"

    # Rollout / model
    history_batch: int = 128
    history_duration: float = 12.0
    num_waypoints_history: int = 12
    pause_prob: float = 0.30
    sim_timestep: float = 0.001
    dataset_base_path: str = "datasets/tam"
    save_original_split: bool = False  # if False, save only perturbed trajectories

    # End-effector payload / COM randomization.
    ee_payload_mass_delta_range: tuple[float, float] = (0.0, 1.5)
    ee_payload_com_offset_min_local_m: tuple[float, float, float] = (-0.075, -0.075, -0.075)
    ee_payload_com_offset_max_local_m: tuple[float, float, float] = (0.075, 0.075, 0.075)
    joint_model_major_ee_scale: float = 0.02
    joint_model_major_global_scale: float = 0.02

    # End-effector external force impulses (world-frame force only, no torque)
    external_force_num_impulses: int = 5
    external_force_magnitude_min_n: float = 10.0
    external_force_magnitude_max_n: float = 100.0
    external_force_duration_min_s: float = 0.08
    external_force_duration_max_s: float = 0.80
    external_force_apply_to_perturbed: bool = True
    external_force_apply_to_original: bool = False
    external_force_body_name: Optional[str] = "hand"

    # Per-robot datagen randomization profiles loaded from a JSON table.
    # Keys should match xml stem by default (e.g., "piper_description").
    datagen_profile_table_path: str = "assets/datagen_profiles.json"
    datagen_profile_key: Optional[str] = None
