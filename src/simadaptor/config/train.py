from dataclasses import dataclass, field
from typing import Optional

from .data import DataConfig
from .encoder import EncoderConfig


@dataclass
class WandBConfig:
    project: str = "tam"
    tags: tuple[str, ...] = ()
    mode: str = "online"  # "online" | "offline" | "disabled"
    group: Optional[str] = None
    notes: Optional[str] = None
    artifact_interval: int = 0  # steps; <=0 disables periodic artifact upload
    artifact_type: str = "model"
    artifact_name: Optional[str] = None  # defaults to tam_ckpt_<run_name>


@dataclass
class CheckpointConfig:
    run_name: Optional[str] = None
    workdir: str = "./checkpoints/tam"
    max_to_keep: int = 0  # 0 keeps every checkpoint step in the run directory.
    resume_step: Optional[int] = None  # e.g., 10000


@dataclass
class SimEvalConfig:
    dataset_eval_enabled: bool = True
    dataset_eval_interval: int = 2000
    dataset_eval_batch_size: Optional[int] = 0  # None -> inherit history_batch; <=0 disables dataset loss eval
    dataset_eval_num_batches: int = 50
    dataset_eval_robot_key: tuple[str, ...] = ()
    dataset_eval_include_primary: bool = True
    dataset_eval_jointwise_rmse: bool = True


@dataclass
class TrainConfig:
    # Runtime / restore
    platform: Optional[str] = None  # e.g. "gpu", "cpu", "tpu"
    seed: int = 42
    num_workers: int = 2
    num_data_limit: Optional[int] = None
    restore_cfg_from_ckpt: bool = True
    override_from_cli: tuple[str, ...] = ()
    use_norm_stats: bool = False
    nan_noise_std: float = 1e-8
    robot_key: tuple[str, ...] = ()  # filter dataset dirs by robot key(s); empty means all

    # Dataset / batching
    history_batch: int = 256
    max_rollout_time_chunk: Optional[int] = None

    # TAM model
    emb_dim: int = 64
    tam_hidden: int = 64
    tam_depth: int = 3
    tam_seq_length: int = 8

    # Optimization / run loop
    lr: float = 3e-4
    max_steps: int = 20_000_000
    log_interval: int = 500
    ckpt_interval: int = 2000

    # TAM training losses / augmentation
    training_seq_length: int = 2
    traj_mix_enable: bool = True
    rollout_loss_weight: float = 1.0
    tau_map_sample_no: int = 256
    tau_recon_dt: float = 1e-3
    tau_recon_huber_delta: float = 1e-2
    random_input_delay_enable: bool = True
    dq_delay_range_ms: tuple[float, float] = (0.0, 2.0)
    torque_delay_range_ms: tuple[float, float] = (0.0, 4.0)

    # Hz randomization
    hz_randomization_enable: bool = True
    hz_randomization_choices: tuple[int, ...] = (200, 500, 1000)
    hz_randomization_base_hz: int = 1000
    hz_filter: tuple[int, ...] = ()  # optional active Hz override/filter, e.g. --hz_filter 200 500 1000

    # Nested configs
    wandb: WandBConfig = field(default_factory=WandBConfig)
    ckpt: CheckpointConfig = field(default_factory=CheckpointConfig)
    sim_eval: SimEvalConfig = field(default_factory=SimEvalConfig)
    enc: EncoderConfig = field(default_factory=EncoderConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @property
    def ablation_mode(self) -> str:
        return "tam"

    @ablation_mode.setter
    def ablation_mode(self, value: str) -> None:
        del value

    @property
    def adaptor_hidden(self) -> int:
        return self.tam_hidden

    @adaptor_hidden.setter
    def adaptor_hidden(self, value: int) -> None:
        self.tam_hidden = int(value)

    @property
    def adaptor_depth(self) -> int:
        return self.tam_depth

    @adaptor_depth.setter
    def adaptor_depth(self, value: int) -> None:
        self.tam_depth = int(value)

    @property
    def adaptor_seq_length(self) -> int:
        return self.tam_seq_length

    @adaptor_seq_length.setter
    def adaptor_seq_length(self, value: int) -> None:
        self.tam_seq_length = int(value)
