from dataclasses import dataclass
from typing import Optional


@dataclass
class EncoderConfig:
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 5
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    emb_dropout: float = 0.1
    # Training-time local attention: None = full causal; int = window size
    train_window: Optional[int] = None

    # History embedded parameters
    patch_size: int = 400
    patch_stride: int = 200

    # RoPE parameters
    # rope_base: float = 10000.0
    rope_base: float = 5000.0
    # jointwise_flat uses flattened joint-time tokens; keep this large enough for long contexts.
    rope_max_len: int = 1024
    # Max supported DoF tokens for mixed-embodiment jointwise training.
    # Joint id embeddings are allocated to this size and sliced per robot DoF.
    max_dof_tokens: int = 20

    masked_fit_max_neighbors_each_side: int = 50
    masked_fit_q_weight: float = 2.0
    masked_fit_qd_weight: float = 1.0
