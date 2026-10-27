"""Small JSON configs; no framework-specific training configuration."""

import json
from pathlib import Path

WAN_1_3B = dict(
    model_type="t2v",
    patch_size=[1, 2, 2],
    text_len=512,
    in_dim=16,
    dim=1536,
    ffn_dim=8960,
    freq_dim=256,
    text_dim=4096,
    out_dim=16,
    num_heads=12,
    num_layers=30,
    window_size=[-1, -1],
    qk_norm=True,
    cross_attn_norm=True,
    eps=1e-6,
)

TINY = dict(
    model_type="t2v",
    patch_size=[1, 2, 2],
    text_len=8,
    in_dim=4,
    dim=48,
    ffn_dim=96,
    freq_dim=16,
    text_dim=24,
    out_dim=4,
    num_heads=4,
    num_layers=4,
    window_size=[-1, -1],
    qk_norm=True,
    cross_attn_norm=True,
    eps=1e-6,
)


def load_config(path):
    with Path(path).open() as f:
        cfg = json.load(f)
    for section in ("backbone", "branch", "training"):
        if section not in cfg:
            raise ValueError(f"Missing config section: {section}")
    return cfg


def tiny_config():
    return {
        "backbone": dict(TINY),
        "branch": {
            "action_dim": 14,
            "action_stride": 4,
            "action_hidden_dim": 48,
            "injection_layers": [0, 2],
            "context_scale": 1.0,
            "gradient_checkpointing": False,
        },
        "training": {
            "epochs": 2,
            "batch_size": 1,
            "gradient_accumulation": 2,
            "learning_rate": 0.001,
            "min_learning_rate": 0.0001,
            "warmup_steps": 0,
            "weight_decay": 0.0001,
            "betas": [0.9, 0.95],
            "max_grad_norm": 1.0,
            "precision": "float32",
            "seed": 22,
            "num_workers": 0,
            "log_every": 1,
        },
    }
