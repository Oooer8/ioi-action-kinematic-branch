import copy
import json
import math
from pathlib import Path

import pytest
import torch

from ioi_action_branch._vendor.attention import flash_attention
from ioi_action_branch.config import WAN_1_3B
from ioi_action_branch.flow import flow_matching_loss
from ioi_action_branch.model import ActionEncoder, create_model


def forward_args(batch):
    return (
        batch["video_latent"],
        torch.tensor([500.0]),
        batch["actions"],
        batch["observation_latent"],
        batch["text_context"],
    )


def open_gates(model):
    with torch.no_grad():
        torch.nn.init.normal_(model.before_proj.weight, std=0.02)
        for p in model.after_projs:
            torch.nn.init.normal_(p.weight, std=0.02)


def test_initial_model_matches_frozen_wan(model, batch):
    model.eval()
    args = forward_args(batch)
    with torch.no_grad():
        actual = model(*args)
        _, _, frames, height, width = batch["video_latent"].shape
        baseline = torch.stack(
            model.backbone(
                list(batch["video_latent"]),
                args[1],
                list(batch["text_context"]),
                seq_len=frames * (height // 2) * (width // 2),
            )
        )
    torch.testing.assert_close(actual, baseline, rtol=0, atol=0)


def test_ordered_action_groups_and_broadcast():
    encoder = ActionEncoder(2, 8, 12, stride=4)
    actions = torch.arange(18).float().view(1, 9, 2)
    grouped = encoder.group_actions(actions, 3)
    torch.testing.assert_close(grouped[0, 0], actions[0, 0].repeat(4))
    torch.testing.assert_close(grouped[0, 1:].flatten(), actions[0, 1:].flatten())
    tokens = encoder(actions, (3, 2, 3)).reshape(1, 3, 6, 12)
    torch.testing.assert_close(tokens[:, :, 0], tokens[:, :, 5])
    assert not torch.allclose(tokens[:, 0], tokens[:, 1])
    assert encoder(actions[:, :1], (1, 2, 3)).shape == (1, 6, 12)
    with pytest.raises(ValueError, match="Expected 9"):
        encoder(actions[:, :-1], (3, 2, 3))


def test_only_branch_trains_and_actions_observations_affect_output(model, batch):
    model.train()
    frozen = {n: p.detach().clone() for n, p in model.backbone.named_parameters()}
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=0.001)
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        loss = flow_matching_loss(model, batch)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    assert all(p.grad is None for p in model.backbone.parameters())
    assert sum(p.grad.abs().sum() for p in model.action_encoder.parameters()) > 0
    for name, p in model.backbone.named_parameters():
        torch.testing.assert_close(p, frozen[name], rtol=0, atol=0)
    model.eval()
    args = list(forward_args(batch))
    with torch.no_grad():
        original = model(*args)
        args[2] = args[2] + 3
        changed_action = model(*args)
        args = list(forward_args(batch))
        args[3] = args[3] + 3
        changed_observation = model(*args)
    assert (original - changed_action).abs().max() > 1e-8
    assert (original - changed_observation).abs().max() > 1e-8


def test_checkpointing_preserves_output_and_gradients(model, batch):
    open_gates(model)
    other = copy.deepcopy(model)
    model.gradient_checkpointing = False
    other.gradient_checkpointing = True
    model.train()
    other.train()
    out1, out2 = model(*forward_args(batch)), other(*forward_args(batch))
    torch.testing.assert_close(out1, out2)
    out1.square().mean().backward()
    out2.square().mean().backward()
    for (name, p), (other_name, q) in zip(model.named_parameters(), other.named_parameters()):
        assert name == other_name
        if p.requires_grad:
            assert p.grad is not None, name
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-4, atol=1e-6)


def test_adapter_checkpoint_roundtrip(model, batch, cfg, tmp_path):
    open_gates(model)
    model.backbone.save_pretrained(tmp_path / "backbone")
    restored = create_model(cfg, pretrained=tmp_path / "backbone")
    restored.load_adapter_state_dict(model.adapter_state_dict())
    assert not any(key.startswith("backbone.") for key in model.adapter_state_dict())
    model.eval()
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(*forward_args(batch)), restored(*forward_args(batch)))
    with pytest.raises(ValueError, match="Adapter keys differ"):
        restored.load_adapter_state_dict({})
    bad_cfg = copy.deepcopy(cfg)
    bad_cfg["backbone"]["num_layers"] += 2
    with pytest.raises(ValueError, match="Backbone config mismatch"):
        create_model(bad_cfg, pretrained=tmp_path / "backbone")


def test_sdpa_honors_padding():
    torch.manual_seed(1)
    q, k, v = [torch.randn(2, 5, 2, 8) for _ in range(3)]
    output = flash_attention(q, k, v, k_lens=torch.tensor([3, 5]))
    altered = v.clone()
    altered[0, 3:] = 10000
    torch.testing.assert_close(output, flash_attention(q, k, altered, k_lens=torch.tensor([3, 5])))
    scores = torch.einsum("bqhd,bkhd->bhqk", q[:1], k[:1, :3]) / math.sqrt(8)
    reference = torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), v[:1, :3])
    torch.testing.assert_close(output[:1], reference, rtol=1e-5, atol=1e-6)


def test_full_size_config_and_parameter_budget():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/wan2.1_1.3b.json").read_text())
    assert config["backbone"] == WAN_1_3B
    assert config["branch"]["injection_layers"] == list(range(0, 30, 2))
    with torch.device("meta"):
        full_model = create_model(config, allow_random=True)
    counts = full_model.parameter_counts()
    expected = json.loads((root / "docs/parameter_counts.json").read_text())
    for key, value in counts.items():
        assert expected[key] == value
    assert len(full_model.kinematic_blocks) == 15
