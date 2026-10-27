import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from ioi_action_branch.adaln import ThreeViewEncoder
from ioi_action_branch.config import WAN_1_3B, tiny_config
from ioi_action_branch.data import collate_records, read_record
from ioi_action_branch.flow import flow_matching_loss
from ioi_action_branch.model import create_model
from ioi_action_branch.prepare import VIEW_KEYS, encode_record
from ioi_action_branch.smoke import synthetic_batch


def setup(variant="geometry_adaln", batch_size=1):
    torch.set_num_threads(2)
    torch.manual_seed(22)
    cfg = tiny_config(variant)
    return cfg, create_model(cfg, allow_random=True), synthetic_batch(cfg, batch_size)


def args_for(model, batch):
    return [
        batch["video_latent"],
        torch.full((batch["video_latent"].shape[0],), 500.0),
        batch[model.condition_key],
        batch["observation_latent"],
        batch["text_context"],
    ]


def open_heads(model):
    with torch.no_grad():
        for head in model.adaln_heads:
            torch.nn.init.normal_(head[-1].weight, std=0.02)


@pytest.mark.parametrize("variant", ["geometry_adaln", "action_adaln"])
def test_adaln_zero_init_and_training(variant):
    _, model, batch = setup(variant, batch_size=2)
    model.eval()
    args = args_for(model, batch)
    with torch.no_grad():
        baseline = torch.stack(
            model.backbone(
                list(args[0]),
                args[1],
                list(args[4]),
                seq_len=12,
            )
        )
        torch.testing.assert_close(model(*args), baseline, rtol=0, atol=0)
    assert not hasattr(model, "kinematic_blocks")
    frozen = {name: p.detach().clone() for name, p in model.backbone.named_parameters()}
    model.train()
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-3)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = flow_matching_loss(model, batch)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and p.grad.abs().sum() > 0, name
    assert all(p.grad is None for p in model.backbone.parameters())
    for name, p in model.backbone.named_parameters():
        torch.testing.assert_close(p, frozen[name], rtol=0, atol=0)
    model.eval()
    with torch.no_grad():
        original = model(*args)
        changed = list(args)
        changed[3] = args[3] + 3
        assert (model(*changed) - original).abs().max() > 1e-7
        if variant == "geometry_adaln":
            for view in range(3):
                changed = list(args)
                changed[2] = args[2].clone()
                changed[2][:, view] += 3
                assert (model(*changed) - original).abs().max() > 1e-7
        else:
            changed = list(args)
            changed[2] = args[2] + 3
            assert (model(*changed) - original).abs().max() > 1e-7


def test_geometry_encoder_retains_view_spatial_and_temporal_order():
    torch.manual_seed(22)
    encoder = ThreeViewEncoder(4, 4, 4, 48)
    geometry = torch.randn(2, 3, 4, 3, 4, 4, requires_grad=True)
    output = encoder(geometry)
    assert output.shape == (2, 3, 48)
    for view in range(3):
        changed = geometry.detach().clone()
        changed[:, view, :, 1] += 1
        result = encoder(changed)
        torch.testing.assert_close(result[:, [0, 2]], output[:, [0, 2]], rtol=0, atol=0)
        assert not torch.allclose(result[:, 1], output[:, 1])
    assert not torch.allclose(encoder(geometry.flip(-1)), output)
    assert not torch.allclose(encoder(geometry.flip(1)), output)
    output.square().mean().backward()
    assert all(geometry.grad[:, view].abs().sum() > 0 for view in range(3))
    with pytest.raises(ValueError, match="exactly three"):
        encoder(geometry[:, :2])


@pytest.mark.parametrize("variant", ["geometry_adaln", "action_adaln"])
def test_adaln_checkpointing_and_adapter_roundtrip(variant, tmp_path):
    cfg, model, batch = setup(variant)
    open_heads(model)
    other = copy.deepcopy(model)
    model.gradient_checkpointing = False
    other.gradient_checkpointing = True
    model.train()
    other.train()
    out1, out2 = model(*args_for(model, batch)), other(*args_for(other, batch))
    torch.testing.assert_close(out1, out2)
    out1.square().mean().backward()
    out2.square().mean().backward()
    for (name, p), (other_name, q) in zip(model.named_parameters(), other.named_parameters()):
        assert name == other_name
        if p.requires_grad:
            assert p.grad is not None, name
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-4, atol=1e-6)
    model.backbone.save_pretrained(tmp_path / "backbone")
    restored = create_model(cfg, pretrained=tmp_path / "backbone")
    restored.load_adapter_state_dict(model.adapter_state_dict())
    model.eval()
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(
            model(*args_for(model, batch)), restored(*args_for(restored, batch))
        )


def test_geometry_record_shapes_and_target_free_sampling_input(tmp_path):
    cfg, model, batch = setup()
    record = {key: value[0].contiguous() for key, value in batch.items()}
    path = tmp_path / "sample.safetensors"
    save_file(record, str(path))
    loaded = read_record(path, cfg)
    assert collate_records([loaded, loaded])["geometry_latent"].shape == (2, 3, 4, 3, 4, 4)
    record["video_latent"].fill_(float("nan"))
    save_file(record, str(path))
    assert "video_latent" not in read_record(path, cfg, require_target=False)
    record.pop("video_latent")
    save_file(record, str(path))
    read_record(path, cfg, require_target=False)
    with pytest.raises(ValueError, match="missing keys"):
        read_record(path, cfg)
    record["geometry_latent"] = record["geometry_latent"][:2].contiguous()
    save_file(record, str(path))
    with pytest.raises(ValueError, match="invalid geometry"):
        read_record(path, cfg, require_target=False)
    args = args_for(model, batch)
    args[2] = args[2][:, :, :, :-1]
    with pytest.raises(ValueError, match="time dimension"):
        model(*args)
    args = args_for(model, batch)
    args[3] = args[3][:, :, :1]
    with pytest.raises(ValueError, match="Observation latent"):
        model(*args)


def test_cache_encodes_each_view_independently_and_never_uses_target_as_condition(tmp_path):
    cfg = tiny_config("geometry_adaln")
    entry = {"observation": "obs.npy", "video": "future.npy", "prompt": ""}
    for key in VIEW_KEYS:
        entry[key] = key + ".npy"
    for i, key in enumerate(["observation", "video", *VIEW_KEYS]):
        frames = 5 if key == "observation" else 9
        np.save(tmp_path / entry[key], np.full((frames, 32, 32, 3), i * 20, np.uint8))

    class FakeVAE:
        def __init__(self):
            self.calls = []

        def encode(self, videos):
            assert len(videos) == 1 and not torch.is_grad_enabled()
            video = videos[0]
            self.calls.append(video.mean().item())
            return [video.mean().expand(4, 1 + (video.shape[1] - 1) // 4, 4, 4).clone()]

    def fake_t5(prompts, device):
        assert prompts == [""] and not torch.is_grad_enabled()
        return [torch.ones(5, 24)]

    vae = FakeVAE()
    record = encode_record(entry, tmp_path, cfg, vae, fake_t5, torch.device("cpu"))
    assert len(vae.calls) == 5
    assert len(set(vae.calls)) == 5
    assert record["geometry_latent"].shape == (3, 4, 3, 4, 4)
    for i in range(3):
        torch.testing.assert_close(
            record["geometry_latent"][i], torch.full((4, 3, 4, 4), vae.calls[2 + i])
        )
    save_file(record, str(tmp_path / "cache.safetensors"))
    read_record(tmp_path / "cache.safetensors", cfg)
    entry.pop("video")
    conditions = encode_record(
        entry, tmp_path, cfg, FakeVAE(), fake_t5, torch.device("cpu"), condition_only=True
    )
    assert "video_latent" not in conditions
    torch.testing.assert_close(conditions["geometry_latent"], record["geometry_latent"])


def test_full_adaln_configs_share_backbone_history_and_injection_sites():
    root = Path(__file__).resolve().parents[1]
    expected = json.loads((root / "docs/adaln_parameter_counts.json").read_text())
    models = []
    for variant in ["geometry_adaln", "action_adaln"]:
        cfg = json.loads((root / f"configs/{variant}_wan2.1_1.3b.json").read_text())
        assert cfg["backbone"] == WAN_1_3B
        with torch.device("meta"):
            model = create_model(cfg, allow_random=True)
        models.append(model)
        assert model.parameter_counts() == expected[variant]
        assert model.injection_layers == list(range(0, 30, 2))
        assert model.parameter_counts()["kinematic_blocks"] == 0
        assert model.parameter_counts()["backbone_frozen"] == 1418996800
    for attribute in ["history_encoder", "adaln_heads"]:
        shapes = [
            {name: tuple(p.shape) for name, p in getattr(m, attribute).named_parameters()}
            for m in models
        ]
        assert shapes[0] == shapes[1]
    assert models[0].geometry_encoder.fusion[0].weight.shape == (32, 48)
    assert models[0].geometry_encoder.fusion[2].weight.shape == (16, 32)
