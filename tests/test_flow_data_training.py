import argparse
import json

import pytest
import torch
from safetensors.torch import save_file

from ioi_action_branch import train as training
from ioi_action_branch.data import LatentDataset, read_record
from ioi_action_branch.flow import sample_latents


def test_euler_direction():
    noise = torch.randn((1, 4, 3, 4, 4), generator=torch.Generator().manual_seed(22))
    target = torch.full_like(noise, 2.0)

    class Oracle(torch.nn.Module):
        def forward(self, x, timestep, actions, obs, context):
            return noise - target

    dummy = torch.zeros(1)
    actual = sample_latents(Oracle(), noise.shape, dummy, dummy, dummy, steps=5, seed=22)
    torch.testing.assert_close(actual, target, rtol=1e-5, atol=1e-6)


def test_data_validation_and_condition_only(cfg, batch, tmp_path):
    record = {k: v[0].contiguous() for k, v in batch.items()}
    save_file(record, str(tmp_path / "sample.safetensors"))
    (tmp_path / "manifest.jsonl").write_text('{"path":"sample.safetensors"}\n')
    ds = LatentDataset(tmp_path / "manifest.jsonl", cfg)
    assert ds[0]["actions"].shape == (9, 14)
    record.pop("video_latent")
    save_file(record, str(tmp_path / "condition.safetensors"))
    assert "video_latent" not in read_record(
        tmp_path / "condition.safetensors", cfg, require_target=False
    )
    with pytest.raises(ValueError, match="missing keys"):
        read_record(tmp_path / "condition.safetensors", cfg)
    record["actions"][0, 0] = float("nan")
    save_file(record, str(tmp_path / "bad.safetensors"))
    with pytest.raises(ValueError, match="finite"):
        read_record(tmp_path / "bad.safetensors", cfg, require_target=False)


def test_training_epoch_resume_matches_uninterrupted(cfg, batch, model, tmp_path, monkeypatch):
    model.backbone.save_pretrained(tmp_path / "backbone")
    save_file(
        {k: v[0].contiguous() for k, v in batch.items()}, str(tmp_path / "sample.safetensors")
    )
    # Three batches exercise a partial accumulation window.
    (tmp_path / "manifest.jsonl").write_text('{"path":"sample.safetensors"}\n' * 3)
    cfg["training"]["epochs"] = 2
    (tmp_path / "config.json").write_text(json.dumps(cfg))

    def args(output, resume=None):
        return argparse.Namespace(
            config=tmp_path / "config.json",
            backbone=tmp_path / "backbone",
            manifest=tmp_path / "manifest.jsonl",
            output=tmp_path / output,
            resume=resume,
            device="cpu",
        )

    training.train(args("full"))
    original_save = training.save_training_checkpoint

    def stop_after_first_epoch(*a, **kw):
        original_save(*a, **kw)
        if a[5] == 1:
            raise RuntimeError("simulated epoch-boundary interruption")

    monkeypatch.setattr(training, "save_training_checkpoint", stop_after_first_epoch)
    with pytest.raises(RuntimeError, match="simulated"):
        training.train(args("resumed"))
    monkeypatch.setattr(training, "save_training_checkpoint", original_save)
    training.train(args("resumed", tmp_path / "resumed/last.pt"))
    full = torch.load(tmp_path / "full/last.pt", weights_only=True)
    resumed = torch.load(tmp_path / "resumed/last.pt", weights_only=True)
    assert full["global_step"] == resumed["global_step"] == 4
    for key in full["adapter"]:
        torch.testing.assert_close(full["adapter"][key], resumed["adapter"][key], atol=0, rtol=0)
