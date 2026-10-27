"""Exercise forward/backward/adapter roundtrip with a tiny random Wan model."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from .config import tiny_config
from .flow import flow_matching_loss, sample_latents
from .model import create_model


def synthetic_batch(config, batch_size=1):
    b, branch = config["backbone"], config["branch"]
    return {
        "video_latent": torch.randn(batch_size, b["in_dim"], 3, 4, 4),
        "observation_latent": torch.randn(batch_size, b["in_dim"], 2, 4, 4),
        "actions": torch.randn(batch_size, 9, branch["action_dim"]),
        "text_context": torch.randn(batch_size, 5, b["text_dim"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/smoke")
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()
    if args.steps < 3:
        raise ValueError("Use at least 3 steps to exercise both zero-initialized projections.")
    torch.set_num_threads(2)
    torch.manual_seed(22)
    cfg, path = tiny_config(), Path(args.output)
    path.mkdir(parents=True, exist_ok=True)
    model = create_model(cfg, allow_random=True).train()
    model.backbone.save_pretrained(path / "tiny_backbone")
    batch = synthetic_batch(cfg)
    optimizer = torch.optim.AdamW(list(model.trainable_parameters()), lr=1e-3)
    losses = []
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = flow_matching_loss(model, batch)
        loss.backward()
        assert all(p.grad is None for p in model.backbone.parameters())
        optimizer.step()
        losses.append(loss.item())
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.action_encoder.parameters()
    ), "Action encoder received no gradients."
    checkpoint = {"config": cfg, "adapter": model.adapter_state_dict(), "format_version": 1}
    torch.save(checkpoint, path / "adapter.pt")
    restored = create_model(cfg, pretrained=path / "tiny_backbone")
    restored.load_adapter_state_dict(checkpoint["adapter"])
    model.eval()
    restored.eval()
    with torch.no_grad():
        args_forward = (
            batch["video_latent"],
            torch.tensor([500.0]),
            batch["actions"],
            batch["observation_latent"],
            batch["text_context"],
        )
        torch.testing.assert_close(model(*args_forward), restored(*args_forward))
    generated = sample_latents(
        restored,
        batch["video_latent"].shape,
        batch["actions"],
        batch["observation_latent"],
        batch["text_context"],
        steps=2,
    )
    assert torch.isfinite(generated).all()
    save_file({k: v[0].contiguous() for k, v in batch.items()}, str(path / "sample.safetensors"))
    (path / "manifest.jsonl").write_text('{"path":"sample.safetensors"}\n')
    (path / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    result = {
        "status": "passed",
        "scope": "tiny random weights / CPU, not generation quality",
        "losses": losses,
        "generated_shape": list(generated.shape),
        "parameters": model.parameter_counts(),
    }
    (path / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
