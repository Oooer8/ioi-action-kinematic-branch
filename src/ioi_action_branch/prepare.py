"""Cache frozen Wan VAE/T5 features from an explicit raw-data manifest."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from .config import condition_key, load_config
from .data import read_record

VIEW_KEYS = ("geometry_front", "geometry_side", "geometry_top")


def load_video(path):
    path = Path(path)
    if path.suffix == ".npy":
        frames = np.load(path, allow_pickle=False)
    else:
        import imageio.v3 as iio

        frames = np.stack(list(iio.imiter(path)))
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise ValueError(f"{path}: expected uint8 RGB video [T,H,W,3].")
    return torch.from_numpy(frames.copy()).permute(3, 0, 1, 2).float() / 127.5 - 1


@torch.no_grad()
def encode_record(entry, source_directory, config, vae, t5, device, condition_only=False):
    def resolve(key):
        path = Path(entry[key])
        return path if path.is_absolute() else Path(source_directory) / path

    obs = load_video(resolve("observation"))
    videos = {"observation_latent": obs}
    if not condition_only:
        videos["video_latent"] = load_video(resolve("video"))
    geometry = condition_key(config) == "geometry_latent"
    if geometry:
        for key in VIEW_KEYS:
            videos[key] = load_video(resolve(key))
        future_frames = videos[VIEW_KEYS[0]].shape[1]
        if any(videos[key].shape != videos[VIEW_KEYS[0]].shape for key in VIEW_KEYS):
            raise ValueError("The three rendered views must have identical T/H/W.")
    else:
        actions = torch.from_numpy(np.load(resolve("actions"), allow_pickle=False)).float()
        if (
            actions.ndim != 2
            or actions.shape[1] != config["branch"]["action_dim"]
            or not torch.isfinite(actions).all()
        ):
            raise ValueError(
                "Expected finite frame-aligned actions [future frame count,action_dim]."
            )
        future_frames = actions.shape[0]
    if not condition_only and videos["video_latent"].shape[1] != future_frames:
        raise ValueError("Future video and condition must have the same raw frame count.")
    if future_frames < 1 or future_frames % 4 != 1:
        raise ValueError("Future frame count must be 4k+1 for the Wan VAE.")
    for video in videos.values():
        if (
            video.shape[1] < 1
            or video.shape[1] % 4 != 1
            or video.shape[-2:] != obs.shape[-2:]
            or video.shape[-2] % 16
            or video.shape[-1] % 16
        ):
            raise ValueError("Use matching H/W divisible by 16 and clip/history lengths 4k+1.")
    # Separate calls preserve the same frozen VAE's per-clip causal convention.
    encoded = {
        key: vae.encode([video.to(device)])[0].float().cpu() for key, video in videos.items()
    }
    if geometry:
        encoded["geometry_latent"] = torch.stack([encoded.pop(key) for key in VIEW_KEYS])
    else:
        encoded["actions"] = actions
    encoded["text_context"] = t5([entry.get("prompt", "")], device)[0].float().cpu()
    return {key: tensor.contiguous() for key, tensor in encoded.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        required=True,
        help="JSONL: video, observation, actions or geometry views, prompt.",
    )
    parser.add_argument("--config", default="configs/wan2.1_1.3b.json")
    parser.add_argument(
        "--condition-only", action="store_true", help="Cache without future ground truth."
    )
    parser.add_argument("--backbone", required=True, help="Official Wan snapshot including VAE/T5.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    args = parser.parse_args()
    config = load_config(args.config)
    try:
        from wan.modules.t5 import T5EncoderModel
        from wan.modules.vae import WanVAE
    except ImportError as exc:
        raise RuntimeError("Caching needs the optional official Wan package; see README.") from exc
    root, source, output = Path(args.backbone), Path(args.manifest), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    vae = WanVAE(vae_pth=str(root / "Wan2.1_VAE.pth"), device=device)
    t5 = T5EncoderModel(
        text_len=config["backbone"]["text_len"],
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device=device,
        checkpoint_path=str(root / "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=str(root / "google/umt5-xxl"),
    )
    entries = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    records = []
    for index, entry in enumerate(entries):
        record = encode_record(entry, source.parent, config, vae, t5, device, args.condition_only)
        target = output / f"{index:07d}.safetensors"
        save_file(record, str(target))
        read_record(target, config, require_target=not args.condition_only)
        records.append(json.dumps({"path": target.name}))
        print(f"{index + 1}/{len(entries)} {target.name}", flush=True)
    (output / "manifest.jsonl").write_text("\n".join(records) + "\n")


if __name__ == "__main__":
    main()
