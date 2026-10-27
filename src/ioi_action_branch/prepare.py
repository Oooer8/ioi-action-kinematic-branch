"""Cache frozen Wan VAE/T5 features from an explicit raw-data manifest."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", required=True, help="JSONL: video, observation, actions, prompt."
    )
    parser.add_argument("--backbone", required=True, help="Official Wan snapshot including VAE/T5.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    args = parser.parse_args()
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
        text_len=512,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device=device,
        checkpoint_path=str(root / "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=str(root / "google/umt5-xxl"),
    )
    entries = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    records = []
    for index, entry in enumerate(entries):

        def resolve(key):
            p = Path(entry[key])
            return p if p.is_absolute() else source.parent / p

        video, obs = load_video(resolve("video")), load_video(resolve("observation"))
        actions = torch.from_numpy(np.load(resolve("actions"), allow_pickle=False)).float()
        if (
            video.shape[1] % 4 != 1
            or obs.shape[1] % 4 != 1
            or video.shape[-2:] != obs.shape[-2:]
            or video.shape[-2] % 16
            or video.shape[-1] % 16
        ):
            raise ValueError("Use matching H/W divisible by 16 and clip/history lengths 4k+1.")
        if actions.shape != (video.shape[1], 14) or not torch.isfinite(actions).all():
            raise ValueError("Expected finite frame-aligned actions [future frame count,14].")
        with torch.no_grad():
            latent = vae.encode([video.to(device)])[0].float().cpu()
            history = vae.encode([obs.to(device)])[0].float().cpu()
            text = t5([entry.get("prompt", "")], device)[0].float().cpu()
        target = output / f"{index:07d}.safetensors"
        save_file(
            {
                "video_latent": latent.contiguous(),
                "observation_latent": history.contiguous(),
                "actions": actions.contiguous(),
                "text_context": text.contiguous(),
            },
            str(target),
        )
        records.append(json.dumps({"path": target.name}))
        print(f"{index + 1}/{len(entries)} {target.name}", flush=True)
    (output / "manifest.jsonl").write_text("\n".join(records) + "\n")


if __name__ == "__main__":
    main()
