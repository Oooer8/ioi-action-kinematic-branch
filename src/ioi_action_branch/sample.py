"""Generate future latents using observed history and actions only."""

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file

from .data import read_record
from .flow import sample_latents
from .model import create_model
from .train import precision_context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--adapter", required=True, help="Training last.pt checkpoint.")
    parser.add_argument(
        "--condition", required=True, help="safetensors with obs/actions/T5 context."
    )
    parser.add_argument("--output", required=True, help="Output latent .safetensors file.")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--precision", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--vae-checkpoint", help="Optional Wan2.1_VAE.pth for MP4 decoding.")
    parser.add_argument("--fps", type=int, default=16)
    args = parser.parse_args()
    state = torch.load(args.adapter, map_location="cpu", weights_only=True)
    cfg = state["config"]
    device = torch.device(args.device)
    model = create_model(cfg, pretrained=args.backbone).to(device)
    model.load_adapter_state_dict(state["adapter"])
    model.eval()
    record = read_record(args.condition, cfg, require_target=False)
    # Future ground truth, even if present in an evaluation record, is never read
    # by the sampler or used to obtain its output shape.
    act = record["actions"].unsqueeze(0).to(device)
    obs = record["observation_latent"].unsqueeze(0).to(device)
    text = record["text_context"].unsqueeze(0).to(device)
    stride = cfg["branch"]["action_stride"]
    if (act.shape[1] - 1) % stride:
        raise ValueError("Action count must be 1 modulo action_stride.")
    frames = 1 + (act.shape[1] - 1) // stride
    shape = (1, cfg["backbone"]["out_dim"], frames, *obs.shape[-2:])
    with precision_context(device, args.precision):
        latent = sample_latents(model, shape, act, obs, text, args.steps, args.seed)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"video_latent": latent[0].float().cpu().contiguous()}, str(path))
    print(f"Saved generated latent: {path}")
    if args.vae_checkpoint:
        try:
            import imageio.v3 as iio
            from wan.modules.vae import WanVAE
        except ImportError as exc:
            raise RuntimeError(
                "MP4 decoding needs official Wan and the [video] extra; see README."
            ) from exc
        # Release the diffusion model before loading the VAE.
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        vae = WanVAE(vae_pth=args.vae_checkpoint, device=device)
        video = vae.decode([latent[0]])[0].clamp(-1, 1)
        frames_rgb = ((video.permute(1, 2, 3, 0) + 1) * 127.5).round().byte().cpu().numpy()
        iio.imwrite(path.with_suffix(".mp4"), frames_rgb, fps=args.fps)


if __name__ == "__main__":
    main()
