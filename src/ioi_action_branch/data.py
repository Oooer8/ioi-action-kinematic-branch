"""Fixed-shape cached-latent dataset; no dataset-specific robotics dependency."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch.utils.data import Dataset

REQUIRED = {"video_latent", "observation_latent", "actions", "text_context"}


def read_record(path, config, require_target=True):
    tensors = load_file(str(path), device="cpu")
    required = REQUIRED if require_target else REQUIRED - {"video_latent"}
    if not required <= set(tensors):
        raise ValueError(f"{path}: missing keys {sorted(required - set(tensors))}")
    for name in required:
        value = tensors[name]
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"{path}: {name} must contain finite floating-point values.")
    obs, act, text = (tensors[k] for k in ("observation_latent", "actions", "text_context"))
    b, branch = config["backbone"], config["branch"]
    if obs.ndim != 4 or obs.shape[0] != b["in_dim"] or obs.shape[1] < 1:
        raise ValueError(f"{path}: invalid observation_latent [C,To,H,W].")
    if act.ndim != 2 or act.shape[1] != branch["action_dim"] or act.shape[0] < 1:
        raise ValueError(f"{path}: invalid actions [Ta,action_dim].")
    if text.ndim != 2 or text.shape[1] != b["text_dim"] or not 1 <= text.shape[0] <= b["text_len"]:
        raise ValueError(f"{path}: invalid text_context [L,text_dim].")
    if require_target:
        video = tensors["video_latent"]
        if (
            video.ndim != 4
            or video.shape[0] != b["in_dim"]
            or video.shape[1] < 1
            or video.shape[-2:] != obs.shape[-2:]
            or video.shape[-2] % 2
            or video.shape[-1] % 2
        ):
            raise ValueError(f"{path}: incompatible video/observation latent grids.")
        expected = 1 + (video.shape[1] - 1) * branch["action_stride"]
        if act.shape[0] != expected:
            raise ValueError(f"{path}: expected {expected} actions, got {act.shape[0]}.")
    return {k: v.float() for k, v in tensors.items() if k in REQUIRED}


class LatentDataset(Dataset):
    def __init__(self, manifest, config):
        self.config = config
        manifest = Path(manifest)
        self.paths = []
        for lineno, line in enumerate(manifest.read_text().splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record.get("path"), str):
                raise ValueError(f"{manifest}:{lineno}: expected a string 'path'.")
            path = Path(record["path"])
            self.paths.append(path if path.is_absolute() else manifest.parent / path)
        if not self.paths:
            raise ValueError(f"Empty manifest: {manifest}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return read_record(self.paths[index], self.config)


def collate_records(records):
    result = {}
    for key in REQUIRED:
        if key == "text_context":
            max_len = max(r[key].shape[0] for r in records)
            items = [
                torch.nn.functional.pad(r[key], (0, 0, 0, max_len - r[key].shape[0]))
                for r in records
            ]
        else:
            items = [r[key] for r in records]
            if len({tuple(x.shape) for x in items}) != 1:
                raise ValueError("Batch clips must have identical latent/action shapes.")
        result[key] = torch.stack(items)
    return result
