"""Fixed-shape cached-latent dataset; no dataset-specific robotics dependency."""

import json
from pathlib import Path

import torch
from safetensors import safe_open
from torch.utils.data import Dataset

from .config import condition_key


def read_record(path, config, require_target=True):
    key = condition_key(config)
    required = {"observation_latent", "text_context", key}
    if require_target:
        required.add("video_latent")
    with safe_open(str(path), framework="pt", device="cpu") as file:
        missing = required - set(file.keys())
        if missing:
            raise ValueError(f"{path}: missing keys {sorted(missing)}")
        # Inference never even loads the future-target tensor, if one is present.
        tensors = {name: file.get_tensor(name) for name in required}
    for name in required:
        value = tensors[name]
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"{path}: {name} must contain finite floating-point values.")
    obs, cond, text = (tensors[k] for k in ("observation_latent", key, "text_context"))
    b, branch = config["backbone"], config["branch"]
    if (
        obs.ndim != 4
        or obs.shape[0] != b["in_dim"]
        or min(obs.shape[1:]) < 1
        or obs.shape[-2] % 2
        or obs.shape[-1] % 2
    ):
        raise ValueError(f"{path}: invalid observation_latent [C,To,H,W].")
    if config.get("variant", "action_branch").endswith("_adaln"):
        expected = (
            b["in_dim"],
            branch["history_latent_frames"],
            branch["latent_height"],
            branch["latent_width"],
        )
        if obs.shape != expected:
            raise ValueError(f"{path}: observation must match configured AdaLN grid {expected}.")
    if key == "geometry_latent":
        if (
            cond.ndim != 5
            or cond.shape[:2] != (3, b["in_dim"])
            or cond.shape[2] < 1
            or cond.shape[-2:] != obs.shape[-2:]
        ):
            raise ValueError(f"{path}: invalid geometry_latent [3,C,Tz,H,W].")
        future_frames = cond.shape[2]
    else:
        if cond.ndim != 2 or cond.shape[1] != branch["action_dim"] or cond.shape[0] < 1:
            raise ValueError(f"{path}: invalid actions [Ta,action_dim].")
        stride = branch["action_stride"]
        if (cond.shape[0] - 1) % stride:
            raise ValueError(f"{path}: action count must be 1 modulo action_stride.")
        future_frames = 1 + (cond.shape[0] - 1) // stride
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
        if video.shape[1] != future_frames:
            raise ValueError(f"{path}: condition and future latent time dimensions do not match.")
    return {k: tensors[k].float() for k in required}


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
    if not records or any(set(record) != set(records[0]) for record in records):
        raise ValueError("Batch records must be non-empty and share the same keys.")
    result = {}
    for key in records[0]:
        if key == "text_context":
            max_len = max(r[key].shape[0] for r in records)
            items = [
                torch.nn.functional.pad(r[key], (0, 0, 0, max_len - r[key].shape[0]))
                for r in records
            ]
        else:
            items = [r[key] for r in records]
            if len({tuple(x.shape) for x in items}) != 1:
                raise ValueError("Batch clips must have identical latent/condition shapes.")
        result[key] = torch.stack(items)
    return result
