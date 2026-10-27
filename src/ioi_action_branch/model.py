"""Raw actions -> VACE-style branch -> frozen Wan2.1.

The frozen backbone is upstream Wan. The branch copies selected backbone blocks,
but its cross-attention consumes encoded observation history, following the IOI
paper rather than VACE's original text-conditioned branch.
"""

import copy
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ._vendor.wan_model import WanModel as UpstreamWanModel
from ._vendor.wan_model import sinusoidal_embedding_1d


class WanModel(UpstreamWanModel):
    """Retain all architecture fields when saving/loading custom-size test models."""

    ignore_for_config = []


class ActionEncoder(nn.Module):
    """Causally group frame-aligned actions without dropping intermediate steps.

    For stride 4: [a0,a0,a0,a0], [a1,a2,a3,a4], [a5,...,a8], ...
    A two-layer MLP maps each ordered group to one latent-time embedding.
    """

    def __init__(self, action_dim, hidden_dim, dim, stride=4):
        super().__init__()
        if stride < 1:
            raise ValueError("action_stride must be positive.")
        self.action_dim = action_dim
        self.stride = stride
        self.mlp = nn.Sequential(
            nn.Linear(action_dim * stride, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def group_actions(self, actions, latent_frames):
        if actions.ndim != 3 or actions.shape[-1] != self.action_dim:
            raise ValueError(f"actions must be [B, T, {self.action_dim}].")
        expected = 1 + (latent_frames - 1) * self.stride
        if actions.shape[1] != expected:
            raise ValueError(
                f"Expected {expected} frame-aligned actions for {latent_frames} latent "
                f"frames (stride {self.stride}); got {actions.shape[1]}. "
                "Provide the correct timestamp alignment; no silent interpolation is applied."
            )
        first = actions[:, :1].repeat(1, 1, self.stride)
        rest = actions[:, 1:].reshape(
            actions.size(0), latent_frames - 1, self.action_dim * self.stride
        )
        return torch.cat((first, rest), dim=1)

    def forward(self, actions, grid):
        frames, height, width = grid
        grouped = self.group_actions(actions, frames)
        z = self.mlp(grouped)
        # Same flatten order as Conv3d(...).flatten(2).transpose(1, 2): t, h, w.
        return (
            z[:, :, None, :]
            .expand(-1, -1, height * width, -1)
            .reshape(actions.size(0), frames * height * width, -1)
        )


def observation_positions(grid, dim, device, dtype):
    """Fixed 3D sinusoidal positions; a documented reconstruction choice."""
    if dim < 6:
        raise ValueError("Hidden width must be >= 6.")
    part = 2 * (dim // 6)
    dims = [dim - 2 * part, part, part]
    axes = torch.meshgrid(*(torch.arange(n, device=device) for n in grid), indexing="ij")
    return torch.cat(
        [sinusoidal_embedding_1d(d, axis.reshape(-1)).float() for d, axis in zip(dims, axes)],
        dim=-1,
    ).to(dtype)[None]


class FrozenWanAdapter(nn.Module):
    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def adapter_state_dict(self):
        return {
            k: v.detach().cpu()
            for k, v in self.state_dict().items()
            if not k.startswith("backbone.")
        }

    def load_adapter_state_dict(self, state):
        expected = {k for k in self.state_dict() if not k.startswith("backbone.")}
        if set(state) != expected:
            raise ValueError(
                f"Adapter keys differ: missing={sorted(expected - set(state))}, "
                f"extra={sorted(set(state) - expected)}"
            )
        self.load_state_dict(state, strict=False)

    def _run_block(self, block, x, kwargs):
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(block, x, use_reentrant=False, **kwargs)
        return block(x, **kwargs)


class ActionKinematicModel(FrozenWanAdapter):
    condition_key = "actions"

    def __init__(
        self,
        backbone,
        action_dim=14,
        action_stride=4,
        action_hidden_dim=None,
        injection_layers=None,
        context_scale=1.0,
        gradient_checkpointing=True,
    ):
        super().__init__()
        if backbone.model_type != "t2v":
            raise ValueError("Use the Wan2.1 T2V backbone, not I2V/VACE checkpoint weights.")
        if tuple(backbone.patch_size) != (1, 2, 2):
            raise ValueError("This implementation expects patch_size=(1, 2, 2).")
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.injection_layers = list(
            range(0, backbone.num_layers, 2) if injection_layers is None else injection_layers
        )
        if (
            not self.injection_layers
            or self.injection_layers[0] != 0
            or self.injection_layers != sorted(set(self.injection_layers))
            or self.injection_layers[-1] >= backbone.num_layers
        ):
            raise ValueError(
                "Injection layers must be sorted, unique, start at 0, and fit backbone."
            )
        self.layer_to_hint = {layer: i for i, layer in enumerate(self.injection_layers)}
        dim = backbone.dim
        self.action_encoder = ActionEncoder(
            action_dim, action_hidden_dim or dim, dim, action_stride
        )
        self.observation_embedding = copy.deepcopy(backbone.patch_embedding)
        self.observation_embedding.requires_grad_(True)
        self.kinematic_blocks = nn.ModuleList(
            [copy.deepcopy(backbone.blocks[layer]) for layer in self.injection_layers]
        )
        self.kinematic_blocks.requires_grad_(True)
        self.before_proj = nn.Linear(dim, dim)
        self.after_projs = nn.ModuleList([nn.Linear(dim, dim) for _ in self.injection_layers])
        for projection in [self.before_proj, *self.after_projs]:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)
        self.context_scale = context_scale
        self.gradient_checkpointing = gradient_checkpointing

    def parameter_counts(self):
        return {
            "backbone_frozen": sum(p.numel() for p in self.backbone.parameters()),
            "action_encoder": sum(p.numel() for p in self.action_encoder.parameters()),
            "observation_embedding": sum(
                p.numel() for p in self.observation_embedding.parameters()
            ),
            "kinematic_blocks": sum(p.numel() for p in self.kinematic_blocks.parameters()),
            "input_output_projections": sum(
                p.numel() for m in [self.before_proj, self.after_projs] for p in m.parameters()
            ),
            "total_trainable": sum(p.numel() for p in self.trainable_parameters()),
        }

    def forward(self, noisy_latent, timestep, actions, observation_latent, text_context):
        """All clips in a batch use the same spatial/temporal grid.

        noisy_latent: [B,16,Tz,H/8,W/8], timestep: [B] in [0,1000].
        actions: [B,1+4*(Tz-1),14], observation_latent: [B,16,To,H/8,W/8].
        text_context: cached *unprojected* Wan T5 embeddings [B,L,4096].
        """
        b = self.backbone
        if noisy_latent.ndim != 5 or observation_latent.ndim != 5:
            raise ValueError("Video and observation latents must be [B,C,T,H,W].")
        batch, channels, frames, height, width = noisy_latent.shape
        if channels != b.in_dim or height % 2 or width % 2:
            raise ValueError("Latent channels must match Wan; latent H/W must be divisible by 2.")
        if (
            observation_latent.shape[0] != batch
            or observation_latent.shape[1] != channels
            or observation_latent.shape[-2:] != (height, width)
        ):
            raise ValueError(
                "Observation batch/channels/spatial grid must match the future latent."
            )
        if timestep.shape != (batch,) or actions.shape[0] != batch:
            raise ValueError("Timestep/actions batch dimensions do not match the video.")
        if (
            text_context.ndim != 3
            or text_context.shape[0] != batch
            or text_context.shape[-1] != b.text_dim
            or not 1 <= text_context.shape[1] <= b.text_len
        ):
            raise ValueError("text_context must be [B,L,text_dim] with 1 <= L <= text_len.")
        device = noisy_latent.device
        if b.freqs.device != device:
            b.freqs = b.freqs.to(device)
        x = b.patch_embedding(noisy_latent)
        grid = tuple(x.shape[2:])
        if max(grid) > b.freqs.size(0):
            raise ValueError("Video grid exceeds upstream RoPE table.")
        x = x.flatten(2).transpose(1, 2)
        grid_sizes = torch.tensor([grid] * batch, dtype=torch.long, device=device)
        seq_lens = torch.full((batch,), math.prod(grid), dtype=torch.long, device=device)
        with torch.autocast(device_type=device.type, enabled=False):
            e = b.time_embedding(sinusoidal_embedding_1d(b.freq_dim, timestep).float())
            e0 = b.time_projection(e).unflatten(1, (6, b.dim))
        text = torch.nn.functional.pad(text_context, (0, 0, 0, b.text_len - text_context.shape[1]))
        text = b.text_embedding(text)
        common = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=b.freqs,
            context_lens=None,
        )
        obs = self.observation_embedding(observation_latent)
        obs_grid = tuple(obs.shape[2:])
        obs = obs.flatten(2).transpose(1, 2)
        obs = obs + observation_positions(obs_grid, b.dim, device, obs.dtype)
        c = self.before_proj(self.action_encoder(actions, grid)) + x
        hints = []
        for block, projection in zip(self.kinematic_blocks, self.after_projs):
            c = self._run_block(block, c, dict(common, context=obs))
            hints.append(projection(c))
        for i, block in enumerate(b.blocks):
            # Frozen parameters still participate in autograd: later layers carry
            # the loss gradient back to earlier injected hints.
            x = self._run_block(block, x, dict(common, context=text))
            if i in self.layer_to_hint:
                x = x + self.context_scale * hints[self.layer_to_hint[i]]
        x = b.head(x, e)
        return torch.stack(b.unpatchify(x, grid_sizes))


def create_model(config, pretrained=None, allow_random=False):
    if pretrained:
        stored_config = json.loads((Path(pretrained) / "config.json").read_text())
        for name, expected in config["backbone"].items():
            if name in stored_config and stored_config[name] != expected:
                raise ValueError(
                    f"Backbone config mismatch: {name}={stored_config[name]}, expected {expected}."
                )
        # Upstream intentionally omits these fields from serialized configs.
        overrides = {k: v for k, v in config["backbone"].items() if k not in stored_config}
        backbone, loading = WanModel.from_pretrained(
            str(pretrained),
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False,
            local_files_only=True,
            output_loading_info=True,
            **overrides,
        )
        if any(
            loading.get(k)
            for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        ):
            raise ValueError(f"Backbone checkpoint did not load exactly: {loading}")
        for name, expected in config["backbone"].items():
            actual = getattr(backbone, name)
            if isinstance(expected, list):
                actual = list(actual)
            if actual != expected:
                raise ValueError(f"Backbone config mismatch: {name}={actual}, expected {expected}.")
    elif allow_random:
        backbone = WanModel(**config["backbone"])
        # Upstream initializes its output head to zero. A nonzero dummy head is
        # necessary to exercise adapter gradients in random-weight smoke tests.
        nn.init.normal_(backbone.head.head.weight, std=0.02)
    else:
        raise ValueError(
            "A local Wan2.1 T2V checkpoint is required; random weights are smoke-only."
        )
    variant = config.get("variant", "action_branch")
    if variant == "action_branch":
        return ActionKinematicModel(backbone, **config["branch"])
    if variant in ("geometry_adaln", "action_adaln"):
        from .adaln import AdaLNModel

        conditioning = "geometry" if variant == "geometry_adaln" else "action"
        return AdaLNModel(backbone, conditioning=conditioning, **config["branch"])
    raise ValueError(f"Unknown model variant: {variant}")
