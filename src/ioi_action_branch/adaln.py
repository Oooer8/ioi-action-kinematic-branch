"""Geometry/action AdaLN controls on the same frozen Wan backbone.

The paper specifies view embeddings, a 48->32->16 fusion MLP, and AdaLN
scale/shift prediction. Spatial readout, history conditioning, and precise
modulation sites here are explicit reconstruction choices.
"""

import math

import torch
from torch import nn

from ._vendor.wan_model import sinusoidal_embedding_1d
from .model import ActionEncoder, FrozenWanAdapter


class ThreeViewEncoder(nn.Module):
    """Fixed view order: front, side, top; frozen-VAE latents are input."""

    def __init__(self, channels, height, width, dim, fusion_hidden_dim=None):
        super().__init__()
        self.channels, self.height, self.width = channels, height, width
        self.view_embeddings = nn.Parameter(torch.randn(3, channels) * 0.02)
        self.fusion = nn.Sequential(
            nn.Linear(3 * channels, fusion_hidden_dim or 2 * channels),
            nn.SiLU(),
            nn.Linear(fusion_hidden_dim or 2 * channels, channels),
        )
        # Preserve spatial order rather than global-average pooling.
        self.readout = nn.Sequential(nn.Linear(height * width * channels, dim), nn.SiLU())

    def fuse(self, geometry_latent):
        if geometry_latent.ndim != 6:
            raise ValueError("geometry_latent must be [B,3,C,Tz,Hlatent,Wlatent].")
        if (
            geometry_latent.shape[1] != 3
            or geometry_latent.shape[2] != self.channels
            or geometry_latent.shape[-2:] != (self.height, self.width)
            or geometry_latent.shape[3] < 1
        ):
            raise ValueError(
                "Geometry must contain exactly three views on the configured latent grid."
            )
        z = geometry_latent.permute(0, 3, 4, 5, 1, 2)  # B,T,H,W,V,C
        z = z + self.view_embeddings[None, None, None, None]
        return self.fusion(z.flatten(-2))  # B,T,H,W,C

    def forward(self, geometry_latent):
        return self.readout(self.fuse(geometry_latent).flatten(2))  # B,T,D


def _normalize_with_delta(norm, x, shift, scale, delta_shift, delta_scale):
    batch, length, dim = x.shape
    times = delta_shift.shape[1]
    z = norm(x).float().reshape(batch, times, length // times, dim)
    scale = scale[:, None] + delta_scale[:, :, None].float()
    shift = shift[:, None] + delta_shift[:, :, None].float()
    return (z * (1 + scale) + shift).reshape(batch, length, dim)


def adaln_block_forward(
    x,
    *,
    block,
    delta,
    e,
    seq_lens,
    grid_sizes,
    freqs,
    context,
    context_lens,
):
    """Wan block with additive scale/shift deltas at its two existing AdaLNs.

    The native diffusion timestep scales/shifts and both residual gates remain.
    No new cross-attention layer or Transformer block is introduced.
    """
    with torch.autocast(device_type=x.device.type, enabled=False):
        native = (block.modulation.float() + e.float()).chunk(6, dim=1)
    shift_sa, scale_sa, shift_ffn, scale_ffn = delta.unbind(dim=2)
    y = _normalize_with_delta(block.norm1, x, native[0], native[1], shift_sa, scale_sa)
    y = block.self_attn(y, seq_lens, grid_sizes, freqs)
    x = x + y * native[2]
    x = x + block.cross_attn(block.norm3(x), context, context_lens)
    y = _normalize_with_delta(block.norm2, x, native[3], native[4], shift_ffn, scale_ffn)
    x = x + block.ffn(y) * native[5]
    return x


class AdaLNModel(FrozenWanAdapter):
    def __init__(
        self,
        backbone,
        conditioning="geometry",
        latent_height=30,
        latent_width=40,
        history_latent_frames=2,
        fusion_hidden_dim=None,
        action_dim=14,
        action_stride=4,
        action_hidden_dim=None,
        injection_layers=None,
        context_scale=1.0,
        gradient_checkpointing=True,
    ):
        super().__init__()
        if conditioning not in ("geometry", "action"):
            raise ValueError("conditioning must be 'geometry' or 'action'.")
        if backbone.model_type != "t2v" or tuple(backbone.patch_size) != (1, 2, 2):
            raise ValueError("Use the Wan2.1 T2V backbone with patch_size=(1,2,2).")
        if min(latent_height, latent_width, history_latent_frames) < 1:
            raise ValueError("Latent dimensions must be positive.")
        if latent_height % 2 or latent_width % 2:
            raise ValueError("Latent height and width must be divisible by 2.")
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.conditioning = conditioning
        self.condition_key = "geometry_latent" if conditioning == "geometry" else "actions"
        self.latent_height, self.latent_width = latent_height, latent_width
        self.history_latent_frames = history_latent_frames
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
        self.layer_to_head = {layer: i for i, layer in enumerate(self.injection_layers)}
        dim = backbone.dim
        if conditioning == "geometry":
            self.geometry_encoder = ThreeViewEncoder(
                backbone.in_dim, latent_height, latent_width, dim, fusion_hidden_dim
            )
        else:
            self.action_encoder = ActionEncoder(
                action_dim, action_hidden_dim or dim, dim, action_stride
            )
        # Preserve both spatial and temporal ordering of the observed history.
        # This exact readout is shared by geometry-AdaLN and action-AdaLN.
        self.history_encoder = nn.Sequential(
            nn.Linear(backbone.in_dim * history_latent_frames * latent_height * latent_width, dim),
            nn.SiLU(),
        )
        self.adaln_heads = nn.ModuleList(
            [nn.Sequential(nn.SiLU(), nn.Linear(dim, 4 * dim)) for _ in self.injection_layers]
        )
        for head in self.adaln_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        self.context_scale = context_scale
        self.gradient_checkpointing = gradient_checkpointing

    def parameter_counts(self):
        encoder = self.geometry_encoder if self.conditioning == "geometry" else self.action_encoder
        result = {
            "backbone_frozen": sum(p.numel() for p in self.backbone.parameters()),
            "condition_encoder": sum(p.numel() for p in encoder.parameters()),
            "history_encoder": sum(p.numel() for p in self.history_encoder.parameters()),
            "adaln_heads": sum(p.numel() for p in self.adaln_heads.parameters()),
            "kinematic_blocks": 0,
            "total_trainable": sum(p.numel() for p in self.trainable_parameters()),
        }
        return result

    def forward(self, noisy_latent, timestep, condition, observation_latent, text_context):
        b = self.backbone
        if noisy_latent.ndim != 5:
            raise ValueError("noisy_latent must be [B,C,Tz,Hlatent,Wlatent].")
        batch, channels, times, height, width = noisy_latent.shape
        if (
            channels != b.in_dim
            or times < 1
            or (height, width) != (self.latent_height, self.latent_width)
        ):
            raise ValueError("Future latent does not match the configured spatial readout.")
        if observation_latent.shape != (batch, channels, self.history_latent_frames, height, width):
            raise ValueError("Observation latent does not match the configured history/grid.")
        if timestep.shape != (batch,) or condition.shape[0] != batch:
            raise ValueError("Timestep/condition batch size does not match.")
        if (
            text_context.ndim != 3
            or text_context.shape[0] != batch
            or text_context.shape[-1] != b.text_dim
            or not 1 <= text_context.shape[1] <= b.text_len
        ):
            raise ValueError("text_context must be [B,L,text_dim], 1 <= L <= text_len.")
        if self.conditioning == "geometry":
            if condition.ndim != 6 or condition.shape[3] != times:
                raise ValueError("Geometry time dimension must match future latent time.")
            c = self.geometry_encoder(condition)
        else:
            c = self.action_encoder.mlp(self.action_encoder.group_actions(condition, times))
        history = observation_latent.permute(0, 2, 3, 4, 1).flatten(1)
        c = c + self.history_encoder(history)[:, None]
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
        text = b.text_embedding(
            torch.nn.functional.pad(text_context, (0, 0, 0, b.text_len - text_context.shape[1]))
        )
        common = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=b.freqs,
            context=text,
            context_lens=None,
        )
        for i, block in enumerate(b.blocks):
            if i in self.layer_to_head:
                delta = self.adaln_heads[self.layer_to_head[i]](c)
                delta = delta.reshape(batch, times, 4, b.dim) * self.context_scale
                x = self._run_block(adaln_block_forward, x, dict(common, block=block, delta=delta))
            else:
                x = self._run_block(block, x, common)
        return torch.stack(b.unpatchify(b.head(x, e), grid_sizes))
