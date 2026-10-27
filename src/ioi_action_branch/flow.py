"""Wan velocity convention: sigma=1 is noise; sigma=0 is data.

Equivalent to the paper's tau convention via tau=1-sigma and v_tau=-v_sigma.
"""

import torch
import torch.nn.functional as F


def flow_matching_loss(model, batch, generator=None):
    clean = batch["video_latent"]
    noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
    sigma = torch.rand(clean.size(0), device=clean.device, generator=generator)
    weight = sigma[:, None, None, None, None]
    noisy = (1 - weight) * clean + weight * noise
    prediction = model(
        noisy,
        sigma * 1000.0,
        batch["actions"],
        batch["observation_latent"],
        batch["text_context"],
    )
    return F.mse_loss(prediction.float(), (noise - clean).float())


@torch.no_grad()
def sample_latents(
    model,
    shape,
    actions,
    observation_latent,
    text_context,
    steps=50,
    seed=22,
):
    if steps < 1:
        raise ValueError("steps must be positive.")
    device = observation_latent.device
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(shape, device=device, dtype=observation_latent.dtype, generator=generator)
    sigmas = torch.linspace(1, 0, steps + 1, device=device)
    for current, following in zip(sigmas[:-1], sigmas[1:]):
        velocity = model(
            x, current.expand(shape[0]) * 1000.0, actions, observation_latent, text_context
        )
        x = x + (following - current) * velocity
    return x
