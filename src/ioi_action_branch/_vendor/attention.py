"""Wan-compatible attention using PyTorch SDPA.

Unlike upstream's optional fallback, key padding is respected. SDPA selects
available CUDA kernels automatically; FlashAttention is not a required package.
"""

import torch
import torch.nn.functional as F


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    if tuple(window_size) != (-1, -1):
        raise NotImplementedError("This ablation uses full attention, not local windows.")
    # Upstream RoPE returns float32 Q/K; use the value dtype inside SDPA.
    out_dtype = q.dtype
    q = q.transpose(1, 2).to(v.dtype)
    k = k.transpose(1, 2).to(v.dtype)
    v = v.transpose(1, 2)
    if q_scale is not None:
        q = q * q_scale
    mask = None
    if k_lens is not None:
        lengths = torch.as_tensor(k_lens, device=k.device)
        mask = torch.arange(k.size(-2), device=k.device)[None, :] < lengths[:, None]
        mask = mask[:, None, None, :]
    if causal and mask is not None:
        triangle = torch.ones(q.size(-2), k.size(-2), device=q.device, dtype=torch.bool)
        mask = mask & triangle.tril()[None, None]
    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
        dropout_p=dropout_p,
        is_causal=causal and mask is None,
        scale=softmax_scale,
    ).transpose(1, 2)
    if q_lens is not None:
        lengths = torch.as_tensor(q_lens, device=q.device)
        valid = torch.arange(out.size(1), device=q.device)[None, :] < lengths[:, None]
        out = out * valid[:, :, None, None]
    return out.contiguous().to(out_dtype)


attention = flash_attention
