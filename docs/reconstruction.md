# Reconstruction boundaries

This implements the raw-action control requested for the IOI ablation. It is a
reconstruction from the supplied manuscript and public Wan/VACE code, not the
missing original IOI release.

## Preserved structure

- Frozen Wan2.1 T2V 1.3B: 30 layers, width 1536, 12 heads, FFN 8960.
- 15 trainable blocks attached after main layers 0,2,...,28.
- c = before_proj(action_tokens) + x0 before the first branch block.
  x0 is the noisy-video patch embedding at the current diffusion evaluation.
- Branch blocks form one cascade; each after_proj emits a residual added after
  the corresponding main block. All before/after projections start at zero.
- Branch block weights are copied from selected T2V blocks after loading.
- Frozen backbone parameters still participate in forward autograd where needed
  to propagate gradients to earlier residual hints.
- Branch cross-attention reads observation history, following the IOI description.
  Main blocks retain text conditioning. Public VACE uses text in both branches.
- Official Wan state-dict names are preserved. Missing/unexpected/mismatched
  pretrained weights cause an error.

## Reconstruction choices

1. Actions are grouped in order by the VAE's temporal stride 4, with the first
   action repeated 4 times. The resulting 56D group enters a two-layer MLP of
   width 1536. This retains intermediate actions, but this exact mapping was
   not specified by the original raw-action baseline.
2. Observation tokenization uses a trainable Conv3d initialized from the main
   patch embedding and fixed 3D sinusoidal positions. Original alignment-
   embedder code was unavailable.
3. History enters branch cross-attention. Main text conditioning uses cached T5
   embeddings, with an empty prompt by default. No future truth enters conditions.
4. PyTorch SDPA replaces required FlashAttention; key padding is respected.
   Kernels may produce numerically different results.
5. Use Wan's sigma convention: x=(1-sigma)*data+sigma*noise,
   target=noise-data, timestep=1000*sigma. Euler integrates 1 down to 0.
   This is equivalent to paper tau=1-sigma with the opposite velocity sign.
6. Single-device/DDP training over cached latents, not FSDP. No latent-space
   augmentation or action CFG is silently introduced.
7. Save the final epoch; FVD validation and best-FVD selection are not implemented.

Matching the paper's historical FVD additionally requires the same data,
preprocessing, geometric branch, training, sampling, and evaluation. Dimensions
alone do not establish direct comparability to 41.23.

## Verification

CPU tests use tiny random models and cover initial backbone equivalence, action
timing/broadcast, adapter learning, frozen backbone invariance, action/observation
sensitivity, checkpointing gradient parity, strict adapter/backbone loading, padded
attention, Euler direction, dataset validation, and epoch-boundary resume.

Full-sized parameter counts use meta tensors. Actual pretrained 1.3B loading,
CUDA BF16/DDP, VAE/T5 caching/decoding, full training and quality metrics need a
target GPU environment and are not established by these CPU tests.
