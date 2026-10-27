# IOI Raw Action + 15 Kinematic Blocks

基于 **Wan2.1 T2V 1.3B** 和 **VACE 条件分支结构**实现的原始动作消融：

**14D actions → 时间分组 + 两层 MLP → 空间广播 → 15 个 Kinematic Blocks → 冻结 Wan 主干**

分支的 cross-attention 读取观测历史；主干保留原来的文本条件。动作进入分支的 token 输入端。

**这是依据论文与公开 VACE/Wan 代码重建的实验仓库。原始 IOI 源码缺失，未声称复现论文 FVD，也不保证与原实现逐项相同。** 当前验证范围是小模型 CPU 测试；真实 1.3B 权重、CUDA 训练及生成质量仍需在目标环境验证。见 [重建说明](docs/reconstruction.md)。

## 模型配置

| 项目 | 设置 |
|---|---|
| 官方基础模型 | Wan-AI/Wan2.1-T2V-1.3B |
| 主干 | 30 Blocks，冻结 |
| 条件分支 | 15 Blocks，从主干第 0、2、…、28 层复制初始化 |
| Hidden / heads / FFN | 1536 / 12 / 8960 |
| 输入动作 | 14 维，逐帧对齐，保留中间动作 |
| 分支观测条件 | 历史 latent → patch embedding → 固定 3D 位置编码 |
| 分支注入 | 第一层 before_proj；每层 after_proj；均零初始化 |
| 训练 | Flow matching，AdamW，BF16，梯度检查点 |
| 采样 | 50 步均匀 Euler，默认无额外 CFG |

实际参数统计：

| 部分 | 参数数 |
|---|---:|
| 冻结 Wan 主干 | 1,418,996,800 |
| 15 个 Kinematic Blocks | 696,610,560 |
| before/after projections | 37,773,312 |
| 动作编码器 | 2,448,384 |
| 观测 patch embedding | 99,840 |
| **总可训练参数** | **736,932,096** |

“1.3B”沿用官方模型名；实际 tensor 计数约 1.419B，分支约 0.737B，总计约 2.156B。论文中的近似写法不等于精确参数数。

## 安装与检查

Python 3.10+，建议 Python 3.11。正式训练需要支持 BF16 的 NVIDIA GPU。

~~~bash
python -m venv .venv
source .venv/bin/activate
# GPU 环境先按 PyTorch 官方方式安装匹配 CUDA 的 torch。
pip install -e ".[test]"
pytest -q
ioi-inspect --config configs/wan2.1_1.3b.json
ioi-smoke --output runs/smoke
~~~

参数统计使用 meta device，不分配完整模型存储。Smoke 使用缩小的随机 Wan 模型，不是生成质量测试。核心模型与训练使用 PyTorch SDPA，不强制安装 FlashAttention 或完整官方 Wan 包。

## 基础权重

~~~bash
python scripts/download_backbone.py --output checkpoints/Wan2.1-T2V-1.3B
~~~

使用官方 **T2V 1.3B 基础权重**，不接受 I2V、14B 或已训练的 VACE adapter 权重。权重和数据不包含在本仓库。

## 数据

训练 manifest 为 JSONL；相对路径基于 manifest 目录：

~~~json
{"path": "0000000.safetensors"}
{"path": "0000001.safetensors"}
~~~

每个 safetensors 文件包含以下浮点 tensors，**不含 batch 维**：

| Key | 论文仿真设置的形状 | 含义 |
|---|---|---|
| video_latent | [16,21,30,40] | 81 帧、240×320 未来视频的 Wan VAE latent |
| observation_latent | [16,2,30,40] | 5 帧观测历史单独编码的 latent |
| actions | [81,14] | 与未来帧对应的原始动作 |
| text_context | [L,4096]，1≤L≤512 | 未经过 Wan text_embedding 的 T5 输出 |

- 使用官方 WanVAE encode 输出，包含其原生缩放约定。
- 未来视频与观测历史分别编码；历史必须位于预测起点及之前。
- 无语言指令时，缓存空字符串的真实 T5 embedding，不使用零向量替代。
- 动作单位、坐标系、绝对量/增量定义由数据提供者保持一致；无 FK、IK、URDF 或自动归一化。
- 必须提供 81 个明确对齐未来帧的动作。只有 80 条帧间动作时，需先确定起点/终点约定，不能盲目补一条。
- 时间分组为 [a0,a0,a0,a0]、[a1,a2,a3,a4]、…，每组拼接后经 MLP，再广播到对应时间步的空间网格。
- 同一 batch 的 latent/action 形状一致；文本长度可以不同。
- 正式实验需保持与比较方法相同的训练/验证划分。

### 可选：从 RGB 视频缓存

只有缓存与 MP4 解码需要完整官方 Wan（VAE/T5）和视频依赖：

~~~bash
pip install -e ".[video]"
pip install "wan @ git+https://github.com/Wan-Video/Wan2.1@9737cba9c1c3c4d04b33fcad41c111989865d315"
~~~

按官方 Wan 安装说明配置 CUDA 依赖。原始 JSONL：

~~~json
{"video":"future.mp4","observation":"history.mp4","actions":"actions.npy","prompt":""}
~~~

也支持 uint8 RGB 的 .npy 视频 [T,H,W,3]。脚本不自动缩放或裁剪。

~~~bash
ioi-prepare \
  --manifest data/raw.jsonl \
  --backbone checkpoints/Wan2.1-T2V-1.3B \
  --output data/cached
~~~

## 训练

~~~bash
ioi-train \
  --config configs/wan2.1_1.3b.json \
  --backbone checkpoints/Wan2.1-T2V-1.3B \
  --manifest data/cached/manifest.jsonl \
  --output runs/action_branch
~~~

支持 DDP：

~~~bash
torchrun --standalone --nproc_per_node=8 \
  -m ioi_action_branch.train \
  --config configs/wan2.1_1.3b.json \
  --backbone checkpoints/Wan2.1-T2V-1.3B \
  --manifest data/cached/manifest.jsonl \
  --output runs/action_branch
~~~

分支约 0.737B 参数；显存还包括 AdamW 状态和激活。DDP 不分片参数，不能把多卡总显存当作单卡可用显存。已启用梯度检查点；尚未实现论文的 FSDP。

默认 50 epochs，LR 5e-5，500-step warm-up，cosine 到 1e-6，AdamW β=(0.9,0.95)，weight decay=1e-4，gradient clipping=1，seed=22。每卡 batch=1、累积=4；有效 batch=4×GPU 数，64 卡时为论文的 256。小数据调试应缩短 warm-up。

每个 epoch 保存 last.pt，包含 adapter、优化器、调度器与配置；主干不重复保存。run_info.json 记录实际参数数和有效 batch。默认保存最后一个 epoch，未实现最佳验证 FVD checkpoint 选择。

~~~bash
ioi-train \
  --config configs/wan2.1_1.3b.json \
  --backbone checkpoints/Wan2.1-T2V-1.3B \
  --manifest data/cached/manifest.jsonl \
  --output runs/action_branch \
  --resume runs/action_branch/last.pt
~~~

恢复支持 epoch 边界；需同一配置、backbone 路径、数据与 GPU 数。无 epoch 内恢复。

## 采样

条件文件只需 observation_latent、actions、text_context；不需要未来真值。输出长度由动作数推导。

~~~bash
ioi-sample \
  --backbone checkpoints/Wan2.1-T2V-1.3B \
  --adapter runs/action_branch/last.pt \
  --condition data/condition.safetensors \
  --output runs/generated.safetensors \
  --steps 50 --seed 22
~~~

可选添加 VAE 解码 MP4 参数：

~~~text
--vae-checkpoint checkpoints/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth --fps 16
~~~

## 文件

- model.py：动作编码、观测条件、15-block 分支。
- _vendor/：官方 Wan backbone + padding-aware SDPA。
- flow.py：Wan 原生 flow 方向、Euler。
- train.py / sample.py：训练、恢复、采样。
- prepare.py / data.py：可选缓存、数据契约。
- inspect.py / smoke.py：完整参数统计、小模型检查。
- tests/：数值与训练流程测试。

## 来源

[Wan2.1](https://github.com/Wan-Video/Wan2.1) 固定参考 9737cba9c1c3c4d04b33fcad41c111989865d315；
[VACE](https://github.com/ali-vilab/VACE) 固定参考 48eb44f1c4be87cc65a98bff985a26976841e9f3；
[VACE 论文](https://arxiv.org/abs/2503.07598)。

IOI 原文 PDF、审稿文本、数据和权重均未包含在仓库。
Apache-2.0；第三方归属见 [NOTICE](NOTICE) 与 [LICENSE](LICENSE)。
