# 三视角几何 + AdaLN

新增配置：`configs/geometry_adaln_wan2.1_1.3b.json`。继续使用冻结的
**Wan2.1 T2V 1.3B**，训练三视角编码、观测历史编码和 AdaLN 预测头。
没有额外的 Kinematic Blocks，也没有新增 cross-attention。

## 结构

```text
front / side / top 的正交几何渲染视频
    ↓ 同一个冻结 Wan VAE 分别编码
[B,3,16,Tz,30,40]
    ↓ 各加一个可学习的 16D view embedding
    ↓ 在相同 t,h,w 上拼接三个视角的通道
[B,Tz,30,40,48]
    ↓ 逐位置融合 MLP：48 → 32 → 16，SiLU
[B,Tz,30,40,16]
    ↓ 每个时间步按 H,W,C 顺序展开，Linear(19200,1536) + SiLU
[B,Tz,1536]
    + 历史 latent 按 T,H,W,C 展开后的 Linear(38400,1536) + SiLU
    ↓ 15 个独立的 SiLU + Linear(1536,4×1536) 预测头
每层的 δshift_self、δscale_self、δshift_ffn、δscale_ffn
    ↓ 在冻结主干第 0,2,…,28 层，按时间对齐并广播到空间 tokens
归一化后的特征调制 → Wan 原生 attention / FFN → 预测速度
```

每个注入点使用：

```text
y = norm(x) * (1 + scale_timestep + δscale_condition)
              + shift_timestep + δshift_condition
```

self-attention 和 FFN 分别使用自己的 scale/shift。保留 Wan 的 timestep
embedding、原生 residual gates 和文本 cross-attention。新增预测头的最后一层
权重和 bias 全部置零，初始输出与同一冻结 Wan 完全一致；训练时主干参与梯度
传播但不更新其参数。一个 latent 时间步的调制向量只广播到该时间步的空间网格。

三视角融合和有序空间展开避免全局平均把机械臂的位置抹掉；得到的仍然是 AdaLN
通道调制，没有引入空间 token attention 或另一个 Transformer 分支。

## 哪些来自论文，哪些是补充实验设计

| 项目 | 来源 |
|---|---|
| Wan2.1 T2V 1.3B、30 层、1536 维、冻结主干和 VAE | 补充材料 B.3 |
| front / side / top、3×16 view embedding、48→32→16 SiLU 融合 | B.2、B.3、Table 2 |
| 从条件 embedding 预测逐层 scale/shift | F.5 对原始动作 AdaLN 的描述 |
| 从融合 latent 到 AdaLN 向量的有序空间展开 + 线性映射 | 本仓库的补充设计；论文未规定 geometry+AdaLN |
| 共用的历史有序展开编码、与条件向量相加 | 本仓库的补充设计；原始 AdaLN 的历史接法未说明 |
| 在 15 个对应位置同时调制 norm1 / norm2，增量零初始化 | 明确选择的实现；F.5 未给出这一级细节 |

这是审稿补充消融的实现，不能称为原始 AdaLN 源码的复原。与原 MKAI 相比，历史
条件的接法也发生变化：本仓库的 15-block 分支通过 cross-attention 读取历史，
AdaLN 版通过历史向量调制归一化。因此，和旧 MKAI 分数的比较涉及整个条件模块，
不能声称只改变了一个归一化运算。

## 数据

这里的三视角是 **URDF + 未来动作生成的机器人几何渲染**，不是三个真实相机
视频，也不包含未来场景 RGB 真值。按 B.2 在机器人基座坐标系中定义：
front 看向 −x、up=+z；side 看向 −y、up=+z；top 看向 −z、up=+x。
渲染器、URDF 以及动作到关节角的转换由现有数据管线提供，本仓库接收渲染视频。
使用和 MKAI 对照完全相同的渲染、裁剪、时间同步与 VAE latent 缩放。

训练原始 manifest 的每行：

```json
{"video":"future.mp4","observation":"history.mp4","geometry_front":"front.mp4","geometry_side":"side.mp4","geometry_top":"top.mp4","prompt":""}
```

也支持 uint8 RGB `.npy` 视频。仿真默认每个渲染与未来视频均为 81 帧、240×320；
历史 5 帧。所有路径相对 manifest 目录；三个视角始终按 front、side、top 排列。

按 README 安装官方 Wan VAE/T5 的可选依赖后缓存：

```bash
ioi-prepare \
  --config configs/geometry_adaln_wan2.1_1.3b.json \
  --manifest data/geometry_raw.jsonl \
  --backbone checkpoints/Wan2.1-T2V-1.3B \
  --output data/geometry_cached
```

输出每个 safetensors 文件，不带 batch 维：

| Key | 形状 |
|---|---|
| video_latent | [16,21,30,40]，训练目标 |
| observation_latent | [16,2,30,40]，已观测历史 |
| geometry_latent | [3,16,21,30,40]，三视角条件 |
| text_context | [L,4096]，原始 T5 输出 |

不需要 `actions` 字段。几何渲染必须和目标未来帧逐帧对应。
VAE 分别编码每个 clip；三个视角用同一组冻结 VAE 权重。学习到的 view embedding
和融合 MLP 在训练模型里执行，不提前缓存。无语言时缓存空字符串的真实 T5 输出。

仅准备推理条件时，manifest 可以省略 `video`，在缓存命令增加 `--condition-only`。
采样从 geometry_latent 推导未来长度，从 observation_latent 取得 H/W，不读取未来真值。
默认的有序读出固定 latent H/W=30/40、历史长度=2；更换分辨率或历史长度需更改配置
并重新训练相应编码器，不能把旧 adapter 直接用于新网格。

## 训练和采样

```bash
ioi-train \
  --config configs/geometry_adaln_wan2.1_1.3b.json \
  --backbone checkpoints/Wan2.1-T2V-1.3B \
  --manifest data/geometry_cached/manifest.jsonl \
  --output runs/geometry_adaln

ioi-sample \
  --backbone checkpoints/Wan2.1-T2V-1.3B \
  --adapter runs/geometry_adaln/last.pt \
  --condition data/geometry_condition.safetensors \
  --output runs/geometry_generated.safetensors \
  --steps 50 --seed 22
```

DDP、梯度累积、检查点恢复和可选 VAE 解码均沿用 README 的用法。

## 配套的原始动作 + AdaLN

`configs/action_adaln_wan2.1_1.3b.json` 使用同一 AdaLN 模型，只把几何编码器换成
原始动作的时间分组 + 两层 MLP。两者共用主干、历史编码、15 个注入位置、预测头
尺寸和训练超参数。动作缓存格式沿用 README；训练命令替换配置、manifest 和输出路径。

这两个新配置可以在相同数据、训练预算、采样和评估协议下重新比较条件表示。
**新 action+AdaLN 不是论文 56.47 的已验证复现；新 geometry+AdaLN 没有实测 FVD。**
两个条件编码器的参数数不同，应一起报告，不宣称严格等参数：

| 参数 | Geometry + AdaLN | Action + AdaLN |
|---|---:|---:|
| 冻结 backbone | 1,418,996,800 | 1,418,996,800 |
| 条件编码器 | 29,494,880 | 2,448,384 |
| 历史编码器 | 58,983,936 | 58,983,936 |
| 15 个 AdaLN 预测头 | 141,649,920 | 141,649,920 |
| 可训练合计 | **230,128,736** | **203,082,240** |

使用 `ioi-inspect --config configs/geometry_adaln_wan2.1_1.3b.json` 可在 meta device
重新统计，无需下载权重。AdaLN 版没有 15-block 分支，故可训练参数不需要凑到 0.7B。

## 验证范围

```bash
ioi-smoke --variant geometry_adaln --output runs/geometry_smoke
ioi-smoke --variant action_adaln --output runs/action_adaln_smoke
pytest -q
```

测试覆盖零初始化与原主干输出一致、三视角与空间/时间顺序、每个视角和历史都能
影响输出、条件编码器获得梯度、主干权重冻结、梯度检查点一致性、adapter 保存加载、
缓存视角独立性和不读取未来目标、训练恢复。缓存管线测试使用 VAE/T5 替身。
真实 VAE/T5、1.3B 预训练权重、CUDA BF16 和正式指标仍需在目标 GPU 环境验证。
