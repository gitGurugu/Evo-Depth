# LiteGauss 训练命令

与 Evo `train.sh` 对齐：

- 默认 **`batch_size=16`**
- 学习率：**线性 warmup（默认 1000 step）+ cosine 降到 0**（与 `Evo_depth/scripts/train.py` 相同公式）
- 峰值 lr 默认 **`1e-4`**（只训 GST/action head；Evo 全模常用 `1e-5`）

每个 `--steps` = 一次 optimizer 更新 = 平均 **16 条样本** 的 loss（内部仍按单条做 GST/encode，再 mean）。

默认 checkpoint 目录：`checkpoints/litegauss/<代名词>/`

```text
checkpoints/litegauss/<name>/
  config.json       # 策略配置 + run 元数据（含 batch_size）
  norm_stats.json   # 与 Evo server 同款归一化统计（单 robot key）
  model.pt          # gst + action_head（+ optimizer）
```

SwanLab：项目默认 `litegauss`，每次实验名 = `--name`（与目录名一致）。

## 准备

```bash
cd ~/Evo-Depth
source .venv/bin/activate
swanlab login          # 首次；之后可省略
```

自检（不训练、不上传）：

```bash
bash scripts/run_litegauss_train.sh --selfcheck
```

## 常用训练

冒烟（默认 batch 16）：

```bash
bash scripts/run_litegauss_train.sh --name smoke-100 --steps 100
# 约看 100×16=1600 条样本
```

推荐 5k step：

```bash
bash scripts/run_litegauss_train.sh \
  --swanlab-project Evo-LiteGauss \
  --name run2-5k \
  --steps 5000
# -> checkpoints/litegauss/run2-5k/
# 约看 5000×16=80000 条样本（~0.29 epoch，数据约 27.5 万）
```

## 参数说明

| 参数 / 环境变量 | 作用 | 默认 |
|-----------------|------|------|
| `--name` / `--run-name` / `RUN_NAME` | 本次代名词（ckpt 目录名 + SwanLab run 名） | `litegauss-steps<N>` |
| `--steps` / `STEPS` | **optimizer 步数**（不是单样本次数） | `100` |
| `--batch-size` / `BATCH_SIZE` | 每步样本数（对齐 Evo） | `16` |
| `--lr` / `LR` | 峰值学习率 | `1e-4` |
| `--warmup-steps` / `WARMUP_STEPS` | 线性 warmup，随后 cosine | `1000` |
| `--output` | 手动指定 **checkpoint 目录** | `checkpoints/litegauss/<name>/` |
| `--swanlab-project` / `SWANLAB_PROJECT` | SwanLab 项目 | `litegauss` |
| `--no-swanlab` / `SWANLAB_MODE=disabled` | 关闭上传 | 默认 cloud |
| `--dataset_config_path` | 数据配置 | `Evo_depth/dataset/config.libero.yaml` |

换实验就换 `--name`；目录已存在会拒绝覆盖。

## 旧版单文件 `.pt` 转目录

```bash
source .venv/bin/activate
python Evo_depth/scripts/train_litegauss_libero.py \
  --export-from-pt checkpoints/litegauss/run1.pt \
  --dataset_config_path Evo_depth/dataset/config.libero.yaml \
  --output checkpoints/litegauss/run1
```
