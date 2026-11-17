# LiteGauss 训练命令

Batch size = 1。  
默认 checkpoint 目录：`checkpoints/litegauss/<代名词>/`

```text
checkpoints/litegauss/<name>/
  config.json       # 策略配置 + run 元数据
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

冒烟 100 step：

```bash
bash scripts/run_litegauss_train.sh --name smoke-100 --steps 100
# -> checkpoints/litegauss/smoke-100/{config.json,norm_stats.json,model.pt}
```

第一版可测 ckpt（推荐）：

```bash
bash scripts/run_litegauss_train.sh --name run1-1k --steps 1000
# -> checkpoints/litegauss/run1-1k/
```

加长一点：

```bash
bash scripts/run_litegauss_train.sh --name run2-5k --steps 5000
```

## 参数说明

| 参数 / 环境变量 | 作用 | 默认 |
|-----------------|------|------|
| `--name` / `--run-name` / `RUN_NAME` | 本次代名词（ckpt 目录名 + SwanLab run 名） | `litegauss-steps<N>` |
| `--steps` / `STEPS` | 训练步数 | `100` |
| `--output` | 手动指定 **checkpoint 目录** | `checkpoints/litegauss/<name>/` |
| `--swanlab-project` / `SWANLAB_PROJECT` | SwanLab 项目 | `litegauss` |
| `--no-swanlab` / `SWANLAB_MODE=disabled` | 关闭上传 | 默认 cloud |
| `--dataset_config_path` | 数据配置 | `Evo_depth/dataset/config.libero.yaml` |

同一次训练里：**代名词只设一次**。换实验就换 `--name`；目录已存在会拒绝覆盖。

## 指定项目名示例

```bash
bash scripts/run_litegauss_train.sh \
  --swanlab-project Evo-LiteGauss \
  --name run1-1k \
  --steps 1000
```

## 手动指定输出目录

```bash
bash scripts/run_litegauss_train.sh \
  --name run1-1k \
  --steps 1000 \
  --output checkpoints/litegauss/custom/run1-1k
```

## 旧版单文件 `.pt` 转目录

```bash
source .venv/bin/activate
python Evo_depth/scripts/train_litegauss_libero.py \
  --export-from-pt checkpoints/litegauss/run1.pt \
  --dataset_config_path Evo_depth/dataset/config.libero.yaml \
  --output checkpoints/litegauss/run1
```
