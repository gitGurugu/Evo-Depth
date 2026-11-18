# LiteGauss 训练命令

LiteGauss = 冻结 InternVL + DA3，只训 GST + flow action head。入口：

```bash
bash scripts/run_litegauss_train.sh ...
```

详细实现：`Evo_depth/scripts/train_litegauss_libero.py`。

## 训练语义（按 step，不是按 epoch）

- **`--steps`**：optimizer 更新次数（停机条件）。
- **`--batch-size B`**：**真 batch**——一步 forward/backward 同时处理 **B 条样本**（张量带 batch 维），然后 **更新一次**权重。
- 单条样本格式固定（多相机、448×448 等），与 B 无关；B 只决定一次叠几条。
- DA3 仍按图独立推理（避免多无关图被当成同一场景多视角）；InternVL / GST / action head 走真 B。
- 日志里的 `~x.xx epoch` 只是「这些步大约扫了多少遍数据」的换算，**不是**按 epoch 停训。

学习率：线性 warmup（默认 1000 step）+ cosine 降到 0（与 Evo `train.py` 同公式）；峰值 lr 默认 `1e-4`。

## Checkpoint

默认目录：`checkpoints/litegauss/<name>/`

```text
checkpoints/litegauss/<name>/
  config.json       # 策略配置 + run 元数据（含 batch_size）
  norm_stats.json   # 与 Evo server 同款归一化统计（单 robot key）
  model.pt          # gst + action_head（+ optimizer）
```

换实验就换 `--name`；目录已存在会拒绝覆盖。

## SwanLab

- 默认项目名：**`litegauss`**（不是 `Evo-LiteGauss`）。
- 要进以前的项目必须显式：`--swanlab-project Evo-LiteGauss`。
- Run 名 = `--name`（与 ckpt 目录名一致）。
- `--no-swanlab` 或 `SWANLAB_MODE=disabled` 关闭上传。

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

单场景过拟合（顶层菜谱，参数写在脚本里；对齐 Evo `train.sh` 用法）：

```bash
bash scripts/train_litegauss_overfit_spatial.sh
# -> 仅 libero_spatial；warmup=50；checkpoints/litegauss/overfit-spatial/
```

冒烟（默认 B=16）：

```bash
bash scripts/run_litegauss_train.sh \
  --swanlab-project Evo-LiteGauss \
  --name smoke-100 \
  --steps 100
# 约看 100×16=1600 条样本
```

24GB 卡上真 B 经验（仅训 GST/head，仍吃激活）：

| B | 说明 |
|---|------|
| 128 | 一般稳定 |
| 192 / 224 | 常用折中；不必是 2 的幂 |
| 256 | 探针偶发能过，长跑易 OOM，不推荐硬开 |

推荐 5k step + 较大真 B：

```bash
bash scripts/run_litegauss_train.sh \
  --swanlab-project Evo-LiteGauss \
  --name run-bs192-5k \
  --steps 5000 \
  --batch-size 192
# -> checkpoints/litegauss/run-bs192-5k/
# 约看 5000×192 条样本
```

自动探测「≤上界」的最大真 B（训前 binary search，含 backward + AdamW 工作区）：

```bash
bash scripts/run_litegauss_train.sh \
  --swanlab-project Evo-LiteGauss \
  --name auto \
  --steps 5000 \
  --batch-size 256 \
  --probe-max-batch
```

注意：probe 选中 256 也不代表长跑绝对稳（碎片 / 不同 batch 峰值可能更高）。偏稳可把上界设 192/224，或 probe 后再手动降一档。

## 参数说明

| 参数 / 环境变量 | 作用 | 默认 |
|-----------------|------|------|
| `--name` / `--run-name` / `RUN_NAME` | 代名词（ckpt 目录 + SwanLab run） | `litegauss-steps<N>` |
| `--steps` / `STEPS` | **optimizer 步数** | `100` |
| `--batch-size` / `BATCH_SIZE` | 真 batch：每步样本数 | `16` |
| `--probe-max-batch` | 训前找 ≤`--batch-size` 的最大可跑真 B | 关 |
| `--lr` / `LR` | 峰值学习率 | `1e-4` |
| `--warmup-steps` / `WARMUP_STEPS` | 线性 warmup，随后 cosine | `1000` |
| `--output` | 手动指定 checkpoint 目录 | `checkpoints/litegauss/<name>/` |
| `--swanlab-project` / `SWANLAB_PROJECT` | SwanLab 项目 | `litegauss` |
| `--no-swanlab` / `SWANLAB_MODE=disabled` | 关闭上传 | 默认 cloud |
| `--dataset_config_path` | 数据配置 | `Evo_depth/dataset/config.libero.yaml` |

## 旧版单文件 `.pt` 转目录

```bash
source .venv/bin/activate
python Evo_depth/scripts/train_litegauss_libero.py \
  --export-from-pt checkpoints/litegauss/run1.pt \
  --dataset_config_path Evo_depth/dataset/config.libero.yaml \
  --output checkpoints/litegauss/run1
```
## LIBERO 评测（LiteGauss server）

不能把目录丢给 `Evo_depth_server.py`（它要 `mp_rank_00_model_states.pt`）。用专用 server，协议与 Evo 相同，客户端不用改：

```bash
# 只起服务（另开终端再跑 client 也可以）
LITEGAUSS_CKPT_DIR=checkpoints/litegauss/run-bs224-400 \
EVO_DEPTH_SERVER_PORT=9000 \
PYTHONPATH=Evo_depth \
  .venv/bin/python -u Evo_depth/scripts/litegauss_server.py

# 或：起 server + LIBERO client
bash scripts/run_litegauss_libero.sh libero_spatial
# 与训练 --name 对齐，日志写到独立目录（不覆盖旧评测）：
LITEGAUSS_CKPT=checkpoints/litegauss/overfit-spatial \
  bash scripts/run_litegauss_libero.sh libero_spatial --name overfit-spatial
# 或直接指定目录：
#   bash scripts/run_litegauss_libero.sh libero_spatial --log-dir /tmp/my_eval
```

默认日志：`LIBERO-evaluation/logs/litegauss_<suite>/`；带 `--name TAG` 则为 `…/litegauss_<suite>_<tag>/`（也可用 `--log-dir` / `LITEGAUSS_LOG_DIR`）。
端口占用时先停掉旧 Evo/LiteGauss server（只杀对应 pid 文件里的进程）。

