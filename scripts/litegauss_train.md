# LiteGauss 训练命令

LiteGauss = 冻结 InternVL + DA3，只训 GST + flow action head。入口：

```bash
bash scripts/run_litegauss_train.sh ...
```

详细实现：`Evo_depth/scripts/train_litegauss_libero.py`。

## 先缓存 DA3，再训练（默认必须命中）

在服务器仓库根目录运行；`config.libero.yaml` 中的四个数据路径须先指向本机 LIBERO/LeRobot 数据。
两个入口都保留 `HF_ENDPOINT=https://hf-mirror.com` 和清除失效代理的处理。

```bash
cd ~/Evo-Depth
source .venv/bin/activate
export DATASET_CONFIG="$PWD/Evo_depth/dataset/config.libero.yaml"
export LITEGAUSS_DEPTH_CACHE="$PWD/data/litegauss_da3_depth_cache"

# 只需构建一次；中断后原命令重跑会跳过已有有效键。
bash scripts/run_litegauss_depth_cache.sh --action-horizon 16

# 默认 cache-required：缺失立即报错，也不加载 DA3 权重。
bash scripts/run_litegauss_train.sh --name cached-bs16-5k --steps 5000 --batch-size 16

# 可选：命中读缓存，未命中才逐图在线 DA3（不自动回写缓存）。
bash scripts/run_litegauss_train.sh --name mixed-smoke --steps 100 --allow-online-da3
```

缓存构建器复用 `LiteGaussDataset`、相同 `view_map`、视频解码与无增强 Resize→RGB 转换，
每个 DA3 调用严格为 `inference_images([rgb])`；agent/wrist 也分别推理。
默认 action horizon=16；自定义训练 horizon 时构建器传相同 `--action-horizon`。

键布局为 `<cache-root>/da3-base-mono-rgb448-v1/<agent|wrist>/<sha256前2位>/<sha256>.npz`。
哈希输入是实际训练 RGB 的 448×448 像素字节；LeRobot 当前不返回 episode/frame，且解码失败会随机换样本，
因此不以请求索引推断身份。相同相机相同像素跨 episode/数据集/action window 自动去重；像素变化生成新键。
每个文件只存 `depth`：DA3 原输出分辨率的有限 float32 H×W，无额外尺度归一化。
训练按每张 RGB 的 K 缩放到该深度分辨率，再走原 GST patch 对齐；B>1 按 `[B][V]` 传深度并保持真 batch。

重跑仍会解码 RGB 计算键，但有效文件跳过 DA3；全部命中时构建器不会加载 DA3 权重。
进度输出样本数、新写入数、跳过数和 ETA。写入采用同目录临时文件 + 原子替换；损坏文件明确报错，
确认后删除该文件再重跑。构建沿用训练 loader 的坏视频随机换样本行为，偶发未覆盖帧会在严格训练时暴露。
缓存版本固定对应当前 `depth-anything/da3-base` 单图默认推理；更换权重/DA3 预处理实现后使用新的空缓存目录重建。
约 55 万 RGB 的 float32 原始深度可能占数百 GB，NPZ 压缩后以实际磁盘占用为准；未提供分片或自动清理。
推理/server 仍默认在线 DA3。

独立 CPU 自检，不需要数据或 DA3 权重：

```bash
bash scripts/run_litegauss_train.sh --selfcheck
bash scripts/run_litegauss_depth_cache.sh --selfcheck
```

## 训练语义（按 step，不是按 epoch）

- **`--steps`**：optimizer 更新次数（停机条件）。
- **`--batch-size B`**：**真 batch**——一步 forward/backward 同时处理 **B 条样本**（张量带 batch 维），然后 **更新一次**权重。
- 单条样本格式固定（多相机、448×448 等），与 B 无关；B 只决定一次叠几条。
- DA3 在预处理时按图独立推理；训练默认全部读缓存，显式允许时缺失才单图在线推理；InternVL / GST / action head 走真 B。
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
| `--depth-cache-root` / `LITEGAUSS_DEPTH_CACHE` | 深度缓存目录（构建和训练相同） | 仓库根目录 `data/litegauss_da3_depth_cache` |
| `--allow-online-da3` | 训练缺失缓存时允许单图在线 DA3 | 关：缺失报错 |

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
LITEGAUSS_CKPT=checkpoints/litegauss/run-bs224-400 \
  bash scripts/run_litegauss_libero.sh libero_goal
```

日志：`LIBERO-evaluation/logs/litegauss_<suite>/`。端口占用时先停掉旧 Evo/LiteGauss server（只杀对应 pid 文件里的进程）。

