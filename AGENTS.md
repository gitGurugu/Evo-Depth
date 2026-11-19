# AGENTS.md

给编码 Agent（Cursor / Codex / Claude Code）用的仓库说明。优先按本文执行，不要自己臆造环境与流程。

## 这是什么仓库

NeurIPS 2027 **LiteGauss**：在 **Evo** 基线（InternVL + DA3 深度 + flow-matching 动作头）上加 Gaussian Spatial Tokenizer（GST）的 VLA。

流水线：冻结 InternVL + DA3 → GST tokens → flow action head。训练细节以 `scripts/litegauss_train.md` 为准（flags / env 的权威来源）。

**原版 Evo 代码树只作参考。** 可以读、学写法；除非用户明确要求，**不要改**继承来的 Evo 代码。新工作只放在下面的 LiteGauss 路径里。

## 目录结构

| 路径 | 作用 |
|------|------|
| `Evo_depth/model/litegauss/` | **可改** — GST + `LiteGauss` 策略 |
| `Evo_depth/dataset/litegauss_*.py` | **可改** — LiteGauss 数据集 |
| `Evo_depth/scripts/train_litegauss_*.py`、`litegauss_server.py` | **可改** — 训练 / 推理服务 |
| `scripts/run_litegauss_*.sh`、`scripts/litegauss_train.md` | **可改** — 入口脚本 + 训练说明 |
| `Evo_depth/`（其余）、各 eval 客户端、Evo `train.sh` | **参考** — 原版 Evo；默认不要改 |
| `LIBERO-evaluation/` 等 | Evo 时期的评测客户端；只有接到 LiteGauss 接线任务时再动 |

默认推理端口：**9000**（`EVO_DEPTH_SERVER_PORT` / 客户端 `server_url`）。

## 本地环境

- 项目 venv：仓库根目录 `.venv`（已 gitignore）。激活：`.\.venv\Scripts\Activate.ps1`
- **不要**复用或覆盖文档里的服务器端环境：`.venv-libero`、`.venv-libero-plus`、`.venv-metaworld`、`.venv-vla-arena`
- README 目标 Python **3.10**；本地 `.venv` 可能是 3.12 — 安装失败时先怀疑版本不一致

## 关键代码路径

- LiteGauss 策略：`Evo_depth/model/litegauss/`（`LiteGauss`、`GaussianSpatialTokenizer`）
- 训练入口：`scripts/run_litegauss_train.sh` → `Evo_depth/scripts/train_litegauss_libero.py`
- Evo 基线（只读参考）：DA3 `…/depth_anything_3/depth_anything_v3.py` → `DAV3Module`；InternVL 在 `Evo_depth/model/internvl3/`

## 评测注意

- 先起 server，等到 ready 横幅后再起 client
- 只杀评测文档 pid 文件里记录的进程 — 禁止瞎 `pkill`
- PLUS 空间探测：优先 **camera** / **layout**；真值深度需要 LIBERO/robosuite `camera_depths=True` + `get_real_depth_map`（Evo checkpoint 不带）

## 开发流程（本机 → GitHub → 服务器）

模式：**本机改代码 → git push 到 GitHub → 服务器 pull 再跑**。不要在服务器上直接改业务代码。

- Remote：`origin` → `https://github.com/gitGurugu/Evo-Depth.git`（`upstream` 是原版 Evo，只读参考，别往 upstream push）
- 服务器：`ssh LiteGauss`（`~/.ssh/config`；密钥同 compshare `EVO.pem`）。工作目录通常 `~/Evo-Depth`，拉代码：`git pull`。机器特有路径/密钥写 `AGENTS.local.md`（已 gitignore；有则读，勿编造，勿提交）

```powershell
ssh LiteGauss
# 或：ssh -p 23 -i C:/Users/20494/.ssh/EVO.pem root@117.50.221.107
```

### Git：commit + push 当作一步

- 功能/任务写完就 **commit 并立刻 push**（一步做完；不要只 commit 不 push）。
- **标题**：简短中文，一眼能看懂改了什么。
- **正文**：几条要点即可，写「做了什么 / 为什么」；不要流水账、不要堆文件清单。
- 用户稍后对**同一功能**提出修改时：把该功能相关的多次 commit **squash 成 1 个**，再用同一条简洁中文说明，然后 push（若已推过，用 `git push --force-with-lease`，只压缩本功能未合并的提交，别动别人的历史）。
- 不要提交：`.venv/`、`*/logs/`、checkpoints、secrets、`AGENTS.local.md`。

## 工作约定

- **有 Codex 可用时**：编码实现优先交给 Codex；主 Agent 负责任务拆解、约束对齐（本文）、审查 diff、跑通验证、以及按开发流程 commit/push。Codex 不可用或任务只是改文档/查状态时，主 Agent 自己做即可。
- **Codex 调用方式**：本环境没有 Codex MCP；用 Claude Codex 插件的 `codex-companion` / `codex-rescue` 转发。默认：`--model gpt-6.1-sol --effort high`（用户另指定时再改）。
- 改动尽量小；复用 Evo 时是 **抄思路**，不是去改 Evo 文件
- 不要为了「整洁」去改无关的评测文档或虚拟环境布局
- 训练约定：冻结 InternVL + DA3；只训 GST + flow action head（除非用户另说）

### 编码风格（可读优先；手术式改动）

只作用于 **新建的 LiteGauss 代码和你改到的行**。不要重排 Evo 参考代码的风格。

1. **命名**：一眼能看懂 —— 优先完整单词（`depth_map`、`action_horizon`）。禁止玄学缩写（`proc`、`tmp2`、`x1`）。领域惯用短写可以（`batch`、`loss`、`lr`、`cfg`、`num_epochs`、`da3`、`gst`）。
2. **归类**：类属性/成员写在一起；相关逻辑挨着放（初始化跟初始化、forward 跟 forward、loss 跟 loss）。
3. **朴素写法**：不影响性能时，优先直白逻辑和简单 `for`，少用花哨/堆叠写法。正在改的代码里已有的 torch/向量化写法要沿用；不要为了「好读」把能跑的向量化改成循环。
4. **空行**：你写或改到的代码，逻辑块之间空一行。不要整文件只为加空行而重排；也不要把无关语句黏在一起。
5. **手术式**：最小 diff；扩展 LiteGauss 文件时跟附近风格一致；不要臆造抽象。
