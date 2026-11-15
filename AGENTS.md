# AGENTS.md — Evo-Depth

Cross-tool notes for coding agents (Cursor / Codex / Claude Code). Prefer this over inventing setup.

## What this repo is

NeurIPS 2026 **Evo-Depth**: lightweight depth-enhanced VLA (IDEM/DA3 → SEM/FiLM → flow-matching action expert). Paper: arXiv:2605.14950.

## Layout

| Path | Role |
|------|------|
| `Evo_depth/` | Model, train, websocket server (`scripts/Evo_depth_server.py`) |
| `LIBERO-evaluation/` | LIBERO client |
| `LIBERO-PLUS-evaluation/` | LIBERO-PLUS robustness client (7 perturbation axes) |
| `Metaworld-evaluation/` | MetaWorld MT50 |
| `VLA-Arena-evaluation/` | VLA-Arena |

Default inference port: **9000** (`EVO_DEPTH_SERVER_PORT` / client `server_url`).

## Local env

- Project venv: repo-root `.venv` (gitignored). Activate: `.\.venv\Scripts\Activate.ps1`
- Do **not** reuse or overwrite server-side envs documented as `.venv-libero`, `.venv-libero-plus`, `.venv-metaworld`, `.venv-vla-arena`
- README targets Python **3.10**; local `.venv` may be 3.12 — flag version skew if installs fail

## Critical code paths

- DA3 wrapper: `Evo_depth/model/depth_anything_3/depth_anything_v3.py` → `DAV3Module`
- Embed + SEM/FiLM: `Evo_depth/model/internvl3/internvl3_embedder_da3.py`
- Action-facing tokens: `fused` / `fused_tokens` **after** FiLM, before `action_head` (`return_cls_only=False` in training)
- Train recipe: `Evo_depth/train.sh` / `train_a5000.sh` (3-stage progressive alignment)

## Eval hygiene

- Start server, wait for `Evo_depth server running`, then client
- Kill only PIDs recorded in the eval guides’ pid files — no blanket `pkill`
- PLUS spatial probes: prefer **camera** / **layout**; GT depth needs LIBERO/robosuite `camera_depths=True` + `get_real_depth_map` (not shipped in Evo checkpoint)

## Remote: A5000 / ilsr (IPv6 link-local)

SSH alias in `~/.ssh/config`: **`ilsr`** (user `kjl`). This is the lab box used with Evo (paths like `/home/kjl/Evo-Depth` on server).

| Item | Value |
|------|--------|
| Alias | `ssh ilsr` |
| User | `kjl` |
| Address | `fe80::3eec:efff:feab:46a8` (link-local IPv6) |
| Zone / iface index | Windows「以太网 2」的 **接口号**（不是管道）。`ssh` 写成 `%INDEX`；`~/.ssh/config` 里写成 `%%INDEX` |
| Current index | **18** → HostName `fe80::3eec:efff:feab:46a8%%18` |
| Auth | Password (keys currently rejected). **Do not put the password in this file.** |

```powershell
# 查当前接口号（换网卡/重装驱动后会变；以前用过 21，现为 18）
Get-NetAdapter | Where-Object Status -eq 'Up' | Format-Table Name, InterfaceIndex

ssh ilsr
# 或显式：
ssh kjl@fe80::3eec:efff:feab:46a8%18
```

IPv4 `192.168.1.1` / `192.168.2.1` 从这台 PC 通常超时；用 IPv6 + 正确接口号。

Password / other secrets: fill `AGENTS.local.md` (gitignored). Agents: read that file if present; never commit it.

## Working rules

- Smallest diff; reuse existing patterns
- Don’t commit `.venv/`, logs under `*/logs/`, checkpoints, secrets, or `AGENTS.local.md`
- Don’t rewrite unrelated eval guides or virtualenv layouts “for cleanliness”
