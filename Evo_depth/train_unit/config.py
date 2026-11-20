import math
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 16
    max_steps: int = 1000
    warmup_steps: int = 0
    grad_clip_norm: float = 1.0
    log_interval: int = 1
    ckpt_interval: int = 100
    save_dir: str = "checkpoints"
    resume_path: str | None = None
    resume_pretrain: bool = False
    num_workers: int = 0
    seed: int = 0
    device: str = "cpu"
    run_name: str = "default_run"
    deepspeed_config_file: str | None = str(Path(__file__).resolve().parents[1] / "ds_config.json")
    mixed_precision: str | None = None
    disable_swanlab: bool = True
    swanlab_project: str = "default_run"

    def __post_init__(self):
        if self.mixed_precision not in (None, "no", "fp16", "bf16"):
            raise ValueError("mixed_precision must be no, fp16, bf16, or null")
        for name in ("disable_swanlab", "resume_pretrain"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if self.resume_pretrain and not self.resume_path:
            raise ValueError("resume_pretrain requires resume_path")

        for name in ("batch_size", "max_steps", "log_interval", "ckpt_interval"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        for name in ("warmup_steps", "num_workers"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")

        for name in ("lr", "weight_decay", "grad_clip_norm"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
