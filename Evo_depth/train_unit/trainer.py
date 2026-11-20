import json
import logging
import math
import random
from dataclasses import asdict
from datetime import datetime
from importlib.util import find_spec
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .config import TrainConfig
from .protocol import ModelAdapter


def build_param_groups(model, weight_decay, parameters):
    selected = {id(parameter) for parameter in parameters}
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) not in selected:
            continue
        is_bias = name.endswith("bias") or ".bias" in name
        is_norm = parameter.dim() == 1 or "norm" in name.lower()
        (no_decay if is_bias or is_norm else decay).append(parameter)
    if not decay and not no_decay:
        raise ValueError("Adapter returned no registered trainable parameters")
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def inspect_named_submodules(module_dict: dict, logger: logging.Logger, verbose: bool = True) -> None:
    """Log trainable vs frozen params per submodule (same role as Evo train.py)."""
    total_all, trainable_all = 0, 0
    logger.info("Parameter Inspection by Module:")
    logger.info("=" * 70)
    for module_name, module in module_dict.items():
        total, trainable = 0, 0
        logger.info("Module: %s", module_name)
        logger.info("-" * 70)
        for name, param in module.named_parameters():
            num_params = param.numel()
            total += num_params
            if param.requires_grad:
                trainable += num_params
                if verbose:
                    logger.info("Trainable %-55s | shape: %-20s | %6.2fM",
                                name, str(tuple(param.shape)), num_params / 1e6)
            elif verbose:
                logger.info("Frozen    %-55s | shape: %-20s | %6.2fM",
                            name, str(tuple(param.shape)), num_params / 1e6)
        logger.info("-" * 70)
        logger.info("Total     : %.2fM", total / 1e6)
        logger.info("Trainable : %.2fM", trainable / 1e6)
        logger.info("Frozen    : %.2fM", (total - trainable) / 1e6)
        total_all += total
        trainable_all += trainable
    logger.info("=" * 70)
    logger.info("ALL TOTAL     : %.2fM", total_all / 1e6)
    logger.info("ALL TRAINABLE : %.2fM", trainable_all / 1e6)
    logger.info("ALL FROZEN    : %.2fM", (total_all - trainable_all) / 1e6)
    logger.info("=" * 70)


def validate_train_state(train_state, config):
    saved_config = train_state.get("config")
    if saved_config is None:
        raise ValueError("Checkpoint lacks train.config; cannot verify resume schedule")
    for name in ("lr", "weight_decay", "max_steps", "warmup_steps"):
        if saved_config[name] != getattr(config, name):
            raise ValueError(f"Resume requires unchanged {name}")
    step = train_state["step"]
    if type(step) is not int or not 0 <= step <= config.max_steps:
        raise ValueError("Checkpoint step must be an integer between 0 and max_steps")
    if train_state["scheduler"]["last_epoch"] != step:
        raise ValueError("Checkpoint scheduler and step disagree")
    return step


def save_checkpoint(save_dir, step, model, optimizer, scheduler, accelerator, adapter, config,
                    best_loss=float("inf"), tag=None):
    tag = tag or f"step_{step}"
    checkpoint_dir = save_dir / tag
    train_state = {"step": step, "best_loss": best_loss,
                   "scheduler": scheduler.state_dict(), "config": asdict(config)}
    accelerator.wait_for_everyone()
    uses_deepspeed = accelerator.state.deepspeed_plugin is not None
    if uses_deepspeed:
        # All ranks must save their shards; DeepSpeed owns model and optimizer state.
        model.save_checkpoint(str(save_dir), tag=tag, client_state={"train": train_state})
    elif accelerator.is_main_process:
        checkpoint_path = save_dir / f"{tag}.pt"
        temporary_path = checkpoint_path.with_suffix(".pt.tmp")
        train_state["optimizer"] = optimizer.state_dict()
        try:
            torch.save({"train": train_state,
                        "model": adapter.export_state(accelerator.unwrap_model(model))}, temporary_path)
            temporary_path.replace(checkpoint_path)
        finally:
            temporary_path.unlink(missing_ok=True)
    if accelerator.is_main_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        adapter_config = getattr(adapter, "config", None)
        checkpoint_config = dict(adapter_config) if isinstance(adapter_config, dict) else {}
        checkpoint_config["train"] = asdict(config)
        with (checkpoint_dir / "config.json").open("w", encoding="utf-8") as stream:
            json.dump(checkpoint_config, stream, indent=2)
        norm_stats = getattr(adapter, "norm_stats", None)
        if norm_stats is not None:
            with (checkpoint_dir / "norm_stats.json").open("w", encoding="utf-8") as stream:
                json.dump(norm_stats, stream, indent=2)
        else:
            (checkpoint_dir / "norm_stats.json").unlink(missing_ok=True)
        if uses_deepspeed:
            checkpoint_meta = {"type": "ds_model", "version": 0.0,
                               "checkpoints": "mp_rank_00_model_states.pt"}
            with (checkpoint_dir / "checkpoint.json").open("w", encoding="utf-8") as stream:
                json.dump(checkpoint_meta, stream, indent=2)
        logging.getLogger(__name__).info("Saved checkpoint %s (step=%s, best_loss=%.6f)",
                                        tag, step, best_loss)
    accelerator.wait_for_everyone()


def load_checkpoint_with_deepspeed(model, resume_path, resume_pretrain=False):
    checkpoint_path = Path(resume_path)
    load_path, client_state = model.load_checkpoint(
        str(checkpoint_path.parent), tag=checkpoint_path.name,
        load_module_strict=True, load_optimizer_states=not resume_pretrain,
        load_lr_scheduler_states=False,
    )
    if load_path is None:
        raise ValueError(f"DeepSpeed checkpoint not found: {resume_path}")
    return {} if resume_pretrain else client_state["train"]


class Trainer:
    def __init__(self, config: TrainConfig, adapter: ModelAdapter):
        self.config = config
        self.adapter = adapter

    def run(self) -> None:
        # Keep Accelerate/DeepSpeed optional at import time, including on CPU-only hosts.
        try:
            from accelerate import Accelerator, DeepSpeedPlugin
        except ImportError as error:
            raise ImportError("Train unit requires accelerate to run") from error

        config, adapter = self.config, self.adapter
        cpu = torch.device(config.device).type == "cpu"
        plugin = None
        if config.deepspeed_config_file and not cpu:
            if find_spec("deepspeed") is None:
                raise ImportError("CUDA DeepSpeed training requires deepspeed to be installed")
            with open(config.deepspeed_config_file, encoding="utf-8") as stream:
                deepspeed_config = json.load(stream)
            if deepspeed_config.get("gradient_accumulation_steps", 1) not in (1, "auto"):
                raise ValueError("Train unit requires DeepSpeed gradient_accumulation_steps=1")
            if "optimizer" in deepspeed_config or "scheduler" in deepspeed_config:
                raise ValueError("Train unit owns AdamW and LambdaLR; remove DS optimizer/scheduler entries")
            # Accelerate's DS backward also steps, so clipping must be configured before prepare.
            deepspeed_config.update(
                train_micro_batch_size_per_gpu=config.batch_size,
                train_batch_size="auto",
                gradient_accumulation_steps=1,
                gradient_clipping=config.grad_clip_norm,
            )
            plugin = DeepSpeedPlugin(hf_ds_config=deepspeed_config)
        accelerator = Accelerator(cpu=cpu, mixed_precision=config.mixed_precision,
                                  deepspeed_plugin=plugin, gradient_accumulation_steps=1)
        uses_deepspeed = accelerator.state.deepspeed_plugin is not None
        if plugin is not None and not uses_deepspeed:
            raise RuntimeError("Requested DeepSpeed was not activated by Accelerate")

        random.seed(config.seed)
        torch.manual_seed(config.seed)

        checkpoint = None
        step = 0
        best_loss = float("inf")
        if config.resume_path is not None and not uses_deepspeed:
            checkpoint = torch.load(config.resume_path, map_location="cpu", weights_only=True)
            if not config.resume_pretrain:
                train_state = checkpoint["train"]
                step = validate_train_state(train_state, config)

        model = adapter.build_model(str(accelerator.device))
        if checkpoint is not None:
            adapter.load_state(model, checkpoint["model"])

        loader = adapter.build_loader(config.batch_size, config.num_workers)
        parameters = list(adapter.trainable_parameters(model))
        if not parameters:
            raise ValueError("Adapter returned no trainable parameters")

        optimizer = AdamW(build_param_groups(model, config.weight_decay, parameters), lr=config.lr)
        model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

        def lr_factor(completed_steps: int) -> float:
            if completed_steps < config.warmup_steps:
                return completed_steps / config.warmup_steps

            progress = (completed_steps - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

        scheduler = LambdaLR(optimizer, lr_factor)
        if config.resume_path is not None:
            if uses_deepspeed:
                train_state = load_checkpoint_with_deepspeed(model, config.resume_path,
                                                            config.resume_pretrain)
                if not config.resume_pretrain:
                    step = validate_train_state(train_state, config)
            elif not config.resume_pretrain:
                optimizer.load_state_dict(train_state["optimizer"])
            if not config.resume_pretrain:
                scheduler.load_state_dict(train_state["scheduler"])
                best_loss = train_state.get("best_loss", float("inf"))

        save_dir = Path(config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        file_handler = None
        if accelerator.is_main_process:
            log_path = save_dir / f"train_{datetime.now():%Y%m%d_%H%M%S_%f}.log"
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logger.addHandler(file_handler)
            logger.info("run=%s resume_path=%s resume_pretrain=%s step=%s best_loss=%s",
                        config.run_name, config.resume_path, config.resume_pretrain, step, best_loss)
        unwrapped = accelerator.unwrap_model(model)
        adapter.set_train_mode(unwrapped)
        if accelerator.is_main_process:
            inspect_modules = getattr(adapter, "inspect_modules", None)
            module_dict = inspect_modules(unwrapped) if callable(inspect_modules) else {}
            if module_dict:
                inspect_named_submodules(module_dict, logger)
        swanlab = None
        try:
            if not config.disable_swanlab and accelerator.is_main_process:
                import swanlab

                swanlab.init(project=config.swanlab_project, name=config.run_name, config=asdict(config))

            try:
                batches_per_epoch = len(loader)
            except TypeError:
                batches_per_epoch = 0

            # shortcut: resume restores training state, not RNG/data position; extend for exact replay.
            while step < config.max_steps:
                had_batch = False
                for batch in loader:
                    had_batch = True
                    optimizer.zero_grad(set_to_none=True)
                    loss = adapter.forward_loss(model, batch)
                    if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
                        raise ValueError("Adapter forward_loss must return a scalar tensor")
                    finite = torch.isfinite(loss.detach()).to(dtype=torch.int32)
                    if accelerator.reduce(finite, reduction="sum").item() != accelerator.num_processes:
                        accelerator.print(f"[{config.run_name}] Non-finite loss at step {step}; skipping batch")
                        if accelerator.is_main_process:
                            logger.warning("Non-finite loss at step %s; skipping batch", step)
                        continue

                    learning_rate = optimizer.param_groups[0]["lr"]
                    accelerator.backward(loss)
                    if uses_deepspeed:
                        # DS backward already updated/clipped; the engine caches the pre-clip global norm.
                        grad_norm = model.get_global_grad_norm()
                    elif hasattr(accelerator, "clip_grad_norm_"):
                        grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip_norm,
                                                                  error_if_nonfinite=True)
                    optimizer.step()
                    if accelerator.optimizer_step_was_skipped:
                        continue
                    scheduler.step()
                    step += 1
                    loss_value = accelerator.reduce(loss.detach(), reduction="mean").item()

                    if step % config.log_interval == 0:
                        grad_norm = float(grad_norm)
                        metrics = {"step": step, "loss": loss_value, "lr": learning_rate,
                                   "learning_rate": learning_rate, "grad_norm": grad_norm}
                        if batches_per_epoch:
                            metrics["current_epoch"] = step / batches_per_epoch
                        message = f"[{config.run_name}] " + " ".join(
                            f"{name}={value:.6g}" for name, value in metrics.items()
                            if name != "learning_rate"
                        )
                        accelerator.print(message)
                        if accelerator.is_main_process:
                            logger.info(message)
                        if swanlab is not None:
                            swanlab.log(metrics)

                    # All ranks see the same reduced loss and enter DS shard saving together.
                    if loss_value < best_loss:
                        best_loss = loss_value
                        save_checkpoint(save_dir, step, model, optimizer, scheduler, accelerator,
                                        adapter, config, best_loss, tag="step_best")
                    if step % config.ckpt_interval == 0 or step == config.max_steps:
                        save_checkpoint(save_dir, step, model, optimizer, scheduler, accelerator,
                                        adapter, config, best_loss)

                    if step == config.max_steps:
                        break

                if not had_batch:
                    raise ValueError("Adapter loader produced no batches")
        except Exception:
            if accelerator.is_main_process:
                logger.exception("Training failed at step %s", step)
            raise
        finally:
            if file_handler is not None:
                logger.removeHandler(file_handler)
                file_handler.close()
            if swanlab is not None:
                swanlab.finish()
