from collections.abc import Iterable
from typing import Any, Protocol

import torch
from torch import nn
from torch.utils.data import DataLoader


class ModelAdapter(Protocol):
    def build_model(self, device: str) -> nn.Module: ...

    def set_train_mode(self, model: nn.Module) -> None:
        """Override to keep frozen submodules in evaluation mode."""
        model.train()

    def build_loader(self, batch_size: int, num_workers: int) -> DataLoader: ...

    def forward_loss(self, model: nn.Module, batch: Any) -> torch.Tensor: ...

    def export_state(self, model: nn.Module) -> dict: ...

    def load_state(self, model: nn.Module, state: dict) -> None: ...

    def trainable_parameters(self, model: nn.Module) -> Iterable[nn.Parameter]:
        return (parameter for parameter in model.parameters() if parameter.requires_grad)

    def inspect_modules(self, model: nn.Module) -> dict[str, nn.Module]:
        """Named submodules for freeze/train parameter inspection. Empty skips inspect."""
        return {}
