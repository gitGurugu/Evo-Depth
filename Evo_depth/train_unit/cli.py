import argparse
import json
from dataclasses import fields
from importlib import import_module

from .config import TrainConfig
from .trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Accelerate/DeepSpeed Train unit")
    parser.add_argument("--adapter", required=True, help="Bundled name (evo) or module exporting Adapter")
    parser.add_argument(
        "--adapter-config",
        help="Adapter JSON: model/dataset keys; optional train{} for TrainConfig defaults",
    )
    for field in fields(TrainConfig):
        value_type = type(field.default) if field.default is not None else str
        if value_type is bool:
            parser.add_argument(
                f"--{field.name}",
                action=argparse.BooleanOptionalAction,
                default=argparse.SUPPRESS,
            )
        else:
            parser.add_argument(f"--{field.name}", type=value_type, default=argparse.SUPPRESS)

    arguments = vars(parser.parse_args())
    module_name = arguments.pop("adapter")
    config_path = arguments.pop("adapter_config", None)

    train_options = {}
    if config_path is not None:
        with open(config_path, encoding="utf-8") as stream:
            adapter_config = json.load(stream)
        if not isinstance(adapter_config, dict):
            parser.error("adapter configuration must be a JSON object")
        train_options = adapter_config.get("train", {})
        if not isinstance(train_options, dict):
            parser.error("train must be a JSON object when present")
        unknown = train_options.keys() - {field.name for field in fields(TrainConfig)}
        if unknown:
            parser.error(f"Unknown train fields: {', '.join(sorted(unknown))}")

    # CLI wins over JSON train{}; missing keys keep TrainConfig defaults.
    train_options.update(arguments)
    if "." not in module_name:
        module_name = f"{__package__}.adapters.{module_name}"

    config = TrainConfig(**train_options)
    adapter_class = import_module(module_name).Adapter
    adapter = adapter_class(config_path=config_path) if config_path is not None else adapter_class()
    Trainer(config, adapter).run()


if __name__ == "__main__":
    main()
