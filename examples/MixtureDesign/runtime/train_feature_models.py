#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_scripts_training_module():
    module_path = (
        Path(__file__).resolve().parents[4]
        / "scripts_training"
        / "train_feature_models.py"
    )
    spec = importlib.util.spec_from_file_location("chempropmix_scripts_train_feature_models", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load scripts_training feature-model implementation from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_IMPL = _load_scripts_training_module()

DescriptorOnlyRegressor = _IMPL.DescriptorOnlyRegressor
DescriptorFeatureRegressor = _IMPL.DescriptorFeatureRegressor
XGBoostConfig = _IMPL.XGBoostConfig
_build_parser = _IMPL._build_parser
run_training = _IMPL.run_training
run_hpo = _IMPL.run_hpo


def main(argv: list[str] | None = None) -> int:
    return _IMPL.main(argv)


def __getattr__(name: str):
    return getattr(_IMPL, name)


if __name__ == "__main__":
    raise SystemExit(main())
