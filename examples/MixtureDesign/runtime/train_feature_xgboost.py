#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if any(arg == "--predictor-head" or arg.startswith("--predictor-head=") for arg in argv):
        raise SystemExit(
            "train_feature_xgboost.py always uses --predictor-head xgboost; "
            "remove any explicit --predictor-head override."
        )

    repo_root = Path(__file__).resolve().parents[4]
    scripts_training_dir = repo_root / "scripts_training"
    for path in (repo_root, scripts_training_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    import train_feature_models

    return train_feature_models.main(["--predictor-head", "xgboost", *argv])


if __name__ == "__main__":
    raise SystemExit(main())
