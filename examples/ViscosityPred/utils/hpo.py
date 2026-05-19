from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

PATHLIKE_ARG_KEYS = {
    "mix_csv",
    "split_csv",
    "output",
    "metrics_out",
    "parity_plot_out",
    "output_dir",
    "outputs_root",
    "hpo_results_dir",
    "hpo_manifest",
}

NORMALIZATION_METHODS = ("std", "iqr", "mad", "range")


def choice(values: list[Any]) -> dict[str, Any]:
    return {"sampler": "choice", "values": list(values)}


def loguniform(lower: float, upper: float) -> dict[str, Any]:
    return {"sampler": "loguniform", "lower": float(lower), "upper": float(upper)}


def uniform(lower: float, upper: float) -> dict[str, Any]:
    return {"sampler": "uniform", "lower": float(lower), "upper": float(upper)}


def randint(lower: int, upper: int) -> dict[str, Any]:
    return {"sampler": "randint", "lower": int(lower), "upper": int(upper)}


def clone_namespace(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(**vars(args).copy())


def _coerce_arg_value(key: str, value: Any) -> Any:
    if key in PATHLIKE_ARG_KEYS and value is not None and not isinstance(value, Path):
        return Path(value)
    return value


def apply_trial_config(args: argparse.Namespace, config: dict[str, Any]) -> argparse.Namespace:
    for key, value in config.items():
        setattr(args, key, _coerce_arg_value(key, value))
    return args


def _estimate_manifest_entry_cost(entry: dict[str, Any]) -> tuple[float, str]:
    config = entry.get("config", {})
    mix_csv = config.get("mix_csv")
    split_csv = config.get("split_csv")
    mix_path = Path(mix_csv) if mix_csv is not None else None
    split_path = Path(split_csv) if split_csv is not None else None

    try:
        mix_size = float(mix_path.stat().st_size) if mix_path is not None and mix_path.exists() else float("inf")
    except OSError:
        mix_size = float("inf")
    try:
        split_size = float(split_path.stat().st_size) if split_path is not None and split_path.exists() else 0.0
    except OSError:
        split_size = 0.0
    return mix_size + split_size, str(entry.get("name", ""))



def order_manifest_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(entries, key=_estimate_manifest_entry_cost)



def load_hpo_manifest(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    manifest_root = path.parent
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"HPO manifest must be a JSON list: {path}")

    entries: list[dict[str, Any]] = []
    for idx, raw in enumerate(payload):
        if not isinstance(raw, dict):
            raise ValueError(f"HPO manifest entry {idx} must be an object: {path}")
        mix_csv = raw.get("mix_csv")
        split_csv = raw.get("split_csv")
        if mix_csv is None or split_csv is None:
            raise ValueError(f"HPO manifest entry {idx} must define 'mix_csv' and 'split_csv': {path}")
        mix_csv = Path(mix_csv)
        split_csv = Path(split_csv)
        if not mix_csv.is_absolute():
            mix_csv = (manifest_root / mix_csv).resolve()
        if not split_csv.is_absolute():
            split_csv = (manifest_root / split_csv).resolve()

        overrides = raw.get("overrides", {})
        if overrides is None:
            overrides = {}
        if not isinstance(overrides, dict):
            raise ValueError(f"HPO manifest entry {idx} field 'overrides' must be an object: {path}")

        config = {
            "mix_csv": mix_csv,
            "split_csv": split_csv,
            **overrides,
        }
        entries.append(
            {
                "name": str(raw.get("name") or f"entry_{idx:02d}"),
                "config": {key: _coerce_arg_value(key, value) for key, value in config.items()},
            }
        )
    return order_manifest_entries(entries)


def sanitize_metric_key(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", str(value)).strip("_")
    return cleaned or "entry"


def compute_normalization_scale(values: Any, method: str = "std") -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0

    method = str(method).strip().lower()
    if method == "std":
        scale = float(np.std(arr))
    elif method == "iqr":
        q25, q75 = np.percentile(arr, [25, 75])
        scale = float(q75 - q25)
    elif method == "mad":
        med = float(np.median(arr))
        scale = float(1.4826 * np.median(np.abs(arr - med)))
    elif method == "range":
        scale = float(np.max(arr) - np.min(arr))
    else:
        raise ValueError(f"Unsupported normalization method {method!r}. Expected one of {NORMALIZATION_METHODS}.")

    if not np.isfinite(scale) or scale <= 1e-12:
        fallback = float(np.std(arr))
        if np.isfinite(fallback) and fallback > 1e-12:
            return fallback
        return 1.0
    return scale


def aggregate_manifest_metrics(
    trial_metrics: list[tuple[str, dict[str, Any]]],
    *,
    primary_metric: str,
) -> dict[str, float]:
    if not trial_metrics:
        raise ValueError("Cannot aggregate HPO metrics from an empty manifest.")

    grouped: dict[str, list[float]] = {}
    for _, metrics in trial_metrics:
        for key, value in extract_numeric_metrics(metrics).items():
            grouped.setdefault(key, []).append(float(value))

    if primary_metric not in grouped:
        raise ValueError(f"Primary HPO metric {primary_metric!r} was not reported by manifest entries.")

    summary: dict[str, float] = {"num_manifest_entries": float(len(trial_metrics))}
    for key, values in grouped.items():
        mean_value = sum(values) / len(values)
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        summary[f"mean_{key}"] = float(mean_value)
        summary[f"std_{key}"] = float(variance**0.5)

    primary_values: list[float] = []
    raw_primary_values: list[float] = []
    for _, metrics in trial_metrics:
        numeric_metrics = extract_numeric_metrics(metrics)
        primary_value = float(numeric_metrics[primary_metric])
        scale = float(numeric_metrics.get("normalization_scale", 1.0))
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
        raw_primary_values.append(primary_value)
        primary_values.append(primary_value / scale)

    primary_mean = sum(primary_values) / len(primary_values)
    primary_var = sum((value - primary_mean) ** 2 for value in primary_values) / len(primary_values)
    raw_primary_mean = sum(raw_primary_values) / len(raw_primary_values)
    summary[primary_metric] = float(primary_mean)
    summary[f"mean_raw_{primary_metric}"] = float(raw_primary_mean)
    summary[f"std_normalized_{primary_metric}"] = float(primary_var**0.5)

    for name, metrics in trial_metrics:
        numeric_metrics = extract_numeric_metrics(metrics)
        if primary_metric in numeric_metrics:
            safe_name = sanitize_metric_key(name)
            scale = float(numeric_metrics.get("normalization_scale", 1.0))
            if not np.isfinite(scale) or scale <= 1e-12:
                scale = 1.0
            summary[f"{safe_name}__{primary_metric}"] = float(numeric_metrics[primary_metric] / scale)
            summary[f"{safe_name}__raw_{primary_metric}"] = float(numeric_metrics[primary_metric])
    return summary


def build_partial_manifest_report(
    trial_metrics: list[tuple[str, dict[str, Any]]],
    *,
    primary_metric: str,
    total_manifest_entries: int,
) -> dict[str, float]:
    summary = aggregate_manifest_metrics(trial_metrics, primary_metric=primary_metric)
    completed_entries = len(trial_metrics)
    summary["completed_manifest_entries"] = float(completed_entries)
    summary["remaining_manifest_entries"] = float(max(0, total_manifest_entries - completed_entries))
    summary["manifest_progress"] = float(completed_entries / max(1, total_manifest_entries))
    return summary


def require_ray_tune():
    try:
        import ray
        from ray import tune
        from ray.tune.schedulers import ASHAScheduler
    except Exception as exc:  # pragma: no cover - depends on optional runtime dependency
        raise RuntimeError(
            "Hyperparameter optimization requires Ray Tune. Install it in ProjectDev with "
            "`pip install 'ray[tune]'` and rerun with `--hpo`."
        ) from exc
    return ray, tune, ASHAScheduler


def materialize_search_space(space: dict[str, Any], tune_module: Any) -> dict[str, Any]:
    materialized: dict[str, Any] = {}
    for key, spec in space.items():
        if not isinstance(spec, dict) or "sampler" not in spec:
            materialized[key] = spec
            continue
        sampler = spec["sampler"]
        if sampler == "choice":
            materialized[key] = tune_module.choice(spec["values"])
        elif sampler == "loguniform":
            materialized[key] = tune_module.loguniform(spec["lower"], spec["upper"])
        elif sampler == "uniform":
            materialized[key] = tune_module.uniform(spec["lower"], spec["upper"])
        elif sampler == "randint":
            materialized[key] = tune_module.randint(spec["lower"], spec["upper"])
        else:
            raise ValueError(f"Unsupported sampler {sampler!r} for key {key!r}.")
    return materialized


def build_gnn_search_space(profile: str = "default") -> dict[str, Any]:
    if profile == "smoke":
        return {
            "batch_size": choice([4, 16]),
            "message_hidden_dim": choice([64, 96]),
            "message_passing_depth": choice([3]),
            "predictor_hidden_dim": choice([100, 150]),
            "predictor_n_layers": choice([3]),
            "aggregation": choice(["weightedsum", "attentive"]),
            "mixmp_type": choice(["none", "interaction"]),
            "max_lr": choice([5e-4, 1e-3]),
            "init_lr": 1e-4,
            "final_lr": 1e-4,
            "warmup_epochs": 1,
        }
    if profile == "gsolv100k_round2":
        return {
            "batch_size": choice([4, 16, 32, 64]),
            "message_hidden_dim": choice([64, 128, 192]),
            "message_passing_depth": choice([3, 4, 5]),
            "predictor_hidden_dim": choice([100, 200, 300]),
            "predictor_n_layers": choice([3, 4, 5]),
            "aggregation": choice(["weightedsum", "cat", "deepsets", "attentive", "set2set"]),
            "mixmp_type": choice(["none", "molecular", "interaction"]),
            "max_lr": loguniform(3e-4, 3e-3),
            "init_lr": 1e-4,
            "final_lr": 1e-4,
            "warmup_epochs": 1,
        }
    if profile != "default":
        raise ValueError(f"Unsupported GNN HPO profile {profile!r}.")
    return {
        "batch_size": choice([4, 16, 32, 64]),
        "message_hidden_dim": choice([64, 96, 128, 192]),
        "message_passing_depth": choice([3, 4, 5]),
        "predictor_hidden_dim": choice([100, 150, 200]),
        "predictor_n_layers": choice([3, 4]),
        "aggregation": choice(["weightedsum", "cat", "deepsets", "attentive", "set2set"]),
        "mixmp_type": choice(["none", "molecular", "interaction"]),
        "max_lr": loguniform(3e-4, 3e-3),
        "init_lr": 1e-4,
        "final_lr": 1e-4,
        "warmup_epochs": 1,
    }


def build_feature_search_space(predictor_head: str, profile: str = "default") -> dict[str, Any]:
    predictor_head = str(predictor_head).strip().lower()
    if predictor_head == "ffn":
        if profile == "smoke":
            return {
                "batch_size": choice([4, 16]),
                "ffn_hidden_dim": choice([100, 150]),
                "ffn_n_layers": choice([3]),
                "ffn_lr": choice([1e-4, 5e-4]),
                "aggregation": choice(["weightedsum", "attentive"]),
                "interaction_type": choice(["none", "self_attention"]),
            }
        if profile == "gsolv100k_round2":
            return {
                "batch_size": choice([4, 16, 32, 64]),
                "ffn_hidden_dim": choice([100, 200, 300, 400]),
                "ffn_n_layers": choice([3, 4, 5]),
                "ffn_lr": loguniform(1e-4, 3e-3),
                "aggregation": choice(["concat", "weightedsum", "deepsets", "attentive", "set2set"]),
                "interaction_type": choice(["none", "self_attention"]),
            }
        if profile != "default":
            raise ValueError(f"Unsupported feature-model HPO profile {profile!r} for predictor_head='ffn'.")
        return {
            "batch_size": choice([4, 16, 32, 64]),
            "ffn_hidden_dim": choice([100, 150, 200]),
            "ffn_n_layers": choice([3, 4]),
            "ffn_lr": loguniform(1e-4, 3e-3),
            "aggregation": choice(["concat", "weightedsum", "deepsets", "attentive", "set2set"]),
            "interaction_type": choice(["none", "self_attention"]),
        }

    if predictor_head == "xgboost":
        if profile == "smoke":
            return {
                "xgb_n_estimators": choice([500, 750]),
                "xgb_learning_rate": choice([0.03, 0.1]),
                "xgb_max_depth": choice([4, 6]),
                "xgb_subsample": choice([0.8, 1.0]),
                "xgb_colsample_bytree": choice([0.8, 1.0]),
                "xgb_reg_lambda": choice([0.5, 1.0]),
                "aggregation": choice(["concat", "weightedsum"]),
            }
        if profile == "gsolv100k_round2":
            return {
                "xgb_n_estimators": choice([500, 1000, 2000, 4000]),
                "xgb_learning_rate": loguniform(1e-2, 2e-1),
                "xgb_max_depth": choice([3, 5, 7]),
                "xgb_subsample": choice([0.8, 1.0]),
                "xgb_colsample_bytree": choice([0.8, 1.0]),
                "xgb_reg_lambda": choice([0.5, 1.0, 2.0]),
                "aggregation": choice(["concat", "weightedsum"]),
            }
        if profile != "default":
            raise ValueError(f"Unsupported feature-model HPO profile {profile!r} for predictor_head='xgboost'.")
        return {
            "xgb_n_estimators": choice([500, 750, 1000]),
            "xgb_learning_rate": loguniform(1e-2, 2e-1),
            "xgb_max_depth": choice([3, 4, 5]),
            "xgb_subsample": choice([0.8, 1.0]),
            "xgb_colsample_bytree": choice([0.8, 1.0]),
            "xgb_reg_lambda": choice([0.5, 1.0, 2.0]),
            "aggregation": choice(["concat", "weightedsum"]),
        }

    raise ValueError(f"Unsupported predictor_head={predictor_head!r}; expected 'ffn' or 'xgboost'.")


def extract_numeric_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    extracted: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            extracted[key] = float(value)
    return extracted


def write_hpo_summary(
    path: Path,
    *,
    best_config: dict[str, Any],
    best_metrics: dict[str, Any],
    search_space: dict[str, Any],
    metric: str,
    mode: str,
    num_samples: int,
) -> None:
    payload = {
        "metric": metric,
        "mode": mode,
        "num_samples": int(num_samples),
        "best_config": best_config,
        "best_metrics": extract_numeric_metrics(best_metrics),
        "search_space": search_space,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
