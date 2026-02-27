from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass
class XGBoostConfig:
    n_estimators: int = 2000
    learning_rate: float = 0.03
    max_depth: int = 6
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_lambda: float = 1.0
    random_state: int = 0


class XGBoostRegressor:
    """Thin wrapper around xgboost.XGBRegressor for descriptor-head training."""

    def __init__(self, config: XGBoostConfig):
        try:
            from xgboost import XGBRegressor
        except Exception as e:  # pragma: no cover - depends on local env
            raise RuntimeError(
                "XGBoostRegressor requires the 'xgboost' package. "
                "Install it with `pip install xgboost`."
            ) from e

        self.config = config
        self.model = XGBRegressor(
            n_estimators=config.n_estimators,
            learning_rate=config.learning_rate,
            max_depth=config.max_depth,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            reg_lambda=config.reg_lambda,
            objective="reg:squarederror",
            random_state=config.random_state,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        early_stopping_rounds: int | None = None,
        verbose: bool = False,
    ) -> None:
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val.reshape(-1))]

        fit_kwargs = {"verbose": verbose}
        if eval_set is not None:
            fit_kwargs["eval_set"] = eval_set
        if early_stopping_rounds is not None:
            fit_kwargs["early_stopping_rounds"] = early_stopping_rounds

        try:
            self.model.fit(X_train, y_train.reshape(-1), **fit_kwargs)
        except TypeError:
            # Backward compatibility across xgboost versions with slightly different fit kwargs.
            fit_kwargs.pop("early_stopping_rounds", None)
            self.model.fit(X_train, y_train.reshape(-1), **fit_kwargs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def save_model(self, path: str | Path) -> None:
        self.model.save_model(str(path))

    def save_metadata(
        self,
        path: str | Path,
        *,
        descriptor_featurizer: str,
        scaler: dict[str, object],
    ) -> None:
        payload = {
            "descriptor_only": True,
            "predictor_head": "xgboost",
            "descriptor_featurizer": descriptor_featurizer,
            "scaler": scaler,
            "xgboost_params": self.model.get_params(),
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

