import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from chemprop.nn.xgboost import XGBoostConfig, XGBoostRegressor, XGBoostScaler


class FakeXGBRegressor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fit_calls: list[dict[str, object]] = []

    def fit(self, X, y, **kwargs):
        self.fit_calls.append(
            {
                "X_shape": np.asarray(X).shape,
                "y_shape": np.asarray(y).shape,
                **kwargs,
            }
        )

    def predict(self, X):
        return np.zeros(np.asarray(X).shape[0], dtype=float)

    def save_model(self, path):
        Path(path).write_text("fake-xgb", encoding="utf-8")

    def get_params(self):
        return dict(self.kwargs)


def test_xgboost_wrapper_fit_and_metadata(tmp_path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "xgboost", SimpleNamespace(XGBRegressor=FakeXGBRegressor))

    model = XGBoostRegressor(XGBoostConfig(max_depth=4, random_state=7))
    X_train = np.ones((3, 2), dtype=float)
    y_train = np.arange(3, dtype=float).reshape(-1, 1)
    X_val = np.zeros((2, 2), dtype=float)
    y_val = np.zeros((2, 1), dtype=float)

    model.fit(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        early_stopping_rounds=11,
        verbose=False,
    )

    fit_call = model.model.fit_calls[0]
    assert fit_call["X_shape"] == (3, 2)
    assert fit_call["y_shape"] == (3,)
    assert fit_call["early_stopping_rounds"] == 11
    assert fit_call["eval_set"] == [(X_val, y_val.reshape(-1))]

    model_path = tmp_path / "model.xgb.json"
    metadata_path = tmp_path / "model.scaler.json"
    model.save_model(model_path)
    model.save_metadata(
        metadata_path,
        descriptor_featurizer="rdkit2dnormalized",
        scaler=XGBoostScaler(
            y_mean=1.5,
            y_std=0.5,
            x_mean=np.array([[1.0, 2.0]]),
            x_std=np.array([[3.0, 4.0]]),
        ),
        aggregation="concat",
        interaction_type="none",
    )

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert model_path.read_text(encoding="utf-8") == "fake-xgb"
    assert payload["predictor_head"] == "xgboost"
    assert payload["descriptor_featurizer"] == "rdkit2dnormalized"
    assert payload["aggregation"] == "concat"
    assert payload["interaction_type"] == "none"
    assert payload["scaler"]["x_mean"] == [[1.0, 2.0]]
    assert payload["scaler"]["x_std"] == [[3.0, 4.0]]
    assert payload["xgboost_params"]["max_depth"] == 4
