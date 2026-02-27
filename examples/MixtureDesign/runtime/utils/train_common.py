from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem


def inchi_to_smiles(inchi: object) -> str | None:
    if pd.isna(inchi):
        return None
    mol = Chem.MolFromInchi(str(inchi))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def gather_component_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    comp_cols = sorted(
        [c for c in df.columns if c.startswith("component_inchi_")],
        key=lambda c: int(c.rsplit("_", 1)[1]),
    )
    frac_cols = [f"fraction_{c.rsplit('_', 1)[1]}" for c in comp_cols]
    return comp_cols, frac_cols


def compact_components(
    df_mix: pd.DataFrame,
    max_components: int | None = 5,
    min_fraction: float = 1e-4,
    preserve_solute_at_index0: bool = False,
) -> pd.DataFrame:
    comp_cols, frac_cols = gather_component_columns(df_mix)
    if max_components is None:
        max_components = len(comp_cols)
    keep_cols = [c for c in df_mix.columns if c not in comp_cols and c not in frac_cols]
    rows = []
    for _, row in df_mix.iterrows():
        pairs = []
        for ccol, fcol in zip(comp_cols, frac_cols):
            inchi = row.get(ccol)
            frac = pd.to_numeric(row.get(fcol), errors="coerce")
            if pd.isna(inchi) or pd.isna(frac):
                continue
            frac = float(frac)
            if frac <= 0.0 or frac < min_fraction:
                continue
            pairs.append((str(inchi), frac))

        total_before_truncation = sum(f for _, f in pairs)
        if pairs and not np.isclose(total_before_truncation, 1.0, atol=1e-3, rtol=0.0):
            raise ValueError(
                f"Fractions do not sum to 1 (sum={total_before_truncation:.6f}) for row index {_}."
            )

        pairs.sort(key=lambda x: x[1], reverse=True)
        if preserve_solute_at_index0 and "solute_inchi" in row.index and pairs:
            solute = row.get("solute_inchi")
            if not pd.isna(solute):
                solute = str(solute)
                solute_pair = None
                rest_pairs = []
                for inchi, frac in pairs:
                    if solute_pair is None and inchi == solute:
                        solute_pair = (inchi, frac)
                    else:
                        rest_pairs.append((inchi, frac))
                if solute_pair is not None:
                    pairs = [solute_pair] + rest_pairs
        # Preserve original fractions (no renormalization); only reorder/truncate components.
        pairs = pairs[:max_components]

        out = {k: row[k] for k in keep_cols}
        for i in range(max_components):
            if i < len(pairs):
                out[f"component_inchi_{i}"] = pairs[i][0]
                out[f"fraction_{i}"] = pairs[i][1]
            else:
                out[f"component_inchi_{i}"] = np.nan
                out[f"fraction_{i}"] = 0.0
        rows.append(out)
    return pd.DataFrame(rows)


def _override_fraction_columns(
    df: pd.DataFrame,
    *,
    fraction_basis: str,
) -> pd.DataFrame:
    if fraction_basis == "mole":
        return df
    if fraction_basis != "surface_area":
        raise ValueError(f"Unsupported fraction_basis={fraction_basis!r}; expected 'mole' or 'surface_area'.")

    df = df.copy()
    comp_cols, frac_cols = gather_component_columns(df)
    for ccol, fcol in zip(comp_cols, frac_cols):
        idx = ccol.rsplit("_", 1)[1]
        sa_col = f"surface_area_fraction_{idx}"
        if sa_col not in df.columns:
            raise KeyError(f"Surface-area fraction column '{sa_col}' missing for fraction_basis='surface_area'.")
        df[fcol] = df[sa_col]
    return df


def build_descriptor_matrix(
    df_mix: pd.DataFrame,
    descriptor_featurizer: str = "rdkit2dnormalized",
    include_fraction_features: bool = True,
    component_combine: str = "concat",
) -> np.ndarray:
    # Lazy import so metric helpers in this module stay usable without full Chemprop import stack.
    from chemprop.featurizers.molecule import MoleculeFeaturizerRegistry

    comp_cols, frac_cols = gather_component_columns(df_mix)
    if component_combine not in {"concat", "weighted_sum"}:
        raise ValueError(f"Unsupported component_combine={component_combine!r}; expected 'concat' or 'weighted_sum'.")
    featurizer = MoleculeFeaturizerRegistry[descriptor_featurizer]()
    d_desc = len(featurizer)
    d_frac = len(frac_cols) if include_fraction_features else 0
    d_components = len(comp_cols) * d_desc if component_combine == "concat" else d_desc
    X = np.zeros((len(df_mix), d_components + d_frac), dtype=float)
    zero_desc = np.zeros(d_desc, dtype=float)
    desc_cache: dict[str, np.ndarray] = {}

    for row_idx, row in enumerate(df_mix.itertuples(index=False)):
        row_dict = row._asdict()
        feats = []
        fracs = []
        for comp_col, frac_col in zip(comp_cols, frac_cols):
            frac_val = pd.to_numeric(row_dict.get(frac_col), errors="coerce")
            frac = 0.0 if pd.isna(frac_val) else float(frac_val)
            fracs.append(frac)
            inchi = row_dict.get(comp_col)

            if pd.isna(inchi) or frac <= 0.0:
                feats.append(zero_desc)
                continue

            key = str(inchi)
            if key not in desc_cache:
                smi = inchi_to_smiles(key)
                if smi is None:
                    desc_cache[key] = zero_desc
                else:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is None:
                        desc_cache[key] = zero_desc
                    else:
                        d = np.array(featurizer(mol), dtype=float)
                        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
                        desc_cache[key] = d
            feats.append(desc_cache[key])

        if component_combine == "concat":
            row_feat = np.concatenate(feats, axis=0) if feats else np.zeros(0, dtype=float)
        else:
            # Fraction-weighted sum over component descriptor vectors.
            if feats:
                feat_mat = np.stack(feats, axis=0)
                frac_vec = np.array(fracs, dtype=float).reshape(-1, 1)
                row_feat = (feat_mat * frac_vec).sum(axis=0)
            else:
                row_feat = np.zeros(d_desc, dtype=float)
        if include_fraction_features:
            row_feat = np.concatenate([row_feat, np.array(fracs, dtype=float)], axis=0)
        X[row_idx] = row_feat

    return X


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(float).reshape(-1)
    y_pred = y_pred.astype(float).reshape(-1)
    err = y_pred - y_true
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1]) if y_true.size > 1 else float("nan")
    return {"mse": mse, "rmse": rmse, "mae": mae, "pearson_r": pearson_r}


def _mix_like_from_pure(
    pure_df: pd.DataFrame,
    mix_columns: list[str],
    max_components: int,
) -> pd.DataFrame:
    out = pd.DataFrame(index=pure_df.index)

    # Preserve all non-component/fraction columns that exist in the mixture table shape.
    for c in mix_columns:
        if c.startswith("component_inchi_") or c.startswith("fraction_"):
            continue
        out[c] = pure_df[c] if c in pure_df.columns else np.nan

    if "fraction_type" in mix_columns and "fraction_type" not in out.columns:
        out["fraction_type"] = "molar"

    # Represent pure rows as single-component mixtures.
    out["component_inchi_0"] = pure_df["component_inchi_0"] if "component_inchi_0" in pure_df.columns else np.nan
    out["fraction_0"] = 1.0
    for i in range(1, max_components):
        out[f"component_inchi_{i}"] = np.nan
        out[f"fraction_{i}"] = 0.0

    # Match mixture column order where possible.
    ordered = [c for c in mix_columns if c in out.columns]
    extras = [c for c in out.columns if c not in ordered]
    return out[ordered + extras]


def load_split_dataframes(
    mix_csv: Path,
    split_csv: Path,
    max_components: int | None,
    min_fraction: float,
    fraction_basis: str = "mole",
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mix_csv = Path(mix_csv)
    split_csv = Path(split_csv)

    mix_raw = pd.read_csv(mix_csv)
    mix_raw = _override_fraction_columns(mix_raw, fraction_basis=fraction_basis)
    mix_raw = mix_raw.reset_index().rename(columns={"index": "_source_row_index"})
    actual_max_components = max_components
    if actual_max_components is None:
        actual_max_components = len([c for c in mix_raw.columns if c.startswith("component_inchi_")])
    preserve_solute_at_index0 = "solute_inchi" in mix_raw.columns
    mix_df = compact_components(
        mix_raw,
        max_components=actual_max_components,
        min_fraction=min_fraction,
        preserve_solute_at_index0=preserve_solute_at_index0,
    ).reset_index(drop=True)
    if "_source_row_index" not in mix_df.columns:
        raise ValueError("Internal error: _source_row_index missing after compact_components() for mixture data")
    mix_lookup = {int(rid): pos for pos, rid in enumerate(mix_df["_source_row_index"].tolist())}

    split_df = pd.read_csv(split_csv)
    split_df = split_df[split_df["split"].isin(["train", "val", "test"])].copy()

    pure_csv = mix_csv.with_name(mix_csv.name.replace("_mix_", "_pure_"))
    pure_df = None
    if pure_csv.exists() and (split_df["source"] == "pure").any():
        pure_raw = pd.read_csv(pure_csv).reset_index().rename(columns={"index": "_source_row_index"})
        pure_raw = _override_fraction_columns(pure_raw, fraction_basis=fraction_basis)
        pure_df = _mix_like_from_pure(pure_raw, list(mix_df.columns), max_components=actual_max_components).reset_index(drop=True)
        if "_source_row_index" not in pure_df.columns:
            raise ValueError("Internal error: _source_row_index missing after _mix_like_from_pure()")
        pure_lookup = {int(rid): pos for pos, rid in enumerate(pure_df["_source_row_index"].tolist())}
    else:
        pure_lookup = {}

    def _rows_for(split_name: str) -> pd.DataFrame:
        take = split_df[split_df["split"] == split_name]
        parts: list[pd.DataFrame] = []
        for _, r in take.iterrows():
            src = r.get("source")
            idx = int(r.get("source_row_index"))
            if src == "mix":
                pos = mix_lookup.get(idx)
                if pos is None:
                    raise IndexError(f"Mixture source_row_index {idx} not found in {mix_csv}")
                parts.append(mix_df.iloc[[pos]].copy())
            elif src == "pure":
                if pure_df is None:
                    raise ValueError(f"Split references pure rows but pure CSV was not available: {pure_csv}")
                pos = pure_lookup.get(idx)
                if pos is None:
                    raise IndexError(f"Pure source_row_index {idx} not found in {pure_csv}")
                parts.append(pure_df.iloc[[pos]].copy())
        if not parts:
            return mix_df.iloc[0:0].copy()
        return pd.concat(parts, axis=0, ignore_index=True)

    train_df = _rows_for("train")
    val_df = _rows_for("val")
    test_df = _rows_for("test")

    # Keep training robust for folds with empty validation by carving from train only.
    if len(val_df) == 0 and len(train_df) > 1:
        rng = np.random.default_rng(seed)
        n_val = max(1, int(round(0.1 * len(train_df))))
        n_val = min(n_val, len(train_df) - 1)
        picked = sorted(rng.choice(np.arange(len(train_df)), size=n_val, replace=False).tolist())
        val_df = train_df.iloc[picked].copy().reset_index(drop=True)
        keep = np.ones(len(train_df), dtype=bool)
        keep[picked] = False
        train_df = train_df.iloc[keep].copy().reset_index(drop=True)
        print(f"[split-fallback] Validation split was empty; moved {len(val_df)} training rows to val.")

    if len(train_df) == 0:
        raise ValueError(f"No training rows available in split file: {split_csv}")
    if len(test_df) == 0:
        raise ValueError(f"No test rows available in split file: {split_csv}")

    return train_df, val_df, test_df


def infer_output_dir(outputs_root: Path, mix_csv: Path, split_csv: Path) -> Path:
    dataset_name = mix_csv.stem
    split_name = split_csv.stem
    fold_name = split_csv.stem

    parts = list(split_csv.parts)
    if "split_definitions" in parts:
        i = parts.index("split_definitions")
        if i + 1 < len(parts):
            split_name = parts[i + 1]
    if split_csv.suffix.lower() == ".csv":
        fold_name = split_csv.stem

    return outputs_root / dataset_name / split_name / fold_name


def save_parity_plot(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str) -> None:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=18, alpha=0.7)
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
