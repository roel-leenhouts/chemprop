#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader

from model_core.data.datapoints import MoleculeDatapoint
from model_core.featurizers import get_multi_hot_atom_featurizer
from model_core.featurizers.molgraph import SimpleMoleculeMolGraphFeaturizer
from model_core.nn.agg import MeanAggregation
from model_core.nn.message_passing import BondMessagePassing
from model_core.nn.predictors import RegressionFFN
from model_core.nn.transforms import UnscaleTransform

from utils.mixtures import (
    
    ComponentDatapoint,
    ComponentDataset,
    MixtureDatapoint,
    MixtureDataset,
    MixtureGraphDataset,
    MixtureMPNN,
    MolecularMessagePassing,
    MulticomponentMessagePassing,
    InteractionMessagePassing,
    SimpleMixtureGraphFeaturizer,
    WeightedSumAggregation,
    DeepsetsAggregation,
    AttentiveAggregation,
    Set2SetAggregation,
    ConcatAggregation,
    PhysicsRegressionFFN,
    collate_mixture,
)
from utils.hpo import (
    aggregate_manifest_metrics,
    build_partial_manifest_report,
    apply_trial_config,
    build_gnn_search_space,
    clone_namespace,
    compute_normalization_scale,
    extract_numeric_metrics,
    load_hpo_manifest,
    NORMALIZATION_METHODS,
    materialize_search_space,
    require_ray_tune,
    write_hpo_summary,
)
from utils.train_common import (
    compact_components,
    gather_component_columns,
    inchi_to_smiles,
    infer_output_dir,
    load_split_dataframes,
    regression_metrics,
    save_parity_plot,
)

class LossHistoryRecorder(Callback):
    """Record epoch-level train/val losses emitted by Lightning."""

    def __init__(self) -> None:
        self.rows: list[dict[str, float | int]] = []

    def on_validation_epoch_end(self, trainer, pl_module) -> None:  # noqa: ARG002
        metrics = trainer.callback_metrics
        train_loss = metrics.get("train_loss_epoch")
        val_loss = metrics.get("val_loss")
        val_selection_score = metrics.get("val_selection_score")
        val_excess_loss = metrics.get("val_excess_loss")
        val_excess_mae = metrics.get("val_excess_mae")
        row = {"epoch": int(trainer.current_epoch)}
        if train_loss is not None:
            row["train_loss_epoch"] = float(train_loss.detach().cpu())
        if val_loss is not None:
            row["val_loss"] = float(val_loss.detach().cpu())
        if val_selection_score is not None:
            row["val_selection_score"] = float(val_selection_score.detach().cpu())
        if val_excess_loss is not None:
            row["val_excess_loss"] = float(val_excess_loss.detach().cpu())
        if val_excess_mae is not None:
            row["val_excess_mae"] = float(val_excess_mae.detach().cpu())
        self.rows.append(row)


def take_rows(all_data: list[list], row_ids: list[int]) -> list[list]:
    return [[component_list[i] for i in row_ids] for component_list in all_data]


def _has_explicit_solute_context(df_mix: pd.DataFrame, comp_cols: list[str]) -> bool:
    if "solute_inchi" not in df_mix.columns:
        return False
    if not comp_cols:
        return True
    n_check = min(len(df_mix), 5000)
    if n_check == 0:
        return False
    s = df_mix["solute_inchi"].iloc[:n_check].astype(str)
    matches = pd.Series(False, index=s.index)
    for c in comp_cols:
        matches |= s.eq(df_mix[c].iloc[:n_check].astype(str))
    return float(matches.mean()) < 0.5


def build_all_data(
    df_mix: pd.DataFrame,
    target_col: str = "value",
    solute_component_index: int = -1,
) -> list[list]:
    comp_cols, frac_cols = gather_component_columns(df_mix)
    use_explicit_solute = solute_component_index == 0 and _has_explicit_solute_context(df_mix, comp_cols)
    df_smi = df_mix.copy()
    for c in comp_cols:
        df_smi[c] = df_smi[c].apply(inchi_to_smiles)
    if use_explicit_solute:
        df_smi["solute_inchi"] = df_smi["solute_inchi"].apply(inchi_to_smiles)

    ys = df_mix[[target_col]].values
    all_data: list[list] = []

    component_defs: list[tuple[str, str | None]] = []
    if use_explicit_solute:
        component_defs.append(("solute_inchi", None))
    component_defs.extend((comp_col, frac_col) for comp_col, frac_col in zip(comp_cols, frac_cols))

    for i, (comp_col, frac_col) in enumerate(component_defs):
        is_solute = i == solute_component_index
        comp_entries = []
        if frac_col is None:
            fracs = [1.0] * len(df_mix)
        else:
            fracs = df_mix[frac_col].fillna(0.0).tolist()
        for smi, y, frac in zip(df_smi[comp_col].tolist(), ys, fracs):
            w = 1.0 if is_solute else float(frac)
            if smi is None or (not is_solute and w <= 0.0):
                comp_entries.append(ComponentDatapoint(None))
                continue
            if i == 0:
                comp_entries.append(MoleculeDatapoint.from_smi(smi, y))
            else:
                comp_entries.append(ComponentDatapoint.from_smi(smi, w_fp=w))
        all_data.append(comp_entries)

    first_col, first_frac_col = component_defs[0]
    if first_frac_col is None:
        first_fracs = [1.0] * len(df_mix)
    else:
        first_fracs = df_mix[first_frac_col].fillna(0.0).tolist()
    first_fixed = []
    for smi, y, frac in zip(df_smi[first_col].tolist(), ys, first_fracs):
        is_solute = solute_component_index == 0
        w = 1.0 if is_solute else float(frac)
        if smi is None or (not is_solute and w <= 0.0):
            first_fixed.append(ComponentDatapoint(None))
        else:
            first_fixed.append(ComponentDatapoint.from_smi(smi, y, w_fp=w))
    all_data[0] = first_fixed

    smiss_cols = [c for c in comp_cols]
    if use_explicit_solute:
        smiss_cols = ["solute_inchi"] + smiss_cols
    smiss = df_smi[smiss_cols].values.tolist()
    smiss = [[s if isinstance(s, str) and s.strip() else None for s in row] for row in smiss]
    all_data.append([MixtureDatapoint.from_smis(smi_row) for smi_row in smiss])
    return all_data


def build_all_data_with_xd(
    df_mix: pd.DataFrame,
    x_d: np.ndarray,
    target_col: str = "value",
    solute_component_index: int = -1,
) -> list[list]:
    if x_d.shape[0] != len(df_mix):
        raise ValueError(f"x_d rows ({x_d.shape[0]}) must match df rows ({len(df_mix)})")

    comp_cols, frac_cols = gather_component_columns(df_mix)
    use_explicit_solute = solute_component_index == 0 and _has_explicit_solute_context(df_mix, comp_cols)
    df_smi = df_mix.copy()
    for c in comp_cols:
        df_smi[c] = df_smi[c].apply(inchi_to_smiles)
    if use_explicit_solute:
        df_smi["solute_inchi"] = df_smi["solute_inchi"].apply(inchi_to_smiles)

    ys = df_mix[[target_col]].values
    all_data: list[list] = []
    component_defs: list[tuple[str, str | None]] = []
    if use_explicit_solute:
        component_defs.append(("solute_inchi", None))
    component_defs.extend((comp_col, frac_col) for comp_col, frac_col in zip(comp_cols, frac_cols))

    for i, (comp_col, frac_col) in enumerate(component_defs):
        is_solute = i == solute_component_index
        comp_entries = []
        if frac_col is None:
            fracs = [1.0] * len(df_mix)
        else:
            fracs = df_mix[frac_col].fillna(0.0).tolist()
        for row_idx, (smi, y, frac) in enumerate(zip(df_smi[comp_col].tolist(), ys, fracs)):
            w = 1.0 if is_solute else float(frac)
            if smi is None or (not is_solute and w <= 0.0):
                comp_entries.append(ComponentDatapoint(None))
                continue
            if i == 0:
                comp_entries.append(
                    ComponentDatapoint.from_smi(smi, y, w_fp=w, x_d=np.array(x_d[row_idx], dtype=float, copy=True))
                )
            else:
                comp_entries.append(ComponentDatapoint.from_smi(smi, w_fp=w))
        all_data.append(comp_entries)

    smiss_cols = [c for c in comp_cols]
    if use_explicit_solute:
        smiss_cols = ["solute_inchi"] + smiss_cols
    smiss = df_smi[smiss_cols].values.tolist()
    smiss = [[s if isinstance(s, str) and s.strip() else None for s in row] for row in smiss]
    all_data.append([MixtureDatapoint.from_smis(smi_row) for smi_row in smiss])
    return all_data


def build_mixture_dataset(
    split_data: list[list],
    n_components: int,
    use_mixmp: bool,
    atom_featurizer_mode: str = "ORGANIC",
    component_featurizer: SimpleMoleculeMolGraphFeaturizer | None = None,
) -> MixtureDataset:
    if component_featurizer is None:
        component_featurizer = SimpleMoleculeMolGraphFeaturizer(
            atom_featurizer=get_multi_hot_atom_featurizer(atom_featurizer_mode)
        )
    datasets = [ComponentDataset(split_data[i], featurizer=component_featurizer) for i in range(n_components)]
    if use_mixmp:
        datasets.append(MixtureGraphDataset(split_data[-1]))
    return MixtureDataset(datasets)


def build_model(
    n_components: int,
    scaler,
    aggregation: str = "weightedsum",
    solute_component_index: int = -1,
    legacy_componentwise_mp: bool = False,
    mixmp_type: str = "none",
    x_d_dim: int = 0,
    temperature_mode: str = "none",
    temperature_physics_law: str = "arrhenius",
    atom_fdim: int | None = None,
    bond_fdim: int | None = None,
    message_hidden_dim: int = 101,
    message_passing_depth: int = 4,
    predictor_hidden_dim: int | None = None,
    predictor_n_layers: int = 4,
    warmup_epochs: int = 2,
    init_lr: float = 1e-4,
    max_lr: float = 1e-3,
    final_lr: float = 1e-4,
    predictor_x_d_dim: int = 0,
) -> MixtureMPNN:
    mp_kwargs = {"depth": message_passing_depth, "d_h": message_hidden_dim, "activation": "leakyrelu"}
    if atom_fdim is not None:
        mp_kwargs["d_v"] = atom_fdim
    if bond_fdim is not None:
        mp_kwargs["d_e"] = bond_fdim
    if legacy_componentwise_mp:
        mp_groups = [[i] for i in range(n_components)]
        agg_groups = [list(range(n_components))]
        blocks = [BondMessagePassing(**mp_kwargs)] * n_components
    elif solute_component_index >= 0:
        if solute_component_index >= n_components:
            raise ValueError(f"solute_component_index={solute_component_index} out of bounds for n_components={n_components}")
        solvent_group = [i for i in range(n_components) if i != solute_component_index]
        mp_groups = [[solute_component_index], solvent_group]
        agg_groups = mp_groups
        blocks = [BondMessagePassing(**mp_kwargs)] * 2
    else:
        mp_groups = [list(range(n_components))]
        agg_groups = mp_groups
        blocks = [BondMessagePassing(**mp_kwargs)]

    mcmp = MulticomponentMessagePassing(blocks=blocks, groups=mp_groups, shared=False)
    if mixmp_type == "interaction":
        default_mol_fdim, default_interaction_fdim = SimpleMixtureGraphFeaturizer().shape
        mixmp = InteractionMessagePassing(
            depth=3,
            d_v=mcmp.blocks[0].output_dim + default_mol_fdim,
            d_h=mcmp.blocks[0].output_dim,
            d_e=default_interaction_fdim,
            activation="leakyrelu",
        )
    elif mixmp_type == "molecular":
        _, default_interaction_fdim = SimpleMixtureGraphFeaturizer().shape
        mixmp = MolecularMessagePassing(
            depth=3,
            d_v=mcmp.blocks[0].output_dim,
            d_e=default_interaction_fdim,
            d_h=mcmp.blocks[0].output_dim,
            activation="leakyrelu",
        )
    elif mixmp_type == "none":
        mixmp = None
    else:
        raise ValueError(f"Unknown mixmp_type={mixmp_type!r}")

    graph_agg = MeanAggregation()
    fp_dims = [mcmp.blocks[0].output_dim] * n_components
    if aggregation == "weightedsum":
        mixagg = WeightedSumAggregation(graph_agg=graph_agg, groups=agg_groups, fp_dims=fp_dims, mixmp=mixmp, debug=False)
    elif aggregation == "cat":
        mixagg = ConcatAggregation(graph_agg=graph_agg, groups=agg_groups, fp_dims=fp_dims, mixmp=mixmp)
    elif aggregation == "deepsets":
        mixagg = DeepsetsAggregation(graph_agg=graph_agg, groups=agg_groups, fp_dims=fp_dims, mixmp=mixmp)
    elif aggregation == "attentive":
        mixagg = AttentiveAggregation(graph_agg=graph_agg, groups=agg_groups, fp_dims=fp_dims, mixmp=mixmp)
    elif aggregation == "set2set":
        mixagg = Set2SetAggregation(graph_agg=graph_agg, groups=agg_groups, fp_dims=fp_dims, mixmp=mixmp)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    ffn_hidden_dim = predictor_hidden_dim if predictor_hidden_dim is not None else 200 * (2 if solute_component_index >= 0 else 1)
    output_transform = UnscaleTransform.from_standard_scaler(scaler)

    if temperature_mode == "physics":
        predictor = PhysicsRegressionFFN(
            law=temperature_physics_law,
            input_dim=mixagg.output_dim,
            output_transform=output_transform,
            hidden_dim=ffn_hidden_dim,
            n_layers=predictor_n_layers,
            activation="leakyrelu",
        )
    else:
        predictor = RegressionFFN(
            n_tasks=1,
            input_dim=mixagg.output_dim + (predictor_x_d_dim if temperature_mode == "concat" else 0),
            output_transform=output_transform,
            hidden_dim=ffn_hidden_dim,
            n_layers=predictor_n_layers,
            activation="leakyrelu",
        )

    return MixtureMPNN(
        message_passing=mcmp,
        agg=mixagg,
        predictor=predictor,
        temperature_mode=temperature_mode,
        predictor_x_d_dim=predictor_x_d_dim,
        warmup_epochs=warmup_epochs,
        init_lr=init_lr,
        max_lr=max_lr,
        final_lr=final_lr,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--mix-csv", type=Path, default=repo_root / "datasets/fuel_ignition_numbers/processed_data/dcn_mix_exp.csv")
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=repo_root / "splits/fuel_ignition_numbers/dcn_mix_exp/split_definitions/molecule/cross_validation_10fold/fold_00.csv",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--accelerator", type=str, default="cpu")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--aggregation", type=str, default="weightedsum")
    parser.add_argument(
        "--fraction-basis",
        type=str,
        default="surface_area",
        help="Compatibility flag for shared Slurm launchers; the GNN currently reads fractions directly from the split CSV.",
    )
    parser.add_argument(
        "--use-temperature-feature",
        dest="use_temperature_feature",
        action="store_true",
        help="Backward-compatible alias for --temperature-mode concat.",
    )
    parser.add_argument(
        "--no-temperature-feature",
        dest="use_temperature_feature",
        action="store_false",
        help="Backward-compatible alias for --temperature-mode none.",
    )
    parser.set_defaults(use_temperature_feature=None)
    parser.add_argument(
        "--temperature-mode",
        type=str,
        choices=["none", "concat", "physics"],
        default="none",
        help="How temperature is incorporated into the mixture model.",
    )
    parser.add_argument("--temperature-col", type=str, default="temperature")
    parser.add_argument(
        "--temperature-physics-law",
        type=str,
        choices=["arrhenius", "gibbs", "vant_hoff"],
        default="arrhenius",
        help="Physics law used when --temperature-mode physics.",
    )
    parser.add_argument(
        "--atom-featurizer-mode",
        type=str,
        choices=["V1", "V2", "ORGANIC", "RIGR"],
        default="ORGANIC",
        help="Project multi-hot atom featurizer mode for GNN molecular graphs.",
    )
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-components", type=int, default=None, help="Maximum number of components to keep (default: all available).")
    parser.add_argument("--min-fraction", type=float, default=1e-4)
    parser.add_argument(
        "--use-mixmp",
        dest="use_mixmp",
        action="store_true",
        help="Enable mixture interaction message passing (disabled by default).",
    )
    parser.set_defaults(use_mixmp=False)
    parser.add_argument(
        "--mixmp-type",
        type=str,
        choices=["none", "molecular", "interaction"],
        default="none",
        help="Mixture-level interaction/message-passing type.",
    )
    parser.add_argument("--solute-component-index", type=int, default=-1)
    parser.add_argument("--outputs-root", type=Path, default=repo_root / "scripts_training/outputs")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--metrics-out", type=Path, default=None)
    parser.add_argument("--parity-plot-out", type=Path, default=None)
    parser.add_argument("--val-parity-plot-out", type=Path, default=None)
    parser.add_argument("--loss-history-out", type=Path, default=None)
    parser.add_argument("--message-hidden-dim", type=int, default=101)
    parser.add_argument("--message-passing-depth", type=int, default=4)
    parser.add_argument("--predictor-hidden-dim", type=int, default=None)
    parser.add_argument("--predictor-n-layers", type=int, default=4)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--init-lr", type=float, default=1e-4)
    parser.add_argument("--max-lr", type=float, default=1e-3)
    parser.add_argument("--final-lr", type=float, default=1e-4)
    parser.add_argument("--hpo", action="store_true", help="Run Ray Tune hyperparameter optimization instead of a single training run.")
    parser.add_argument("--hpo-search-space-profile", type=str, choices=["default", "smoke"], default="default")
    parser.add_argument("--hpo-num-samples", type=int, default=10)
    parser.add_argument("--hpo-metric", type=str, default="val_rmse")
    parser.add_argument("--hpo-mode", type=str, choices=["min", "max"], default="min")
    parser.add_argument(
        "--hpo-normalization-method",
        type=str,
        choices=list(NORMALIZATION_METHODS),
        default="std",
        help="How to scale RMSE across manifest entries when different datasets/properties have different magnitudes.",
    )
    parser.add_argument("--hpo-cpus-per-trial", type=float, default=1.0)
    parser.add_argument("--hpo-gpus-per-trial", type=float, default=0.0)
    parser.add_argument("--hpo-max-concurrent-trials", type=int, default=None)
    parser.add_argument("--hpo-grace-period", type=int, default=5)
    parser.add_argument("--hpo-reduction-factor", type=int, default=2)
    parser.add_argument("--hpo-results-dir", type=Path, default=None)
    parser.add_argument(
        "--hpo-resume",
        action="store_true",
        help="Resume an existing Ray Tune experiment from the same results directory if available.",
    )
    parser.add_argument(
        "--hpo-manifest",
        type=Path,
        default=None,
        help="Optional JSON manifest describing multiple mix/split entries to score within each HPO trial.",
    )
    parser.add_argument(
        "--hpo-fixed-aggregation",
        type=str,
        choices=["weightedsum", "cat", "deepsets", "attentive", "set2set"],
        default=None,
        help="Optional fixed aggregation used for every HPO trial; removes aggregation from the search space.",
    )
    parser.add_argument(
        "--hpo-fixed-mixmp-type",
        type=str,
        choices=["none", "molecular", "interaction"],
        default=None,
        help="Optional fixed mixmp_type used for every HPO trial; removes mixmp_type from the search space.",
    )
    parser.add_argument("--agg-debug", action="store_true", help="Enable weighted-sum aggregation debug prints.")
    return parser


def _split_name_from_path(split_csv: Path) -> str:
    parts = list(Path(split_csv).parts)
    if "split_definitions" in parts:
        i = parts.index("split_definitions")
        if i + 1 < len(parts):
            return parts[i + 1]
    return ""


def _standardize_from_train(values: np.ndarray, train_ids: list[int]) -> np.ndarray:
    train_values = values[train_ids]
    mean = float(train_values.mean()) if train_values.size else 0.0
    std = float(train_values.std()) if train_values.size and float(train_values.std()) > 0.0 else 1.0
    return ((values - mean) / std).reshape(-1, 1)


def _build_temperature_features(
    df_mix: pd.DataFrame,
    train_ids: list[int],
    *,
    temperature_col: str,
    temperature_mode: str,
) -> np.ndarray | None:
    if temperature_mode == "none":
        return None
    if temperature_col not in df_mix.columns:
        raise KeyError(f"Temperature column '{temperature_col}' not found in dataframe.")

    t_all = pd.to_numeric(df_mix[temperature_col], errors="coerce")
    t_train = t_all.iloc[train_ids]
    fill_value = float(t_train.mean()) if not t_train.isna().all() else 298.15
    temperatures = t_all.fillna(fill_value).to_numpy(dtype=float)

    if np.any(temperatures <= 0.0):
        raise ValueError("Temperature conditioning requires strictly positive temperatures in Kelvin.")

    if temperature_mode == "concat":
        return _standardize_from_train(temperatures, train_ids)
    if temperature_mode == "physics":
        return temperatures.reshape(-1, 1)

    raise ValueError(f"Unsupported temperature_mode={temperature_mode!r}")


def run_training(args: argparse.Namespace, *, save_artifacts: bool = True) -> dict[str, float]:
    pl.seed_everything(args.seed, workers=True)

    split_name = _split_name_from_path(args.split_csv)
    if args.use_mixmp and args.mixmp_type == "none":
        args.mixmp_type = "interaction"

    if split_name == "pure_to_mixture" and args.mixmp_type != "none":
        args.mixmp_type = "none"
        print("[mixmp] Auto-disabling interaction module for pure_to_mixture split.")

    train_df, val_df, test_df = load_split_dataframes(
        mix_csv=args.mix_csv,
        split_csv=args.split_csv,
        max_components=args.max_components,
        min_fraction=args.min_fraction,
        seed=args.seed,
    )

    rng = np.random.default_rng(args.seed)
    if args.max_train is not None and len(train_df) > args.max_train:
        train_df = train_df.iloc[sorted(rng.choice(np.arange(len(train_df)), size=args.max_train, replace=False).tolist())].copy()
    if args.max_val is not None and len(val_df) > args.max_val:
        val_df = val_df.iloc[sorted(rng.choice(np.arange(len(val_df)), size=args.max_val, replace=False).tolist())].copy()

    df_mix = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
    train_ids = list(range(0, len(train_df)))
    val_ids = list(range(len(train_df), len(train_df) + len(val_df)))
    test_ids = list(range(len(train_df) + len(val_df), len(df_mix)))

    if args.use_temperature_feature is True and args.temperature_mode == "none":
        args.temperature_mode = "concat"
    elif args.use_temperature_feature is False and args.temperature_mode != "none":
        raise ValueError(
            "--no-temperature-feature conflicts with an explicit --temperature-mode. "
            "Use only one temperature configuration mechanism."
        )
    elif args.temperature_mode == "none" and args.temperature_col in df_mix.columns:
        args.temperature_mode = "concat"
        print(
            f"[temperature] Auto-enabling concat mode because column "
            f"'{args.temperature_col}' is available in {args.mix_csv}."
        )

    if args.solute_component_index < 0 and "solute_inchi" in df_mix.columns:
        args.solute_component_index = 0
        print(
            "Auto-setting solute_component_index=0 because explicit solute context "
            f"is available in {args.mix_csv}."
        )

    print(f"[temperature] mode={args.temperature_mode}")
    if args.temperature_mode == "physics":
        print(f"[temperature] physics_law={args.temperature_physics_law}")
    print(f"[features] atom_featurizer_mode={args.atom_featurizer_mode}")
    print(f"[features] explicit_solute_context={'solute_inchi' in df_mix.columns}")
    temperature_features = _build_temperature_features(
        df_mix,
        train_ids,
        temperature_col=args.temperature_col,
        temperature_mode=args.temperature_mode,
    )

    comp_cols, _ = gather_component_columns(df_mix)
    explicit_solute_context = args.solute_component_index == 0 and _has_explicit_solute_context(df_mix, comp_cols)
    n_components = len(comp_cols) + (1 if explicit_solute_context else 0)

    predictor_x_d = temperature_features
    predictor_x_d_dim = 0 if predictor_x_d is None else predictor_x_d.shape[1]
    x_d_parts: list[np.ndarray] = []
    if predictor_x_d is not None:
        x_d_parts.append(predictor_x_d)

    x_d_all = np.hstack(x_d_parts) if x_d_parts else None
    if x_d_all is not None:
        all_data = build_all_data_with_xd(
            df_mix,
            x_d=x_d_all,
            target_col="value",
            solute_component_index=args.solute_component_index,
        )
        x_d_dim = x_d_all.shape[1]
    else:
        all_data = build_all_data(df_mix, target_col="value", solute_component_index=args.solute_component_index)
        x_d_dim = 0
    n_components = len(all_data) - 1

    train_data = take_rows(all_data, train_ids)
    val_data = take_rows(all_data, val_ids)
    test_data = take_rows(all_data, test_ids)

    use_mixmp = args.mixmp_type != "none"
    print(f"[mixmp] type={args.mixmp_type}")
    component_featurizer = SimpleMoleculeMolGraphFeaturizer(
        atom_featurizer=get_multi_hot_atom_featurizer(args.atom_featurizer_mode)
    )

    train_mcdset = build_mixture_dataset(
        train_data,
        n_components,
        use_mixmp=use_mixmp,
        atom_featurizer_mode=args.atom_featurizer_mode,
        component_featurizer=component_featurizer,
    )
    scaler = train_mcdset.normalize_targets()
    val_mcdset = build_mixture_dataset(
        val_data,
        n_components,
        use_mixmp=use_mixmp,
        atom_featurizer_mode=args.atom_featurizer_mode,
        component_featurizer=component_featurizer,
    )
    val_mcdset.normalize_targets(scaler)
    test_mcdset = build_mixture_dataset(
        test_data,
        n_components,
        use_mixmp=use_mixmp,
        atom_featurizer_mode=args.atom_featurizer_mode,
        component_featurizer=component_featurizer,
    )
    test_mcdset.normalize_targets(scaler)

    train_loader = DataLoader(train_mcdset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_mixture)
    val_loader = DataLoader(val_mcdset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_mixture)
    test_loader = DataLoader(test_mcdset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_mixture)

    model = build_model(
        n_components=n_components,
        scaler=scaler,
        aggregation=args.aggregation,
        solute_component_index=args.solute_component_index,
        mixmp_type=args.mixmp_type,
        x_d_dim=x_d_dim,
        temperature_mode=args.temperature_mode,
        temperature_physics_law=args.temperature_physics_law,
        atom_fdim=component_featurizer.atom_fdim,
        bond_fdim=component_featurizer.bond_fdim,
        message_hidden_dim=args.message_hidden_dim,
        message_passing_depth=args.message_passing_depth,
        predictor_hidden_dim=args.predictor_hidden_dim,
        predictor_n_layers=args.predictor_n_layers,
        warmup_epochs=args.warmup_epochs,
        init_lr=args.init_lr,
        max_lr=args.max_lr,
        final_lr=args.final_lr,
        predictor_x_d_dim=predictor_x_d_dim,
    )
    if args.agg_debug and hasattr(model.agg, "debug"):
        model.agg.debug = True

    loss_history_cb = LossHistoryRecorder() if args.loss_history_out is not None else None
    callbacks = [loss_history_cb] if loss_history_cb is not None else []
    checkpoint_cb = None
    run_start_time = time.time()

    output_dir = infer_output_dir(args.outputs_root, args.mix_csv, args.split_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=checkpoint_cb is not None,
        enable_progress_bar=True,
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.epochs,
        callbacks=callbacks or None,
        default_root_dir=str(output_dir),
    )
    trainer.fit(model, train_loader, val_loader)
    if checkpoint_cb is not None and checkpoint_cb.best_model_path:
        checkpoint_payload = torch.load(checkpoint_cb.best_model_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint_payload.get("state_dict")
        if state_dict is None:
            raise KeyError(f"Could not find state_dict in checkpoint {checkpoint_cb.best_model_path}.")
        model.load_state_dict(state_dict)
    trainer.test(model, dataloaders=test_loader, verbose=True)

    model.eval()
    preds = []
    val_preds = []
    with torch.no_grad():
        for batch in val_loader:
            bmgs, v_ds, x_d_batch, _, _, _, _ = batch
            val_preds.append(model(bmgs, v_ds, x_d_batch).cpu().numpy().reshape(-1))
        for batch in test_loader:
            bmgs, v_ds, x_d_batch, _, _, _, _ = batch
            preds.append(model(bmgs, v_ds, x_d_batch).cpu().numpy().reshape(-1))
    y_val_pred = np.concatenate(val_preds, axis=0) if len(val_preds) > 0 else np.array([], dtype=float)
    y_pred = np.concatenate(preds, axis=0)
    y_val_true = val_df["value"].to_numpy(dtype=float).reshape(-1)
    y_true = test_df["value"].to_numpy(dtype=float)
    metrics = {f"val_{k}": v for k, v in regression_metrics(y_val_true, y_val_pred).items()}
    metrics.update(regression_metrics(y_true, y_pred))
    metrics["train_walltime_seconds"] = time.time() - run_start_time
    metrics["normalization_scale"] = compute_normalization_scale(
        train_df["value"].to_numpy(dtype=float),
        method=args.hpo_normalization_method,
    )

    output_path = args.output if args.output is not None else output_dir / "model.pt"
    metrics_path = args.metrics_out if args.metrics_out is not None else output_dir / "metrics.json"
    parity_plot_path = args.parity_plot_out if args.parity_plot_out is not None else output_dir / "parity_plot.png"
    val_parity_plot_path = args.val_parity_plot_out if args.val_parity_plot_out is not None else output_dir / "parity_plot_val.png"
    preds_path = output_dir / "predictions.csv"

    if save_artifacts:
        torch.save(
            {
                "descriptor_only": False,
                "predictor_head": "physics" if args.temperature_mode == "physics" else "ffn",
                "atom_featurizer_mode": args.atom_featurizer_mode,
                "temperature_mode": args.temperature_mode,
                "temperature_physics_law": args.temperature_physics_law,
                "message_passing": model.message_passing.state_dict(),
                "mixmp": model.agg.mixmp.state_dict() if model.agg.mixmp is not None else None,
                "mixagg": model.agg.state_dict(),
                "predictor": model.predictor.state_dict(),
            },
            output_path,
        )
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        if args.loss_history_out is not None and loss_history_cb is not None:
            args.loss_history_out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(loss_history_cb.rows).to_csv(args.loss_history_out, index=False)
        pred_df = test_df.copy()
        if "_source_row_index" in pred_df.columns:
            pred_df = pred_df.rename(columns={"_source_row_index": "source_row_index"})
        pred_df["y_true"] = y_true
        pred_df["y_pred"] = y_pred
        pred_df.to_csv(preds_path, index=False)
        if len(y_val_true) > 0 and len(y_val_pred) == len(y_val_true):
            save_parity_plot(
                y_val_true,
                y_val_pred,
                val_parity_plot_path,
                title=f"{args.mix_csv.stem} | {args.split_csv.stem} | val",
            )
        save_parity_plot(y_true, y_pred, parity_plot_path, title=f"{args.mix_csv.stem} | {args.split_csv.stem}")
    print(
        f"Test metrics: RMSE={metrics['rmse']:.6f}, MAE={metrics['mae']:.6f}, "
        f"Pearson r={metrics['pearson_r']:.6f}"
    )
    print(
        f"Validation metrics: RMSE={metrics['val_rmse']:.6f}, MAE={metrics['val_mae']:.6f}, "
        f"Pearson r={metrics['val_pearson_r']:.6f}"
    )
    print(f"Training walltime: {metrics['train_walltime_seconds']:.2f} s")
    if save_artifacts:
        print(f"Saved metrics to {metrics_path}")
        print(f"Saved parity plot to {parity_plot_path}")
        if len(y_val_true) > 0 and len(y_val_pred) == len(y_val_true):
            print(f"Saved validation parity plot to {val_parity_plot_path}")
        print(f"Saved predictions to {preds_path}")
        if args.loss_history_out is not None:
            print(f"Saved loss history to {args.loss_history_out}")
        print(f"Saved model weights to {output_path}")
    return metrics


def run_hpo(args: argparse.Namespace) -> int:
    ray, tune, ASHAScheduler = require_ray_tune()
    manifest_entries = load_hpo_manifest(args.hpo_manifest) if args.hpo_manifest is not None else None
    output_dir = (
        args.outputs_root / args.hpo_manifest.stem
        if manifest_entries is not None
        else infer_output_dir(args.outputs_root, args.mix_csv, args.split_csv)
    )
    results_dir = args.hpo_results_dir if args.hpo_results_dir is not None else output_dir / "raytune"
    results_dir.mkdir(parents=True, exist_ok=True)

    search_space = build_gnn_search_space(profile=args.hpo_search_space_profile)
    fixed_config: dict[str, object] = {}
    if args.hpo_fixed_aggregation is not None:
        fixed_config["aggregation"] = args.hpo_fixed_aggregation
    if args.hpo_fixed_mixmp_type is not None:
        fixed_config["mixmp_type"] = args.hpo_fixed_mixmp_type
    for key in fixed_config:
        search_space.pop(key, None)
    ray_space = materialize_search_space(search_space, tune)
    scheduler_max_t = max(1, len(manifest_entries)) if manifest_entries is not None else max(1, args.epochs)
    scheduler = ASHAScheduler(
        max_t=scheduler_max_t,
        grace_period=max(1, min(args.hpo_grace_period, scheduler_max_t)),
        reduction_factor=max(2, args.hpo_reduction_factor),
    )

    def objective(config: dict[str, object]) -> None:
        trial_dir_getter = getattr(tune, "get_trial_dir", None)
        if callable(trial_dir_getter):
            trial_dir = Path(trial_dir_getter())
        else:  # pragma: no cover - compatibility path
            trial_dir = Path(tune.get_context().get_trial_dir())
        if manifest_entries is None:
            trial_args = clone_namespace(args)
            trial_args.hpo = False
            apply_trial_config(trial_args, config)
            apply_trial_config(trial_args, fixed_config)
            trial_args.outputs_root = trial_dir / "artifacts"
            metrics = run_training(trial_args, save_artifacts=True)
            tune.report(extract_numeric_metrics(metrics))
            return

        manifest_metrics: list[tuple[str, dict[str, float]]] = []
        total_manifest_entries = len(manifest_entries)
        for entry in manifest_entries:
            trial_args = clone_namespace(args)
            trial_args.hpo = False
            apply_trial_config(trial_args, config)
            apply_trial_config(trial_args, fixed_config)
            apply_trial_config(trial_args, entry["config"])
            trial_args.outputs_root = trial_dir / "artifacts" / entry["name"]
            entry_metrics = run_training(trial_args, save_artifacts=True)
            manifest_metrics.append((entry["name"], entry_metrics))
            tune.report(
                build_partial_manifest_report(
                    manifest_metrics,
                    primary_metric=args.hpo_metric,
                    total_manifest_entries=total_manifest_entries,
                )
            )

    ray.init(ignore_reinit_error=True, include_dashboard=False)
    analysis = tune.run(
        objective,
        config=ray_space,
        num_samples=args.hpo_num_samples,
        metric=args.hpo_metric,
        mode=args.hpo_mode,
        scheduler=scheduler,
        resources_per_trial={"cpu": args.hpo_cpus_per_trial, "gpu": args.hpo_gpus_per_trial},
        storage_path=str(results_dir),
        name=(
            f"{args.hpo_manifest.stem}_gnn_hpo"
            if manifest_entries is not None
            else f"{args.mix_csv.stem}_gnn_hpo"
        ),
        max_concurrent_trials=args.hpo_max_concurrent_trials,
        resume="AUTO" if args.hpo_resume else False,
    )
    best_trial = analysis.get_best_trial(metric=args.hpo_metric, mode=args.hpo_mode, scope="last")
    if best_trial is None:
        raise RuntimeError("Ray Tune completed without producing a best trial.")
    summary_path = results_dir / "best_config.json"
    write_hpo_summary(
        summary_path,
        best_config=best_trial.config,
        best_metrics=best_trial.last_result,
        search_space=search_space,
        metric=args.hpo_metric,
        mode=args.hpo_mode,
        num_samples=args.hpo_num_samples,
    )
    print(f"Saved HPO summary to {summary_path}")
    print(f"Best {args.hpo_metric}: {best_trial.last_result.get(args.hpo_metric)}")
    ray.shutdown()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.hpo:
        return run_hpo(args)
    run_training(args, save_artifacts=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
