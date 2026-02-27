#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader

from chemprop.data.datapoints import MoleculeDatapoint
from chemprop.nn.agg import MeanAggregation
from chemprop.nn.message_passing import BondMessagePassing
from chemprop.nn.predictors import RegressionFFN
from chemprop.nn.transforms import UnscaleTransform

from utils.mixtures import (
    AttentiveAggregation,
    ComponentDatapoint,
    ComponentDataset,
    ConcatAggregation,
    DeepsetsAggregation,
    InteractionMessagePassing,
    MixtureDatapoint,
    MixtureDataset,
    MixtureGraphDataset,
    MixtureMPNN,
    MulticomponentMessagePassing,
    Set2SetAggregation,
    SimpleMixtureGraphFeaturizer,
    WeightedSumAggregation,
    collate_mixture,
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
        row = {"epoch": int(trainer.current_epoch)}
        if train_loss is not None:
            row["train_loss_epoch"] = float(train_loss.detach().cpu())
        if val_loss is not None:
            row["val_loss"] = float(val_loss.detach().cpu())
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


def build_mixture_dataset(split_data: list[list], n_components: int, use_mixmp: bool) -> MixtureDataset:
    datasets = [ComponentDataset(split_data[i]) for i in range(n_components)]
    if use_mixmp:
        datasets.append(MixtureGraphDataset(split_data[-1]))
    return MixtureDataset(datasets)


def build_model(
    n_components: int,
    scaler,
    aggregation: str = "weightedsum",
    solute_component_index: int = -1,
    use_mixmp: bool = False,
    x_d_dim: int = 0,
) -> MixtureMPNN:
    mp_depth, mp_dh = 4, 101
    if solute_component_index >= 0:
        if solute_component_index >= n_components:
            raise ValueError(f"solute_component_index={solute_component_index} out of bounds for n_components={n_components}")
        solvent_group = [i for i in range(n_components) if i != solute_component_index]
        groups = [[solute_component_index], solvent_group]
        blocks = [BondMessagePassing(depth=mp_depth, d_h=mp_dh, activation="leakyrelu")] * 2
    else:
        groups = [list(range(n_components))]
        blocks = [BondMessagePassing(depth=mp_depth, d_h=mp_dh, activation="leakyrelu")]

    mcmp = MulticomponentMessagePassing(blocks=blocks, groups=groups, shared=False)
    if use_mixmp:
        default_mol_fdim, default_interaction_fdim = SimpleMixtureGraphFeaturizer().shape
        mixmp = InteractionMessagePassing(
            depth=3,
            d_v=mcmp.blocks[0].output_dim + default_mol_fdim,
            d_h=mcmp.blocks[0].output_dim,
            d_e=default_interaction_fdim,
            activation="leakyrelu",
        )
    else:
        mixmp = None

    graph_agg = MeanAggregation()
    fp_dims = [mcmp.blocks[0].output_dim] * n_components
    if aggregation == "weightedsum":
        mixagg = WeightedSumAggregation(graph_agg=graph_agg, groups=groups, fp_dims=fp_dims, mixmp=mixmp, debug=False)
    elif aggregation == "cat":
        mixagg = ConcatAggregation(graph_agg=graph_agg, groups=groups, fp_dims=fp_dims, mixmp=mixmp)
    elif aggregation == "deepsets":
        mixagg = DeepsetsAggregation(graph_agg=graph_agg, groups=groups, fp_dims=fp_dims, mixmp=mixmp)
    elif aggregation == "attentive":
        mixagg = AttentiveAggregation(graph_agg=graph_agg, groups=groups, fp_dims=fp_dims, mixmp=mixmp)
    elif aggregation == "set2set":
        mixagg = Set2SetAggregation(graph_agg=graph_agg, groups=groups, fp_dims=fp_dims, mixmp=mixmp)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    predictor = RegressionFFN(
        n_tasks=1,
        input_dim=mixagg.output_dim + x_d_dim,
        output_transform=UnscaleTransform.from_standard_scaler(scaler),
        hidden_dim=202,
        n_layers=4,
        activation="leakyrelu",
    )
    return MixtureMPNN(message_passing=mcmp, agg=mixagg, predictor=predictor)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--mix-csv", type=Path, default=repo_root / "datasets/fuel_ignition_numbers/processed_data/dcn_mix_exp.csv")
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=repo_root / "splits/fuel_ignition_numbers/dcn_mix_exp/split_definitions/molecule_type/cross_validation_10fold/fold_00.csv",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--accelerator", type=str, default="cpu")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--aggregation", type=str, default="weightedsum")
    parser.add_argument(
        "--use-temperature-feature",
        action="store_true",
        help="Include standardized temperature as an extra feature (disabled by default).",
    )
    parser.add_argument("--temperature-col", type=str, default="temperature")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-components", type=int, default=None, help="Maximum number of components to keep (default: all available).")
    parser.add_argument("--min-fraction", type=float, default=1e-4)
    parser.add_argument(
        "--fraction-basis",
        type=str,
        choices=["mole", "surface_area"],
        default="mole",
        help="Which fraction columns to use for mixtures (fraction_* vs surface_area_fraction_*).",
    )
    parser.add_argument("--no-mixmp", action="store_true")
    parser.add_argument("--solute-component-index", type=int, default=-1)
    parser.add_argument("--outputs-root", type=Path, default=repo_root / "scripts_training/outputs")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--metrics-out", type=Path, default=None)
    parser.add_argument("--parity-plot-out", type=Path, default=None)
    parser.add_argument("--val-parity-plot-out", type=Path, default=None)
    parser.add_argument("--loss-history-out", type=Path, default=None)
    parser.add_argument("--agg-debug", action="store_true", help="Enable weighted-sum aggregation debug prints.")
    return parser


def _split_name_from_path(split_csv: Path) -> str:
    parts = list(Path(split_csv).parts)
    if "split_definitions" in parts:
        i = parts.index("split_definitions")
        if i + 1 < len(parts):
            return parts[i + 1]
    return ""


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    pl.seed_everything(args.seed, workers=True)
    print(f"[features] temperature_feature_enabled={args.use_temperature_feature}")
    print(f"[fractions] basis={args.fraction_basis}")

    split_name = _split_name_from_path(args.split_csv)
    if split_name == "pure_to_mixture" and not args.no_mixmp:
        args.no_mixmp = True
        print("[mixmp] Auto-disabling interaction module for pure_to_mixture split.")

    if args.solute_component_index < 0 and args.mix_csv.stem in {"gsolv_mix_exp", "gsolv_mix_comp", "solubility_mix_exp"}:
        args.solute_component_index = 0
        print(
            "Auto-setting solute_component_index=0 for solute-focused task "
            f"'{args.mix_csv.stem}' to use [solute] + [solvents] mixture architecture."
        )

    train_df, val_df, test_df = load_split_dataframes(
        mix_csv=args.mix_csv,
        split_csv=args.split_csv,
        max_components=args.max_components,
        min_fraction=args.min_fraction,
        fraction_basis=args.fraction_basis,
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

    t_scaled = None
    if args.use_temperature_feature:
        if args.temperature_col not in df_mix.columns:
            raise KeyError(f"Temperature column '{args.temperature_col}' not found in {args.mix_csv}")
        t_all = pd.to_numeric(df_mix[args.temperature_col], errors="coerce")
        t_train = t_all.iloc[train_ids]
        t_mean = float(t_train.mean()) if not t_train.isna().all() else 0.0
        t_std = float(t_train.std()) if (not t_train.isna().all() and float(t_train.std()) > 0.0) else 1.0
        t_scaled = ((t_all.fillna(t_mean) - t_mean) / t_std).to_numpy(dtype=float).reshape(-1, 1)

    if t_scaled is not None:
        all_data = build_all_data_with_xd(df_mix, x_d=t_scaled, target_col="value", solute_component_index=args.solute_component_index)
        x_d_dim = 1
    else:
        all_data = build_all_data(df_mix, target_col="value", solute_component_index=args.solute_component_index)
        x_d_dim = 0
    n_components = len(all_data) - 1

    train_data = take_rows(all_data, train_ids)
    val_data = take_rows(all_data, val_ids)
    test_data = take_rows(all_data, test_ids)

    use_mixmp = not args.no_mixmp
    print(f"[mixmp] enabled={use_mixmp}")

    train_mcdset = build_mixture_dataset(train_data, n_components, use_mixmp=use_mixmp)
    scaler = train_mcdset.normalize_targets()
    val_mcdset = build_mixture_dataset(val_data, n_components, use_mixmp=use_mixmp)
    val_mcdset.normalize_targets(scaler)
    test_mcdset = build_mixture_dataset(test_data, n_components, use_mixmp=use_mixmp)
    test_mcdset.normalize_targets(scaler)

    train_loader = DataLoader(train_mcdset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_mixture)
    val_loader = DataLoader(val_mcdset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_mixture)
    test_loader = DataLoader(test_mcdset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_mixture)

    model = build_model(
        n_components=n_components,
        scaler=scaler,
        aggregation=args.aggregation,
        solute_component_index=args.solute_component_index,
        use_mixmp=use_mixmp,
        x_d_dim=x_d_dim,
    )
    if args.agg_debug and hasattr(model.agg, "debug"):
        model.agg.debug = True

    loss_history_cb = LossHistoryRecorder() if args.loss_history_out is not None else None
    callbacks = [loss_history_cb] if loss_history_cb is not None else None

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=True,
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.epochs,
        callbacks=callbacks,
    )
    trainer.fit(model, train_loader, val_loader)
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
    metrics = regression_metrics(y_true, y_pred)

    output_dir = infer_output_dir(args.outputs_root, args.mix_csv, args.split_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output if args.output is not None else output_dir / "model.pt"
    metrics_path = args.metrics_out if args.metrics_out is not None else output_dir / "metrics.json"
    parity_plot_path = args.parity_plot_out if args.parity_plot_out is not None else output_dir / "parity_plot.png"
    val_parity_plot_path = args.val_parity_plot_out if args.val_parity_plot_out is not None else output_dir / "parity_plot_val.png"
    preds_path = output_dir / "predictions.csv"

    torch.save(
        {
            "descriptor_only": False,
            "predictor_head": "ffn",
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
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(preds_path, index=False)
    if len(y_val_true) > 0 and len(y_val_pred) == len(y_val_true):
        save_parity_plot(y_val_true, y_val_pred, val_parity_plot_path, title=f"{args.mix_csv.stem} | {args.split_csv.stem} | val")
    save_parity_plot(y_true, y_pred, parity_plot_path, title=f"{args.mix_csv.stem} | {args.split_csv.stem}")
    print(
        f"Test metrics: RMSE={metrics['rmse']:.6f}, MAE={metrics['mae']:.6f}, "
        f"Pearson r={metrics['pearson_r']:.6f}"
    )
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved parity plot to {parity_plot_path}")
    if len(y_val_true) > 0 and len(y_val_pred) == len(y_val_true):
        print(f"Saved validation parity plot to {val_parity_plot_path}")
    print(f"Saved predictions to {preds_path}")
    if args.loss_history_out is not None:
        print(f"Saved loss history to {args.loss_history_out}")
    print(f"Saved model weights to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
