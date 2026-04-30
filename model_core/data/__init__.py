from importlib import import_module

__all__ = [
    "BatchMolAtomBondGraph",
    "BatchMolGraph",
    "TrainingBatch",
    "collate_batch",
    "MolAtomBondTrainingBatch",
    "collate_mol_atom_bond_batch",
    "MulticomponentTrainingBatch",
    "collate_multicomponent",
    "build_dataloader",
    "LazyMoleculeDatapoint",
    "MoleculeDatapoint",
    "MolAtomBondDatapoint",
    "ReactionDatapoint",
    "MoleculeDataset",
    "CuikmolmakerDataset",
    "ReactionDataset",
    "Datum",
    "MolAtomBondDatum",
    "MolAtomBondDataset",
    "MulticomponentDataset",
    "MolGraphDataset",
    "MolGraph",
    "ClassBalanceSampler",
    "SeededSampler",
    "SplitType",
    "make_split_indices",
    "split_data_by_indices",
]

_EXPORTS = {
    "BatchMolAtomBondGraph": ("model_core.data.collate", "BatchMolAtomBondGraph"),
    "BatchMolGraph": ("model_core.data.collate", "BatchMolGraph"),
    "TrainingBatch": ("model_core.data.collate", "TrainingBatch"),
    "collate_batch": ("model_core.data.collate", "collate_batch"),
    "MolAtomBondTrainingBatch": ("model_core.data.collate", "MolAtomBondTrainingBatch"),
    "collate_mol_atom_bond_batch": ("model_core.data.collate", "collate_mol_atom_bond_batch"),
    "MulticomponentTrainingBatch": ("model_core.data.collate", "MulticomponentTrainingBatch"),
    "collate_multicomponent": ("model_core.data.collate", "collate_multicomponent"),
    "build_dataloader": ("model_core.data.dataloader", "build_dataloader"),
    "LazyMoleculeDatapoint": ("model_core.data.datapoints", "LazyMoleculeDatapoint"),
    "MoleculeDatapoint": ("model_core.data.datapoints", "MoleculeDatapoint"),
    "MolAtomBondDatapoint": ("model_core.data.datapoints", "MolAtomBondDatapoint"),
    "ReactionDatapoint": ("model_core.data.datapoints", "ReactionDatapoint"),
    "MoleculeDataset": ("model_core.data.datasets", "MoleculeDataset"),
    "CuikmolmakerDataset": ("model_core.data.datasets", "CuikmolmakerDataset"),
    "ReactionDataset": ("model_core.data.datasets", "ReactionDataset"),
    "Datum": ("model_core.data.datasets", "Datum"),
    "MolAtomBondDatum": ("model_core.data.datasets", "MolAtomBondDatum"),
    "MolAtomBondDataset": ("model_core.data.datasets", "MolAtomBondDataset"),
    "MulticomponentDataset": ("model_core.data.datasets", "MulticomponentDataset"),
    "MolGraphDataset": ("model_core.data.datasets", "MolGraphDataset"),
    "MolGraph": ("model_core.data.molgraph", "MolGraph"),
    "ClassBalanceSampler": ("model_core.data.samplers", "ClassBalanceSampler"),
    "SeededSampler": ("model_core.data.samplers", "SeededSampler"),
    "SplitType": ("model_core.data.splitting", "SplitType"),
    "make_split_indices": ("model_core.data.splitting", "make_split_indices"),
    "split_data_by_indices": ("model_core.data.splitting", "split_data_by_indices"),
}


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
