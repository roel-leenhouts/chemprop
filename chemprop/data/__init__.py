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
    "BatchMolAtomBondGraph": ("chemprop.data.collate", "BatchMolAtomBondGraph"),
    "BatchMolGraph": ("chemprop.data.collate", "BatchMolGraph"),
    "TrainingBatch": ("chemprop.data.collate", "TrainingBatch"),
    "collate_batch": ("chemprop.data.collate", "collate_batch"),
    "MolAtomBondTrainingBatch": ("chemprop.data.collate", "MolAtomBondTrainingBatch"),
    "collate_mol_atom_bond_batch": ("chemprop.data.collate", "collate_mol_atom_bond_batch"),
    "MulticomponentTrainingBatch": ("chemprop.data.collate", "MulticomponentTrainingBatch"),
    "collate_multicomponent": ("chemprop.data.collate", "collate_multicomponent"),
    "build_dataloader": ("chemprop.data.dataloader", "build_dataloader"),
    "LazyMoleculeDatapoint": ("chemprop.data.datapoints", "LazyMoleculeDatapoint"),
    "MoleculeDatapoint": ("chemprop.data.datapoints", "MoleculeDatapoint"),
    "MolAtomBondDatapoint": ("chemprop.data.datapoints", "MolAtomBondDatapoint"),
    "ReactionDatapoint": ("chemprop.data.datapoints", "ReactionDatapoint"),
    "MoleculeDataset": ("chemprop.data.datasets", "MoleculeDataset"),
    "CuikmolmakerDataset": ("chemprop.data.datasets", "CuikmolmakerDataset"),
    "ReactionDataset": ("chemprop.data.datasets", "ReactionDataset"),
    "Datum": ("chemprop.data.datasets", "Datum"),
    "MolAtomBondDatum": ("chemprop.data.datasets", "MolAtomBondDatum"),
    "MolAtomBondDataset": ("chemprop.data.datasets", "MolAtomBondDataset"),
    "MulticomponentDataset": ("chemprop.data.datasets", "MulticomponentDataset"),
    "MolGraphDataset": ("chemprop.data.datasets", "MolGraphDataset"),
    "MolGraph": ("chemprop.data.molgraph", "MolGraph"),
    "ClassBalanceSampler": ("chemprop.data.samplers", "ClassBalanceSampler"),
    "SeededSampler": ("chemprop.data.samplers", "SeededSampler"),
    "SplitType": ("chemprop.data.splitting", "SplitType"),
    "make_split_indices": ("chemprop.data.splitting", "make_split_indices"),
    "split_data_by_indices": ("chemprop.data.splitting", "split_data_by_indices"),
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
