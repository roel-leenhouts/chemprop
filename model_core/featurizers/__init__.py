from importlib import import_module

__all__ = [
    "Featurizer",
    "S",
    "T",
    "VectorFeaturizer",
    "GraphFeaturizer",
    "MultiHotAtomFeaturizer",
    "AtomFeatureMode",
    "get_multi_hot_atom_featurizer",
    "MultiHotBondFeaturizer",
    "MolGraphCacheFacade",
    "MolGraphCache",
    "MolGraphCacheOnTheFly",
    "SimpleMoleculeMolGraphFeaturizer",
    "BatchCuikMolGraph",
    "CondensedGraphOfReactionFeaturizer",
    "CuikmolmakerMolGraphFeaturizer",
    "CGRFeaturizer",
    "RxnMode",
    "MoleculeFeaturizer",
    "MorganFeaturizerMixin",
    "BinaryFeaturizerMixin",
    "CountFeaturizerMixin",
    "MorganBinaryFeaturizer",
    "MorganCountFeaturizer",
    "RDKit2DFeaturizer",
    "MoleculeFeaturizerRegistry",
    "V1RDKit2DFeaturizer",
    "V1RDKit2DNormalizedFeaturizer",
]

_EXPORTS = {
    "AtomFeatureMode": ("model_core.featurizers.atom", "AtomFeatureMode"),
    "MultiHotAtomFeaturizer": ("model_core.featurizers.atom", "MultiHotAtomFeaturizer"),
    "get_multi_hot_atom_featurizer": ("model_core.featurizers.atom", "get_multi_hot_atom_featurizer"),
    "Featurizer": ("model_core.featurizers.base", "Featurizer"),
    "GraphFeaturizer": ("model_core.featurizers.base", "GraphFeaturizer"),
    "S": ("model_core.featurizers.base", "S"),
    "T": ("model_core.featurizers.base", "T"),
    "VectorFeaturizer": ("model_core.featurizers.base", "VectorFeaturizer"),
    "MultiHotBondFeaturizer": ("model_core.featurizers.bond", "MultiHotBondFeaturizer"),
    "BinaryFeaturizerMixin": ("model_core.featurizers.molecule", "BinaryFeaturizerMixin"),
    "CountFeaturizerMixin": ("model_core.featurizers.molecule", "CountFeaturizerMixin"),
    "MoleculeFeaturizerRegistry": ("model_core.featurizers.molecule", "MoleculeFeaturizerRegistry"),
    "MorganBinaryFeaturizer": ("model_core.featurizers.molecule", "MorganBinaryFeaturizer"),
    "MorganCountFeaturizer": ("model_core.featurizers.molecule", "MorganCountFeaturizer"),
    "MorganFeaturizerMixin": ("model_core.featurizers.molecule", "MorganFeaturizerMixin"),
    "RDKit2DFeaturizer": ("model_core.featurizers.molecule", "RDKit2DFeaturizer"),
    "V1RDKit2DFeaturizer": ("model_core.featurizers.molecule", "V1RDKit2DFeaturizer"),
    "V1RDKit2DNormalizedFeaturizer": ("model_core.featurizers.molecule", "V1RDKit2DNormalizedFeaturizer"),
    "BatchCuikMolGraph": ("model_core.featurizers.molgraph.molecule", "BatchCuikMolGraph"),
    "CuikmolmakerMolGraphFeaturizer": ("model_core.featurizers.molgraph.molecule", "CuikmolmakerMolGraphFeaturizer"),
    "SimpleMoleculeMolGraphFeaturizer": ("model_core.featurizers.molgraph.molecule", "SimpleMoleculeMolGraphFeaturizer"),
    "MolGraphCache": ("model_core.featurizers.molgraph.cache", "MolGraphCache"),
    "MolGraphCacheFacade": ("model_core.featurizers.molgraph.cache", "MolGraphCacheFacade"),
    "MolGraphCacheOnTheFly": ("model_core.featurizers.molgraph.cache", "MolGraphCacheOnTheFly"),
    "CGRFeaturizer": ("model_core.featurizers.molgraph.reaction", "CGRFeaturizer"),
    "CondensedGraphOfReactionFeaturizer": ("model_core.featurizers.molgraph.reaction", "CondensedGraphOfReactionFeaturizer"),
    "RxnMode": ("model_core.featurizers.molgraph.reaction", "RxnMode"),
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
