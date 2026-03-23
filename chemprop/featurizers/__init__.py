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
    "AtomFeatureMode": ("chemprop.featurizers.atom", "AtomFeatureMode"),
    "MultiHotAtomFeaturizer": ("chemprop.featurizers.atom", "MultiHotAtomFeaturizer"),
    "get_multi_hot_atom_featurizer": ("chemprop.featurizers.atom", "get_multi_hot_atom_featurizer"),
    "Featurizer": ("chemprop.featurizers.base", "Featurizer"),
    "GraphFeaturizer": ("chemprop.featurizers.base", "GraphFeaturizer"),
    "S": ("chemprop.featurizers.base", "S"),
    "T": ("chemprop.featurizers.base", "T"),
    "VectorFeaturizer": ("chemprop.featurizers.base", "VectorFeaturizer"),
    "MultiHotBondFeaturizer": ("chemprop.featurizers.bond", "MultiHotBondFeaturizer"),
    "BinaryFeaturizerMixin": ("chemprop.featurizers.molecule", "BinaryFeaturizerMixin"),
    "CountFeaturizerMixin": ("chemprop.featurizers.molecule", "CountFeaturizerMixin"),
    "MoleculeFeaturizerRegistry": ("chemprop.featurizers.molecule", "MoleculeFeaturizerRegistry"),
    "MorganBinaryFeaturizer": ("chemprop.featurizers.molecule", "MorganBinaryFeaturizer"),
    "MorganCountFeaturizer": ("chemprop.featurizers.molecule", "MorganCountFeaturizer"),
    "MorganFeaturizerMixin": ("chemprop.featurizers.molecule", "MorganFeaturizerMixin"),
    "RDKit2DFeaturizer": ("chemprop.featurizers.molecule", "RDKit2DFeaturizer"),
    "V1RDKit2DFeaturizer": ("chemprop.featurizers.molecule", "V1RDKit2DFeaturizer"),
    "V1RDKit2DNormalizedFeaturizer": ("chemprop.featurizers.molecule", "V1RDKit2DNormalizedFeaturizer"),
    "BatchCuikMolGraph": ("chemprop.featurizers.molgraph.molecule", "BatchCuikMolGraph"),
    "CuikmolmakerMolGraphFeaturizer": ("chemprop.featurizers.molgraph.molecule", "CuikmolmakerMolGraphFeaturizer"),
    "SimpleMoleculeMolGraphFeaturizer": ("chemprop.featurizers.molgraph.molecule", "SimpleMoleculeMolGraphFeaturizer"),
    "MolGraphCache": ("chemprop.featurizers.molgraph.cache", "MolGraphCache"),
    "MolGraphCacheFacade": ("chemprop.featurizers.molgraph.cache", "MolGraphCacheFacade"),
    "MolGraphCacheOnTheFly": ("chemprop.featurizers.molgraph.cache", "MolGraphCacheOnTheFly"),
    "CGRFeaturizer": ("chemprop.featurizers.molgraph.reaction", "CGRFeaturizer"),
    "CondensedGraphOfReactionFeaturizer": ("chemprop.featurizers.molgraph.reaction", "CondensedGraphOfReactionFeaturizer"),
    "RxnMode": ("chemprop.featurizers.molgraph.reaction", "RxnMode"),
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
