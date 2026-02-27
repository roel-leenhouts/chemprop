# -*- coding: utf-8 -*-

from abc import abstractmethod
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Iterable, NamedTuple, Sequence, TypeAlias

from lightning import pytorch as pl
import numpy as np
import pandas as pd
from rdkit.Chem import Mol
import rdkit.Chem as Chem
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from chemprop.data.collate import BatchMolGraph, collate_batch
from chemprop.data.datapoints import MoleculeDatapoint, _DatapointMixin
from chemprop.data.datasets import Datum, MoleculeDataset, MulticomponentDataset, ReactionDataset
from chemprop.data.molgraph import MolGraph
from chemprop.data.splitting import make_split_indices, split_data_by_indices
from chemprop.featurizers import Featurizer, SimpleMoleculeMolGraphFeaturizer
from chemprop.models import MulticomponentMPNN, multi
from chemprop.nn.agg import Aggregation, MeanAggregation
from chemprop.nn.hparams import HasHParams
from chemprop.nn.message_passing import BondMessagePassing, MessagePassing, AtomMessagePassing
from chemprop.nn.metrics import ChempropMetric, MSE, RMSE, MAE, R2Score
from chemprop.nn.predictors import Predictor, RegressionFFN
from chemprop.nn.transforms import ScaleTransform, UnscaleTransform
from chemprop.nn.utils import Activation, get_activation_function
from chemprop.utils import make_mol
from lightning.pytorch.core.mixins import HyperparametersMixin
from chemprop.conf import DEFAULT_ATOM_FDIM, DEFAULT_BOND_FDIM, DEFAULT_HIDDEN_DIM

"""### Introducing *Components* as the molecules that are part of a mixture with a given composition

We first extend the `MolGraph` class to include a global attribute `w_fp` which is the weight of the learned fingerprint of the molecule when averaging the fingerprints of components in the mixture.
"""

# See also chemprop.data.molgraph.MolGraph
class ComponentMolGraph(NamedTuple):
    V: np.ndarray
    E: np.ndarray
    edge_index: np.ndarray
    rev_edge_index: np.ndarray
    w_fp: float = 1.0
    """the weight of the component's fingerprint when combining components in the mixture"""

# See also chemprop.data.datasets.Datum
class ComponentDatum(NamedTuple):
    mg: ComponentMolGraph
    V_d: np.ndarray | None
    x_d: np.ndarray | None
    y: np.ndarray | None
    weight: float
    lt_mask: np.ndarray | None
    gt_mask: np.ndarray | None

"""The batched versions of `MolGraph` and `Datum` are created during collating datapoints. These are also extended, as well as the entire batch representing all components in the mixture."""

# See also chemprop.data.collate.BatchMolGraph
@dataclass(repr=False, eq=False, slots=True)
class BatchComponentMolGraph(BatchMolGraph):
    mgs: InitVar[Sequence[ComponentMolGraph]]
    w_fps: Tensor = field(init=False)
    __is_empty: bool = field(init=False)

    def __post_init__(self, mgs):
        # super(BatchComponentMolGraph, self).__post_init__(mgs)
        self._BatchMolGraph__size = len(mgs)
        self.__is_empty = True
        Vs = []
        Es = []
        edge_indexes = []
        rev_edge_indexes = []
        batch_indexes = []
        w_fps = []

        num_nodes = 0
        num_edges = 0
        for i, mg in enumerate(mgs):
            if mg is None:
                continue
            Vs.append(mg.V)
            Es.append(mg.E)
            edge_indexes.append(mg.edge_index + num_nodes)
            rev_edge_indexes.append(mg.rev_edge_index + num_edges)
            batch_indexes.append([i] * len(mg.V))
            w_fps.append(mg.w_fp)

            num_nodes += mg.V.shape[0]
            num_edges += mg.edge_index.shape[1]

        self.V = torch.from_numpy(np.concatenate(Vs)).float() if len(Vs) > 0 else None
        self.E = torch.from_numpy(np.concatenate(Es)).float() if len(Es) > 0 else None
        self.edge_index = torch.from_numpy(np.hstack(edge_indexes)).long() if len(edge_indexes) > 0 else None
        self.rev_edge_index = torch.from_numpy(np.concatenate(rev_edge_indexes)).long() if len(rev_edge_indexes) > 0 else None
        self.batch = torch.tensor(np.concatenate(batch_indexes)).long() if len(batch_indexes) > 0 else None
        self.w_fps = torch.from_numpy(np.array(w_fps)).float() if len(w_fps) > 0 else None

        if len(Vs) > 0:
            self.__is_empty = False

    def to(self, device: str | torch.device):
        if not self.is_empty():
            super(BatchComponentMolGraph, self).to(device)
            self.w_fps = self.w_fps.to(device)

    def is_empty(self) -> bool:
        """whether any :class:`MolGraph`\s are stored in this batch"""
        return self.__is_empty

# See also chemprop.data.collate.TrainingBatch
class BatchComponentDatum(NamedTuple):
    bmg: BatchComponentMolGraph
    V_d: Tensor | None
    X_d: Tensor | None
    Y: Tensor | None
    w: Tensor
    lt_mask: Tensor | None
    gt_mask: Tensor | None

# See also chemprop.data.collate.collate_batch
def collate_component(batch: Iterable[Datum]) -> BatchComponentDatum:
    mgs, V_ds, x_ds, ys, weights, lt_masks, gt_masks = zip(*batch)

    return BatchComponentDatum(
        BatchComponentMolGraph(mgs),
        None if V_ds[0] is None else torch.from_numpy(np.concatenate(V_ds)).float(),
        None if x_ds[0] is None else torch.from_numpy(np.array(x_ds)).float(),
        None if ys[0] is None else torch.from_numpy(np.array(ys)).float(),
        torch.tensor(weights, dtype=torch.float).unsqueeze(1),
        None if lt_masks[0] is None else torch.from_numpy(np.array(lt_masks)),
        None if gt_masks[0] is None else torch.from_numpy(np.array(gt_masks)),
    )

"""We also account for the ``w_fp`` for the datapoint and dataset generation"""

@dataclass
class ComponentDatapoint(MoleculeDatapoint):
    w_fp: np.ndarray | None = None
    """the weight of the molecule's learned fingerprint when averaging in the mixture"""


@dataclass
class ComponentDataset(MoleculeDataset, Dataset[ComponentMolGraph]):
    data: list[ComponentDatapoint]

    @property
    def w_fps(self) -> np.ndarray:
        return np.array([d.w_fp for d in self.data])

    def __getitem__(self, idx: int) -> ComponentDatum:
        d = self.data[idx]
        if d.mol:
            mg = self.mg_cache[idx]
            mg = ComponentMolGraph(w_fp=d.w_fp, *mg)
        else:
            mg = None

        return ComponentDatum(
            mg, self.V_ds[idx], self.X_d[idx], self.Y[idx], d.weight, d.lt_mask, d.gt_mask
        )

@dataclass(repr=False, eq=False)
class MixtureDataset(MulticomponentDataset): # TODO, potentially rename to anvoid confusion with MixtureGraphDataset?
    datasets: list[MoleculeDataset | ReactionDataset | ComponentDataset]

    def __getitem__(self, idx: int) -> list[ComponentDatum | Datum]:
        return [dset[idx] for dset in self.datasets]

"""### Introducing *Mixtures* as objects that represent molecules and their interactions

We first define the ``MixtureGraph`` class, i.e., nodes representing molecules and edges representing intermolecular interactions, together with the corresponding ``MixtureDatum``.
"""

# See also chemprop.data.molgraph.MolGraph
class MixtureGraph(NamedTuple):
    V: np.ndarray
    E: np.ndarray
    edge_index: np.ndarray
    rev_edge_index: np.ndarray

# See also chemprop.data.datasets.Datum
class MixtureDatum(NamedTuple):
    # TODO: we could already make list of Molecule/ComponentMolGraphs here and store molecular information of the mixture at this level instead in stacked component-specific datasets
    mg: MixtureGraph
    V_d: np.ndarray | None
    x_d: np.ndarray | None
    y: np.ndarray | None
    weight: float
    lt_mask: np.ndarray | None
    gt_mask: np.ndarray | None

"""We implement the ``SimpleMixtureGraphFeaturizer`` for ``MixtureGraph`` objects that takes in the molecules of a mixture, constructs a mixture graph, and adds descriptors for intermolecular interactions to the edge features (currently only considering Hydrogen bonding)."""

from chemprop.featurizers.base import VectorFeaturizer
from chemprop.utils.utils import EnumMapping
from enum import auto
from chemprop.featurizers.molgraph.mixins import _MolGraphFeaturizerMixin
from chemprop.featurizers.base import GraphFeaturizer

class MultiHotMolinteractionFeaturizer(VectorFeaturizer[Sequence[Mol]]):
    """A :class:`MultiHotMolinteractionFeaturizer` uses a multi-hot encoding to featurize interactions in mixture graphs.

    The generated interaction features are ordered as follows:
    * hydrogen bonding

    Parameters
    ----------
    max_hbond_num : int
        the number for maximum hydrogen bonds.
    """

    def __init__(
        self,
        h_bonds: Sequence[int],
    ):
        self.h_bonds = {i: i for i in h_bonds}

        self._subfeats: list[dict] = [
            self.h_bonds,
        ]
        subfeat_sizes = [
            1 + len(self.h_bonds),
        ]
        self.__size = sum(subfeat_sizes)

    def __len__(self) -> int:
        return self.__size

    def __call__(self, mols: Sequence[Mol] | None) -> np.ndarray:
        x = np.zeros(self.__size)

        if mols is None:
            return x
        if len(mols) == 1:
            feats = [
                self._descriptor_intra_HB(mols[0])
            ]
        elif len(mols) > 1:
            feats = [
                self._descriptor_inter_HB(mols)
            ]
        else:
            raise ValueError("Tried to featurize molecular interactions but empty list of mols was provided.")

        i = 0
        for feat, choices in zip(feats, self._subfeats):
            j = choices.get(feat, len(choices))
            x[i + j] = 1
            i += len(choices) + 1

        return x

    def _descriptor_intra_HB(self, mol):
        # From https://github.com/edgarsmdn/GH-GNN/blob/main/scr/models/utilities/mol2graph.py
        # Intra hydrogen-bond acidity and basicity
        return min(Chem.rdMolDescriptors.CalcNumHBA(mol), Chem.rdMolDescriptors.CalcNumHBD(mol))

    def _descriptor_inter_HB(self, mol_list):
        # From https://github.com/edgarsmdn/GH-GNN/blob/main/scr/models/utilities/mol2graph.py
        # Inter hydrogen-bond acidity and basicity
        if len(mol_list) != 2:
            raise ValueError(f"Interaction can only be calculated between two molecules. {len(mol_list)} molecules are given.")
        mol_1, mol_2 = mol_list[0], mol_list[1]
        return min(Chem.rdMolDescriptors.CalcNumHBA(mol_1), Chem.rdMolDescriptors.CalcNumHBD(mol_2)) +  min(Chem.rdMolDescriptors.CalcNumHBA(mol_2), Chem.rdMolDescriptors.CalcNumHBD(mol_1))

    @classmethod
    def hb(cls, max_hbond_num: int = 3):
        """The implementation of molecular interactions based on hydrogen bonding (hb) used in [1].

        Parameters
        ----------
        max_hbond_num : int, default=3
            Include a bit for all hydrogen bond numbers in the interval :math:`[1, \mathtt{max\_hbond\_num}]`

        References
        -----------
        .. [1] Medina, E. I. S., Linke, S., Stoll, M., & Sundmacher, K. (2023). Gibbs–Helmholtz graph neural network: capturing the temperature dependency of activity coefficients at infinite dilution. Digital Discovery, 2(3), 781-798. https://doi.org/10.1039/D2DD00142J
        """

        return cls(
            h_bonds=list(range(0, max_hbond_num)),
        )

class MolinteractionFeatureMode(EnumMapping):
    """The mode of an atom is used for featurization into a `MolGraph`"""

    HB = auto()

@dataclass
class _MixtureGraphFeaturizerMixin:
    # mol_featurizer: VectorFeaturizer[Mol] = field(default_factory=) # TODO: we could also featurize mols and thereby provide the option to omit the MPNN on individual molecules
    interaction_featurizer: VectorFeaturizer[list[Mol]] = field(default_factory=MultiHotMolinteractionFeaturizer.hb)

    def __post_init__(self):
        self.mol_fdim = 0 #len(self.mol_featurizer), TODO
        self.interaction_fdim = len(self.interaction_featurizer)

    @property
    def shape(self) -> tuple[int, int]:
        """the feature dimension of the molecules and interactions, respectively, of `MixtureGraph`s generated by
        this featurizer"""
        return self.mol_fdim, self.interaction_fdim

@dataclass
class SimpleMixtureGraphFeaturizer(_MixtureGraphFeaturizerMixin, GraphFeaturizer[Sequence[Mol]]):
    """A :class:`SimpleMoleculeMolGraphFeaturizer` is the default implementation of a
    :class:`MoleculeMolGraphFeaturizer`

    Parameters
    ----------
    interaction_featurizer : InteractionFeaturizer, default=MultiHotMolinteractionFeaturizer()
        the featurizer with which to calculate feature representations of the molecular interactions in a given
        mixture
    extra_interaction_fdim : int, default=0
        the dimension of the additional features that will be concatenated onto the calculated
        features of each interaction
    """

    extra_mol_fdim: InitVar[int] = 0
    extra_interaction_fdim: InitVar[int] = 0

    def __post_init__(self, extra_mol_fdim: int = 0, extra_interaction_fdim: int = 0):
        super().__post_init__()

        self.extra_mol_fdim = extra_mol_fdim
        self.extra_interaction_fdim = extra_interaction_fdim
        self.mol_fdim += self.extra_mol_fdim
        self.interaction_fdim += self.extra_interaction_fdim

    def __call__(
        self,
        mols: list[Chem.Mol],
        mol_features_extra: np.ndarray | None = None,
        interaction_features_extra: np.ndarray | None = None,
    ) -> MolGraph:
        n_mols = len(mols)
        n_interactions = int((n_mols - 1) * n_mols / 2 + n_mols)

        if mol_features_extra is not None and len(mol_features_extra) != n_mols:
            raise ValueError(
                "Input mixture must have same number of molecules as `len(mol_features_extra)`!"
                f"got: {n_mols} and {len(mol_features_extra)}, respectively"
            )

        if interaction_features_extra is not None and len(interaction_features_extra) != n_interactions:
            raise ValueError(
                "Input mixture must have same number of interactions (n_mols * (n_mols / 2 + 1)) as `len(interaction_features_extra)`!"
                f"got: {n_interactions} and {len(interaction_features_extra)}, respectively"
            )

        if n_mols == 0:
            V = np.zeros((1, self.mol_fdim), dtype=np.single)
        elif mol_features_extra is not None:
            V = mol_features_extra
        else:
            V = np.zeros((n_mols, self.mol_fdim), dtype=np.single)

        E = np.empty((2 * n_interactions, self.interaction_fdim))
        edge_index = [[], []]
        i = 0
        for mol1_idx, mol1 in enumerate(mols):
            for mol2_idx, mol2 in enumerate(mols):
                # avoid duplicate interactions
                if mol2_idx < mol1_idx:
                    continue
                # self-interaction
                if mol1_idx == mol2_idx:
                    x_e = self.interaction_featurizer([mol1])
                # intermolecular interaction
                else:
                    x_e = self.interaction_featurizer([mol1, mol2])
                if interaction_features_extra is not None:
                    x_e = np.concatenate((x_e, interation_features_extra[mol_idx1+mol_idx2]), dtype=np.single) # the indexing is not obvious

                E[i : i + 2] = x_e

                edge_index[0].extend([mol1_idx, mol2_idx])
                edge_index[1].extend([mol2_idx, mol1_idx])

                i += 2

        rev_edge_index = np.arange(len(E)).reshape(-1, 2)[:, ::-1].ravel()
        edge_index = np.array(edge_index, int)

        return MixtureGraph(V, E, edge_index, rev_edge_index)

"""Batching of mixture graphs is analogous to molecular graph batching. We introduce similar classes to avoid confusion of mixture graph with molecular graph objects and to make future extensions straightforward."""

class BatchMixtureGraph(BatchMolGraph):
    """A :class:`BatchMixtureGraph` represents a batch of individual :class:`MixtureGraph`\s.

    It has all the attributes of a ``BatchMolGraph``.
    """
    mgs: InitVar[Sequence[MixtureGraph]]


# See also chemprop.data.collate.MulticomponentTrainingBatch
class MixtureBatch(NamedTuple):
    bmgs: list[MixtureGraph]
    V_ds: list[Tensor | None]
    X_d: Tensor | None
    Y: Tensor | None
    w: Tensor
    lt_mask: Tensor | None
    gt_mask: Tensor | None


# See also chemprop.data.collate.TrainingBatch
class BatchMixtureDatum(NamedTuple):
    bmg: BatchMixtureGraph
    V_d: Tensor | None
    X_d: Tensor | None
    Y: Tensor | None
    w: Tensor
    lt_mask: Tensor | None
    gt_mask: Tensor | None

def collate_mixturegraph(batch: Iterable[Datum]) -> BatchMixtureDatum:
    mgs, V_ds, x_ds, ys, weights, lt_masks, gt_masks = zip(*batch)

    return BatchMixtureDatum(
        BatchMixtureGraph(mgs),
        None if V_ds[0] is None else torch.from_numpy(np.concatenate(V_ds)).float(),
        None if x_ds[0] is None else torch.from_numpy(np.array(x_ds)).float(),
        None if ys[0] is None else torch.from_numpy(np.array(ys)).float(),
        torch.tensor(weights, dtype=torch.float).unsqueeze(1),
        None if lt_masks[0] is None else torch.from_numpy(np.array(lt_masks)),
        None if gt_masks[0] is None else torch.from_numpy(np.array(gt_masks)),
    )

# See also chemprop.data.collate.collate_multicomponent
def collate_mixture(batches: Iterable[Iterable[ComponentDatum | Datum]]) -> MixtureBatch:
    tbs = []
    for batch in zip(*batches):
        if isinstance(batch[0], MixtureDatum):
            tbs.append(collate_mixturegraph(batch))
        elif isinstance(batch[0], Datum):
            tbs.append(collate_batch(batch))
        else:
            tbs.append(collate_component(batch))

    return MixtureBatch(
        [tb.bmg for tb in tbs],
        #[tmixb.bmg for tmixb in tmixbs],
        [tb.V_d for tb in tbs],
        tbs[0].X_d,
        tbs[0].Y,
        tbs[0].w,
        tbs[0].lt_mask,
        tbs[0].gt_mask,
    )

"""Generating mixture datapoints and datasets is based on the provided molecules in the form of SMILES strings"""

@dataclass
class _MixtureDatapointMixin:
    mols: list[Chem.Mol]
    """the mixture associated with this datapoint"""

    @classmethod
    def from_smis(
        cls, smis: str, *args, keep_h: bool = False, add_h: bool = False, **kwargs
    ): #-> _MixtureDatapointMixin: TODO?
        mols = [make_mol(smi, keep_h, add_h) for smi in smis if smi]

        kwargs["name"] = smis if "name" not in kwargs else kwargs["name"]

        return cls(mols, *args, **kwargs)

@dataclass
class MixtureDatapoint(_DatapointMixin, _MixtureDatapointMixin):

    V_f: np.ndarray | None = None
    """a numpy array of shape ``V x d_vf``, where ``V`` is the number of molecules in the mixture, and
    ``d_vf`` is the number of additional features that will be concatenated to molecule-level features
    *before* message passing"""
    E_f: np.ndarray | None = None
    """A numpy array of shape ``E x d_ef``, where ``E`` is the number of interactions in the mixture, and
    ``d_ef`` is the number of additional features containing additional features that will be
    concatenated to interaction-level features *before* message passing"""
    V_d: np.ndarray | None = None
    """A numpy array of shape ``V x d_vd``, where ``V`` is the number of molecules in the mixture, and
    ``d_vd`` is the number of additional descriptors that will be concatenated to molecule-level
    descriptors *after* message passing"""

    def __post_init__(self):
        NAN_TOKEN = 0
        if self.V_f is not None:
            self.V_f[np.isnan(self.V_f)] = NAN_TOKEN
        if self.E_f is not None:
            self.E_f[np.isnan(self.E_f)] = NAN_TOKEN
        if self.V_d is not None:
            self.V_d[np.isnan(self.V_d)] = NAN_TOKEN

        super().__post_init__()

    def __len__(self) -> int:
        return 1

@dataclass
class MixtureGraphDataset(MoleculeDataset, Dataset[MixtureGraph]):
    data: list[MixtureDatapoint]
    featurizer: Featurizer[list[Mol], MixtureGraph] = field(default_factory=SimpleMixtureGraphFeaturizer)

    def __getitem__(self, idx: int) -> MixtureDatum:
        d = self.data[idx]
        mg = self.mg_cache[idx]
        return MixtureDatum(
            mg, self.V_ds[idx], self.X_d[idx], self.Y[idx], d.weight, d.lt_mask, d.gt_mask
        )

    @property
    def smiles(self) -> list[list[str]]:
        """the SMILES strings associated with the dataset"""
        return [[Chem.MolToSmiles(mol) for mol in d.mols] for d in self.data]

    @property
    def mols(self) -> list[list[Chem.Mol]]:
        """the molecules associated with the dataset"""
        return [[mol for mol in d.mols] for d in self.data]

"""### Message passing with multiple components from different ``groups``

The groups provide the indices for the respective molecular/component-wise datasets. For example, ``groups = [[0], [1, 2, 3]]``, where the first group corresponds to solute (1 molecule) and the second group corresponds to solvents (3 molecules). If all molecules/components are part of a mixture, one single group should be used, i.e., ``groups = [[0, 1, 2, 3]]``.

Note that the molecules within each group use the message passing block. Also, if ``shared = True``, all groups share a message passing block.
"""

class MulticomponentMessagePassing(nn.Module, HasHParams):
    """A `MulticomponentMessagePassing` performs message-passing on each individual input in a
    multicomponent input then concatenates the representation of each input to construct a
    global representation

    Parameters
    ----------
    blocks : Sequence[MessagePassing]
        the invidual message-passing blocks for each input
    groups: Sequence[Sequence[int]]
        the indices of the molecules/components split into groups
    shared : bool, default=False
        whether one block will be shared among all components in an input. If not, a separate
        block will be learned for each component.
    """

    def __init__(
        self,
        blocks: Sequence[MessagePassing],
        groups: Sequence[Sequence[int]],
        shared: bool = False,
        ):
        super().__init__()
        self.hparams = {
            "cls": self.__class__,
            "blocks": [block.hparams for block in blocks],
            "groups": groups,
            "shared": shared,
        }

        if len(blocks) == 0:
            raise ValueError("arg 'blocks' was empty!")
        if groups is None:
            raise ValueError("arg 'groups' was empty!")

        if shared:
            if len(blocks) > 1:
                logger.warning(
                    "More than 1 block was supplied but 'shared' was True! Using only the 0th block..."
                )
            if len(groups) != sum(len(g) if isinstance(g, list) else 1 for g in groups):
                logger.warning(
                    "Different groups were supplied but 'shared' was True! Using only the 0th block for all groups..."
                )
        else:
            if len(blocks) != len(groups):
                raise ValueError(
                    "arg 'len(groups)' must be equal to `len(blocks)` if 'shared' is False!"
                    f"got: {len(groups)} and {len(blocks)}, respectively."
                )

        self.groups = groups
        self.shared = shared
        self.blocks = nn.ModuleList()
        # Use one message passing block for all groups
        if shared:
            self.blocks.extend([blocks[0]] * len(groups))
        # Use group-wise message passing block
        else:
            b_idx = 0
            for g_idx, g in enumerate(groups):
                self.blocks.extend([blocks[g_idx]] * len(g))

    def __len__(self) -> int:
        return len(self.blocks)

    @property
    def output_dim(self) -> int:
        d_o = sum(block.output_dim for block in self.blocks)

        return d_o

    def forward(self, bmgs: Iterable[BatchMolGraph], V_ds: Iterable[Tensor | None]) -> list[Tensor]:
        """Encode the multicomponent inputs

        Parameters
        ----------
        bmgs : Iterable[BatchMolGraph]
        V_ds : Iterable[Tensor | None]

        Returns
        -------
        list[Tensor]
            a list of tensors of shape `V x d_i` containing the respective encodings of the `i`\th
            component, where `d_i` is the output dimension of the `i`\th encoder
        """
        # Note: check if bmg is None in case we consider mixtures with different number of components
        if V_ds is None:
            return [block(bmg) if not (bmg.V is None) else None for block, bmg in zip(self.blocks, bmgs)]
        else:
            return [block(bmg, V_d) if not (bmg.V is None) else None for block, bmg, V_d in zip(self.blocks, bmgs, V_ds)]

"""### Mixture-level message passing

We provide two types of message passing on the mixture level:
* On-the-fly: a fully-connected mixture graph is constructed during the forward pass of the model, whereas no additional molecular (interaction) descriptors are calculated. That is, we just enable passing information between the learned moleular fingerprints (without any edge features).
* MixtureGraph: a ``MixtureGraphDataset`` (see above) must be generated and passed to the model. We then implement ``MolecularMessagePassing``, mimicking ``AtomMessagePassing``, i.e., node-centered message passing. We also implement ``InteractionMessagePassing``, mimicking ``BondMessagePassing``, i.e., edge-centered message passing.
"""

class MixtureMessagePassing(nn.Module, HyperparametersMixin):
    r"""A :class:`MixtureMessagePassing` encodes a batch of mixtures by passing messages along
    molecules constructing a fully connected graph to model intermolecular interactions.

    It implements the following operation:

    .. math::

        h_v^{(0)} &= \tau \left( \mathbf{W}_i(x_v) \right) \\
        m_v^{(t)} &= \sum_{u \in \mathcal{w \in V \setminu v} h_w^{(t-1)} \\
        h_v^{(T)} &= \tau\left(h_v^{(0)} + \mathbf{W}_h m_v^{(t-1)}\right) \\

    where :math:`\tau` is the activation function; :math:`\mathbf{W}_i`, :math:`\mathbf{W}_h` are learned weight matrices; :math:`e_{vw}` is the feature vector of the
    bond between molecules :math:`v` and :math:`w`; :math:`x_v` is the feature vector of molecule :math:`v`;
    :math:`h_v^{(t)}` is the hidden representation of atom :math:`v` at iteration :math:`t`;
    :math:`m_v^{(t)}` is the message received by atom :math:`v` at iteration :math:`t`; and
    :math:`t \in \{1, \dots, T\}` is the number of message passing iterations.
    """

    def __init__(
        self,
        d_v: int = DEFAULT_HIDDEN_DIM,
        d_e: int | None = None,
        d_h: int = DEFAULT_HIDDEN_DIM,
        d_vd: int | None = None,
        bias: bool = False,
        depth: int = 1,
        activation: str | Activation = Activation.RELU,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.hparams["cls"] = self.__class__

        self.depth = depth
        self.tau = get_activation_function(activation)

        self.W_i = nn.Linear(d_v, d_h, bias)
        self.W_h = nn.Linear(d_h, d_h, bias) # TODO consider E
        self.W_o = None
        self.W_d = None

    def initialize(self, V: Tensor) -> Tensor:
        return self.W_i(V)

    def message(self, H: Tensor):
        # assume fully connected graph, TODO
        H = torch.transpose(H, 0, 1) # b x n x d
        M_t = H.unsqueeze(2).expand(-1, -1, H.size(1), -1) # b x n x n x d
        M_t = self.W_h(M_t)
        mask = ~torch.eye(H.size(1), dtype=bool, device=H.device).unsqueeze(0) # exclude self-loops (n x n)
        M_t = (M_t * mask.unsqueeze(-1)).sum(dim=1) # b x n x d
        M_t = torch.transpose(M_t, 0, 1) # n x b x d
        return M_t

    def update(self, M_t: Tensor, H_0: Tensor):
        H_t = self.tau(H_0 + M_t)
        return H_t

    def finalize(self, H_t: Tensor):
        return [h for h in H_t]

    def forward(self, V: list[Tensor]):
        H_0 = self.initialize(torch.stack(V))
        H = self.tau(H_0)
        for _ in range(self.depth):
            M = self.message(H)
            H = self.update(M, H_0)

        return self.finalize(H)

class MolecularMessagePassing(AtomMessagePassing):

    def forward(self, bmg: BatchMixtureGraph, Hs: list[Tensor], Hs_batch: list[Tensor], V_d: Tensor | None = None) -> Tensor:
        r"""Encode a batch of molecular graphs.

        Parameters
        ----------
        bmg: BatchMixtureGraph
            a batch of :class:`BatchMixtureGraph`s to encode
        Hs: list[Tensor]
            the molecular fiingerprint tensors
        V_d : Tensor | None, default=None
            an optional tensor of shape ``V x d_vd`` containing additional descriptors for each molecule
            in the batch. These will be concatenated to the learned molecular descriptors and
            transformed before the readout phase.

        Returns
        -------
        Tensor
            a tensor of shape ``V x d_h`` or ``V x (d_h + d_vd)`` containing the encoding of each
            molecule in the batch, depending on whether additional molecular descriptors were provided
        """
        batch_size = max(H_b.max().item() for H_b in Hs_batch if not (H_b is None))
        flat_Hs = []
        count_n = [0 for _ in range(len(Hs_batch))]
        flat_idx = []
        for b in range(batch_size+1):
            for idx, (H, H_b) in enumerate(zip(Hs, Hs_batch)):
                if not (H is None):
                    if (b in H_b):
                    # TODO: add extra mol features
                    # if the mixture graph nodes vectors store information, they correspond to extra molecular features that should be considered in message passing
                    #if bmg.V[0].numel() != 0:
                    #    flat_Hs.append(torch.cat((H[count_b], bmg.V[count_mg]), dim=1))
                    # otherwise empty node vectors are overwritten
                    #else:
                        flat_Hs.append(H[count_n[idx]])
                        flat_idx.append(idx)
                        count_n[idx] += 1
        flat_Hs = torch.stack(flat_Hs)
        if not isinstance(bmg, BatchMixtureGraph):
            raise TypeError(f"MixtureGraphMessagePassing requires class :class:`BatchMixtureGraph` as input but received object of :class:`{type(bmg)}`")
        bmg.V = flat_Hs
        Hs = bmg.V
        Hs = super().forward(bmg, V_d)
        flat_idx = torch.tensor(flat_idx)
        Hs = [Hs[flat_idx == i] for i, _ in enumerate(Hs_batch)]
        return Hs


class InteractionMessagePassing(BondMessagePassing):

    def forward(self, bmg: BatchMixtureGraph, Hs: list[Tensor], Hs_batch: list[Tensor], V_d: Tensor | None = None) -> Tensor:
        r"""Encode a batch of molecular graphs.

        Parameters
        ----------
        bmg: BatchMixtureGraph
            a batch of :class:`BatchMixtureGraph`s to encode
        Hs: list[Tensor]
            the molecular fiingerprint tensors
        V_d : Tensor | None, default=None
            an optional tensor of shape ``V x d_vd`` containing additional descriptors for each molecule
            in the batch. These will be concatenated to the learned molecular descriptors and
            transformed before the readout phase.

        Returns
        -------
        Tensor
            a tensor of shape ``V x d_h`` or ``V x (d_h + d_vd)`` containing the encoding of each
            molecule in the batch, depending on whether additional molecular descriptors were provided
        """
        batch_size = max(H_b.max().item() for H_b in Hs_batch if not (H_b is None))
        flat_Hs = []
        count_n = [0 for _ in range(len(Hs_batch))]
        flat_idx = []
        for b in range(batch_size+1):
            for idx, (H, H_b) in enumerate(zip(Hs, Hs_batch)):
                if not (H is None):
                    if (b in H_b):
                    # TODO: add extra mol features
                    # if the mixture graph nodes vectors store information, they correspond to extra molecular features that should be considered in message passing
                    #if bmg.V[0].numel() != 0:
                    #    flat_Hs.append(torch.cat((H[count_b], bmg.V[count_mg]), dim=1))
                    # otherwise empty node vectors are overwritten
                    #else:
                        flat_Hs.append(H[count_n[idx]])
                        flat_idx.append(idx)
                        count_n[idx] += 1
        flat_Hs = torch.stack(flat_Hs)
        if not isinstance(bmg, BatchMixtureGraph):
            raise TypeError(f"MixtureGraphMessagePassing requires class :class:`BatchMixtureGraph` as input but received object of :class:`{type(bmg)}`")
        bmg.V = flat_Hs
        Hs = bmg.V
        Hs = super().forward(bmg, V_d)
        flat_idx = torch.tensor(flat_idx)
        Hs = [Hs[flat_idx == i] for i, _ in enumerate(Hs_batch)]
        return Hs

"""### ``MixtureAggregation``: From molecular to mixture representations

We extend the ``Aggregation`` class suited for molecules to handle both molecules and their mixtures. The abstract ``MixtureAggregation`` class covers:
* ``Aggregation``: Atom-to-molecule aggregation
* ``MixtureMessagePassing``: mixture-level message passing (based on mixture graphs), *Note*: MixtureGraphDataset is assumed to have index `-1`,
while the inheriting classes implement:
* Molecule-to-mixture aggregation.

We currently include:
* ``ConcatAggregation``: Simply concatenating all molecular fingerprints and the individual compositions.
* ``WeightedSumAggregation``: groups-wise sum of molecular fingerprints multiplied by their individual compositions
* ``DeepsetsAggregation``: can be seen as an extension of ``WeightedSumAggregation``, whereas the individual moelcular fingerprints multiplied by the compositions pass a _local_ MLP before being summed and then the group-wise sums pass a _global_ MLP
* ``AttentiveAggregation``: groups-wise attention layer applied to molecular fingerprints multiplied by their individual compositions (meaning that the weighting of the individual fingerprints is adjusted by attention logits)
* ``Set2SetAggregation``: recurrent architecture based on LSTMs that aggregate group-wise moleuclar fingerprints multiplied by their individual composition into a mixture representation

*Note*: All mixture aggregations operate group-wise and then concatenate the group-based molecular/mixture representations
"""

class MixtureAggregation(nn.Module, HasHParams):
    output_dim: int

    def __init__(
        self,
        graph_agg: Aggregation,
        groups: Sequence[Sequence[int]],
        fp_dims: Sequence[int],
        mixmp: MixtureMessagePassing | None,
        *args,
        **kwargs
    ):
        super().__init__()
        self.hparams = {
            "cls": self.__class__,
            "groups": groups,
            "fp_dims": fp_dims,
        }
        self.graph_agg = graph_agg
        self.groups = groups
        self.fp_dims = fp_dims
        self.mixmp = mixmp

    @abstractmethod
    def forward(
        self, H_vs: list[Tensor], bmgs: list[BatchComponentMolGraph | BatchMolGraph]
    ) -> Tensor:
        """Aggregate component representations into a mixture representation"""
        # Atom-to-molecule aggregation
        self.Hs, self.w_fps, self.Hs_batch = self.mol_forward(H_vs, bmgs)

        # Mixture-level message passing
        if not (self.mixmp is None):
            if isinstance(self.mixmp, MixtureMessagePassing):
                # Varying number of components can cause sparse batches, hence complete with zero-entries
                self.Hs, self.w_fps, self.Hs_batch = self.complete_sparse_batch(self.Hs, self.w_fps, self.Hs_batch)
                self.Hs = self.mixmp(self.Hs)
            elif isinstance(self.mixmp, (MolecularMessagePassing, InteractionMessagePassing)):
                # Note that we assume MixtureGraphDataset to have index -1 -> TODO: make this dynamic
                self.Hs = self.mixmp(bmgs[-1], self.Hs, self.Hs_batch)
                # Varying number of components can cause sparse batches, hence complete with zero-entries
                self.Hs, self.w_fps, self.Hs_batch = self.complete_sparse_batch(self.Hs, self.w_fps, self.Hs_batch)
            else:
                raise NotImplementedError(f":class:`MixtureMessagePassing` of type {type(self.mixmp)} not implemented yet.")
        else:
            # Without an interaction module, downstream aggregations still expect dense
            # [component x batch x dim] tensors, so fill sparse component batches here.
            self.Hs, self.w_fps, self.Hs_batch = self.complete_sparse_batch(self.Hs, self.w_fps, self.Hs_batch)

        # Molecule-to-mixture aggregation: tbi in subclasses

    def mol_forward(
        self, H_vs: list[Tensor], bmgs: list[BatchComponentMolGraph | BatchMolGraph]
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        # Hs: n x b x d (but not each mixture has n components, so n x b can be incomplete, hence we need to synthetically adapt the sizes)
        Hs, w_fps, Hs_batch = zip(*[(self.graph_agg(H_v, torch.unique(bmg.batch, return_inverse=True)[1]), bmg.w_fps, torch.unique(bmg.batch)) if (isinstance(bmg, BatchComponentMolGraph) and (bmg.batch is not None)) else (self.graph_agg(H_v, torch.unique(bmg.batch, return_inverse=True)[1]), None, torch.unique(bmg.batch)) if (bmg.batch is not None) else (None, None, None) for H_v, bmg in zip(H_vs, bmgs)])
        Hs, w_fps, Hs_batch =  list(Hs), list(w_fps), list(Hs_batch)
        return Hs, w_fps, Hs_batch

    def complete_sparse_batch(
        self, Hs: list[Tensor], w_fps: list[Tensor], Hs_batch: list[Tensor]
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        # make Hs and w_fps the same size wrt n x b by adding zero-values/tensors
        Hs = self._complete_sparse_tensorlist(Hs, Hs_batch, self.fp_dims)
        if not all(f is None for f in w_fps):
            w_fps = self._complete_sparse_tensorlist(w_fps, Hs_batch, [None for _ in range(len(Hs_batch))])
        return Hs, w_fps, Hs_batch

    def _complete_sparse_tensorlist(
        self, Hs: list[Tensor], Hs_batch: list[Tensor], dim: list[int]
    ) -> list[Tensor]:
        batch_size = max(H_b.max().item() for H_b in Hs_batch if not (H_b is None))
        device = [H.device for H in Hs if not (H is None)][0] # workaround, TODO
        compl_Hs = []
        for n_idx, (n_H, n_H_batch) in enumerate(zip(Hs, Hs_batch)):
            if dim[n_idx]:
                compl_H = torch.zeros((batch_size+1, dim[n_idx]), dtype=torch.float32, device=device)
            else:
                compl_H = torch.zeros((batch_size+1), dtype=torch.float32, device=device)
            if (n_H is not None) and (n_H_batch is not None):
                compl_H[n_H_batch] = n_H
            compl_Hs.append(compl_H)
        return compl_Hs

class ConcatAggregation(MixtureAggregation):
    r"""Concatenate aggregation of the graph-level representation:

    .. math::
        \mathbf h = \text{concat}_c \mathbf h_c
    """
    @property
    def components_in_mixture(self) -> set[int]:
        return {idx for group in self.groups if len(group) > 1 for idx in group}

    @property
    def output_dim(self) -> int:
        return sum(self.fp_dims) + len(self.components_in_mixture)

    def forward(
        self, H_vs: list[Tensor], bmgs: list[BatchComponentMolGraph | BatchMolGraph]
    ) -> Tensor:
        super().forward(H_vs, bmgs)

        w_fps = torch.stack([self.w_fps[idx] for idx in self.components_in_mixture], dim=1)
        return torch.cat(self.Hs + [w_fps], 1)

class WeightedSumAggregation(MixtureAggregation):
    r"""Weighted sum (MolPool) aggregation of the graph-level representation:

    .. math::
        \mathbf h = \sum_{c \in C} w_{FP} \mathbf h_c
    """

    def __init__(self, *args, debug=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug = debug
        self._debug_printed = 0  # to avoid spamming

    @property
    def output_dim(self) -> int:
        return sum(self.fp_dims[group[0]] for group in self.groups)

    def forward(
        self, H_vs: list[Tensor], bmgs: list[BatchComponentMolGraph | BatchMolGraph]
    ) -> Tensor:
        super().forward(H_vs, bmgs)

        combined_Hs = []
        for group_idx, group in enumerate(self.groups):
            if len(group) == 1:
                combined_Hs.append(self.Hs[group[0]])
                continue

            # n: num components in group, b: batch size, d: embedding dim
            group_Hs = torch.stack([self.Hs[idx] for idx in group])   # [n, b, d]
            # Be robust to sparse/single-component inference paths where some component
            # weight vectors may still be missing; absent components should contribute 0.
            group_w_fps_list = []
            for idx in group:
                w = self.w_fps[idx]
                if w is None:
                    w = torch.zeros(group_Hs.size(1), dtype=group_Hs.dtype, device=group_Hs.device)
                group_w_fps_list.append(w)
            group_w_fps = torch.stack(group_w_fps_list)  # [n, b]

            # Weighted sum: h_b = sum_c w_c,b * h_c,b
            combined_H = torch.einsum("nb,nbd->bd", group_w_fps, group_Hs)  # [b, d]

            # ---------------- DEBUG BLOCK ----------------
            if self.debug and self._debug_printed < 3:
                b = group_Hs.size(1)
                d = group_Hs.size(2)
                print(f"\n[WeightedSumAggregation] Group {group_idx} = {group}")
                print(f"  group_Hs shape   : {group_Hs.shape}  (n, b, d)")
                print(f"  group_w_fps shape: {group_w_fps.shape}  (n, b)")

                # Check weights for first few batch items
                max_print_b = min(3, b)
                for b_idx in range(max_print_b):
                    w = group_w_fps[:, b_idx].detach().cpu()
                    print(f"  batch {b_idx}: weights = {w.numpy()}  (sum={w.sum().item():.4f})")

                # Verify einsum vs explicit weighted sum
                manual = (group_w_fps.unsqueeze(-1) * group_Hs).sum(dim=0)  # [b, d]
                diff = (manual - combined_H).abs().max().item()
                print(f"  max |manual - einsum| = {diff:.6e}")

                self._debug_printed += 1
            # ---------------- END DEBUG BLOCK ----------------

            combined_Hs.append(combined_H)

        return torch.cat(combined_Hs, dim=1)


class DeepsetsAggregation(MixtureAggregation):
    r"""Deep sets aggregation of the graph-level representation:

    .. math::
        \mathbf h = \mathrm{MLP_{g}}(\sum_{c \in C} \mathrm{MLP_{l}}(\mathbf h_c))
    """
    def __init__(
        self, graph_agg: Aggregation, groups: Sequence[Sequence[int]], fp_dims: Sequence[int],
        mixmp: MixtureMessagePassing | None, *args, **kwargs
    ):
        super().__init__(graph_agg, groups, fp_dims, mixmp, *args, **kwargs)

        self.MLPs_local = nn.ModuleList([])
        self.MLPs_global = nn.ModuleList([])
        for group in groups:
            # TODO: allow to set hparams for MLP by kwargs (e.g., hidden_dim, n_layers)
            hidden_dim = self.fp_dims[group[0]]
            self.MLPs_local.append(
                nn.Sequential(
                nn.Linear(self.fp_dims[group[0]], hidden_dim, bias=False),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim, bias=False),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.fp_dims[group[0]], bias=False),
                )
            )
            self.MLPs_global.append(
                nn.Sequential(
                nn.Linear(self.fp_dims[group[0]], hidden_dim, bias=False),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim, bias=False),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.fp_dims[group[0]], bias=False),
                )
            )

    @property
    def output_dim(self) -> int:
        return sum(self.fp_dims[group[0]] for group in self.groups)

    def forward(
        self, H_vs: list[Tensor], bmgs: list[BatchComponentMolGraph | BatchMolGraph]
    ) -> Tensor:
        super().forward(H_vs, bmgs)

        combined_Hs = []
        for g_idx, group in enumerate(self.groups):
            # use only global MLP if group only has one component, as local MLP would just be nested into global MLP
            if len(group) == 1:
                combined_Hs.append(self.MLPs_global[g_idx](self.Hs[group[0]]))
                continue
            group_w_Hs = torch.stack([self.MLPs_local[g_idx](self.w_fps[idx].unsqueeze(1) * self.Hs[idx]) for idx in group])  # n x b x d
            combined_H = torch.sum(group_w_Hs, dim=0)
            combined_Hs.append(self.MLPs_global[g_idx](combined_H))
        return torch.cat(combined_Hs, 1)

class AttentiveAggregation(MixtureAggregation):
    r"""Attentive aggregation of the graph-level representation:

    .. math::
        \mathbf h = \sum_{c \in C} \alpha_c \mathbf h_c

        \alpha_c = \mathrm{softmax}(\mathbf h_c)
    """
    def __init__(
        self, graph_agg: Aggregation, groups: Sequence[Sequence[int]], fp_dims: Sequence[int],
        mixmp: MixtureMessagePassing | None, *args, **kwargs
    ):
        super().__init__(graph_agg, groups, fp_dims, mixmp, *args, **kwargs)

        self.Ws_a = nn.ModuleList([
            nn.Linear(self.fp_dims[group[0]], 1, bias=False) for group in groups
            ])

    @property
    def output_dim(self) -> int:
        return sum(self.fp_dims[group[0]] for group in self.groups)

    def forward(
        self, H_vs: list[Tensor], bmgs: list[BatchComponentMolGraph | BatchMolGraph]
    ) -> Tensor:
        super().forward(H_vs, bmgs)

        combined_Hs = []
        for g_idx, group in enumerate(self.groups):
            if len(group) == 1:
                combined_Hs.append((self.Hs[group[0]]))
                continue
            w_Hs = torch.stack([self.w_fps[idx].unsqueeze(1) * self.Hs[idx] for idx in group])  # n x b x d
            attention_logits = self.Ws_a[g_idx](w_Hs).exp().squeeze(2) # n x b
            # Ignore logits that correspond to completed zero-tensors due to missing components
            # Create a mask tensor
            mask = torch.zeros_like(attention_logits, dtype=torch.bool)
            # Fill the mask tensor based on index
            for tmp_i, idx in enumerate(group):
                indices = self.Hs_batch[idx]
                if indices is not None:
                    mask[tmp_i, indices] = True
            # Apply the mask to the original tensor
            attention_logits = attention_logits * mask
            Z = torch.sum(attention_logits, dim=0, keepdim=True)
            alphas = attention_logits / Z
            combined_H = torch.sum(alphas.unsqueeze(-1) * w_Hs, dim=0)
            combined_Hs.append(combined_H)
        return torch.cat(combined_Hs, 1)

class Set2SetAggregation(MixtureAggregation):
    r"""Set2Set aggregation of the graph-level representation:

    .. math::
        \mathbf{q}_t &= \mathrm{LSTM}(\mathbf{q}^{*}_{t-1})

        \alpha_{c,t} &= \mathrm{softmax}(\mathbf{h}_c \cdot \mathbf{q}_t)

        \mathbf{r}_t &= \sum_{c=1}^C \alpha_{c,t} \mathbf{h}_c

        \mathbf{q}^{*}_t &= \mathbf{q}_t \, \Vert \, \mathbf{r}_t,

    where :math:`\mathbf{q}^{*}_T` defines the output of the layer with twice
    the dimensionality as the input.

    Note: This implementation follows PyTorch Geometric (cf. https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/aggr/set2set.html#Set2Set) and is based on `"Order Matters: Sequence to sequence for
    Sets" <https://arxiv.org/abs/1511.06391>`_ paper.
    """
    def __init__(
        self, graph_agg: Aggregation, groups: Sequence[Sequence[int]], fp_dims: Sequence[int],
        mixmp: MixtureMessagePassing | None, *args, **kwargs
    ):
        super().__init__(graph_agg, groups, fp_dims, mixmp, *args, **kwargs)

        # TODO: allow to set hparams for Set2Set by kwargs (e.g., processing steps)
        self.processing_steps = 3
        self.lstms = nn.ModuleList([])
        for group in groups:
            in_channels = self.fp_dims[group[0]]
            out_channels = self.fp_dims[group[0]] * 2
            self.lstms.append(
                torch.nn.LSTM(out_channels, in_channels, **kwargs)
                )

    @property
    def output_dim(self) -> int:
        return sum(self.fp_dims[group[0]] * 2 if len(group) > 1 else self.fp_dims[group[0]] for group in self.groups)

    def forward(
        self, H_vs: list[Tensor], bmgs: list[BatchComponentMolGraph | BatchMolGraph]
    ) -> Tensor:
        super().forward(H_vs, bmgs)

        combined_Hs = []
        for g_idx, group in enumerate(self.groups):
            if len(group) == 1:
                combined_Hs.append((self.Hs[group[0]]))
                continue

            w_Hs = torch.stack([self.w_fps[idx].unsqueeze(1) * self.Hs[idx] for idx in group])
            w_Hs = torch.transpose(w_Hs, 0, 1) # b x n x d
            b_dim = w_Hs.size(0)
            d_dim = w_Hs.size(-1)

            h = (w_Hs.new_zeros((self.lstms[g_idx].num_layers, b_dim, d_dim)),
                w_Hs.new_zeros((self.lstms[g_idx].num_layers, b_dim, d_dim)))
            q_star = w_Hs.new_zeros(b_dim, d_dim * 2)

            for _ in range(self.processing_steps):
                q, h = self.lstms[g_idx](q_star.unsqueeze(0), h)

                q = q.squeeze(0) # b x d
                e = torch.sum(w_Hs * q.unsqueeze(1), dim=2) # b x n
                attention_logits = e.exp() #.squeeze(2) # b x n

                # Ignore logits that correspond to completed zero-tensors due to missing components
                # Create a mask tensor
                mask = torch.zeros_like(e, dtype=torch.bool)
                # Fill the mask tensor based on index tensors
                for tmp_i, idx in enumerate(group):
                    indices = self.Hs_batch[idx]
                    if indices is not None:
                        mask[indices, tmp_i] = True
                # Apply the mask to the original tensor
                attention_logits = attention_logits * mask
                Z = torch.sum(attention_logits, dim=1, keepdim=True)
                alphas = attention_logits / Z
                r = torch.sum(w_Hs * alphas.unsqueeze(2), dim=1) # b x d
                q_star = torch.cat([q, r], dim=1) # b x 2*d

            combined_Hs.append(q_star)
        return torch.cat(combined_Hs, 1)

"""### MPNN model for mixtures"""

class MixtureMPNN(MulticomponentMPNN):
    def __init__(
        self,
        message_passing: MulticomponentMessagePassing,
        agg: Aggregation,
        predictor: Predictor,
        mix_mpn: MixtureMessagePassing | None = None,
        batch_norm: bool = False,
        metrics: Iterable[ChempropMetric] | None = None,
        warmup_epochs: int = 2,
        init_lr: float = 1e-4,
        max_lr: float = 1e-3,
        final_lr: float = 1e-4,
        X_d_transform: ScaleTransform | None = None,
    ):
        super().__init__(
            message_passing,
            agg,
            predictor,
            batch_norm,
            metrics,
            warmup_epochs,
            init_lr,
            max_lr,
            final_lr,
            X_d_transform,
        )
        self.agg: MixtureAggregation

    def fingerprint(
        self,
        bmgs: Iterable[BatchComponentMolGraph | BatchMolGraph],
        V_ds: Iterable[Tensor],
        X_d: Tensor | None = None,
    ) -> Tensor:
        H_vs: list[Tensor] = self.message_passing(bmgs, V_ds)
        H = self.agg(H_vs, bmgs)
        H = self.bn(H)
        return H if X_d is None else torch.cat((H, X_d), 1)

    @classmethod
    def _load(cls, path, map_location, **submodules):
        d = torch.load(path, map_location, weights_only=False)

        try:
            hparams = d["hyper_parameters"]
            state_dict = d["state_dict"]
        except KeyError:
            raise KeyError(f"Could not find hyper parameters and/or state dict in {path}.")

        hparams["message_passing"]["blocks"] = [
            block_hparams.pop("cls")(**block_hparams)
            for block_hparams in hparams["message_passing"]["blocks"]
        ]
        graph_agg_hparams = hparams["agg"]["graph_agg"]
        hparams["agg"]["graph_agg"] = graph_agg_hparams.pop("cls")(**graph_agg_hparams)
        mixmp_hparams = hparams["agg"]["mixmp"]
        hparams["agg"]["mixmp"] = mixmp_hparams.pop("cls")(**mixmp_hparams)
        submodules |= {
            key: hparams[key].pop("cls")(**hparams[key])
            for key in ("message_passing", "agg", "predictor")
            if key not in submodules
        }

        if not hasattr(submodules["predictor"].criterion, "_defaults"):
            submodules["predictor"].criterion = submodules["predictor"].criterion.__class__(
                task_weights=submodules["predictor"].criterion.task_weights
            )

        return submodules, state_dict, hparams
