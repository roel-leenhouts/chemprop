import logging

from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors
import multiprocess
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Mol
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import torch

from model_core.featurizers.base import VectorFeaturizer
from model_core.utils import ClassRegistry

logger = logging.getLogger(__name__)

MoleculeFeaturizerRegistry = ClassRegistry[VectorFeaturizer[Mol]]()


class MorganFeaturizerMixin:
    def __init__(self, radius: int = 2, length: int = 2048, include_chirality: bool = True):
        if radius < 0:
            raise ValueError(f"arg 'radius' must be >= 0! got: {radius}")

        self.length = length
        self.F = GetMorganGenerator(
            radius=radius, fpSize=length, includeChirality=include_chirality
        )

    def __len__(self) -> int:
        return self.length


class BinaryFeaturizerMixin:
    def __call__(self, mol: Chem.Mol) -> np.ndarray:
        return self.F.GetFingerprintAsNumPy(mol)


class CountFeaturizerMixin:
    def __call__(self, mol: Chem.Mol) -> np.ndarray:
        return self.F.GetCountFingerprintAsNumPy(mol).astype(np.int32)


@MoleculeFeaturizerRegistry("morgan_binary")
class MorganBinaryFeaturizer(MorganFeaturizerMixin, BinaryFeaturizerMixin, VectorFeaturizer[Mol]):
    pass


@MoleculeFeaturizerRegistry("morgan_count")
class MorganCountFeaturizer(MorganFeaturizerMixin, CountFeaturizerMixin, VectorFeaturizer[Mol]):
    pass


@MoleculeFeaturizerRegistry("rdkit_2d")
class RDKit2DFeaturizer(VectorFeaturizer[Mol]):
    def __init__(self):
        if multiprocess.current_process().name == "MainProcess":
            logger.warning(
                "The RDKit 2D features can deviate signifcantly from a normal distribution. Consider "
                "manually scaling them using an appropriate scaler before creating datapoints, rather "
                "than using the scikit-learn `StandardScaler` (the default in Chemprop)."
            )

    def __len__(self) -> int:
        return len(Descriptors.descList)

    def __call__(self, mol: Chem.Mol) -> np.ndarray:
        features = np.array(
            [
                0.0 if name == "SPS" and mol.GetNumHeavyAtoms() == 0 else func(mol)
                for name, func in Descriptors.descList
            ],
            dtype=float,
        )

        return features


class V1RDKit2DFeaturizerMixin(VectorFeaturizer[Mol]):
    def __len__(self) -> int:
        return 200

    def __call__(self, mol: Mol) -> np.ndarray:
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        features = self.generator.process(smiles)[1:]

        return np.array(features)


@MoleculeFeaturizerRegistry("v1_rdkit_2d")
class V1RDKit2DFeaturizer(V1RDKit2DFeaturizerMixin):
    def __init__(self):
        self.generator = rdDescriptors.RDKit2D()


@MoleculeFeaturizerRegistry("v1_rdkit_2d_normalized")
class V1RDKit2DNormalizedFeaturizer(V1RDKit2DFeaturizerMixin):
    def __init__(self):
        self.generator = rdNormalizedDescriptors.RDKit2DNormalized()


@MoleculeFeaturizerRegistry("rdkit2dnormalized")
@MoleculeFeaturizerRegistry("rdkit_2d_normalized")
class RDKit2DNormalizedFeaturizer(V1RDKit2DNormalizedFeaturizer):
    """Alias of the Descriptastorus RDKit2DNormalized descriptor set."""


@MoleculeFeaturizerRegistry("molt5")
@MoleculeFeaturizerRegistry("mol_t5")
@MoleculeFeaturizerRegistry("MolT5")
class MolT5Featurizer(VectorFeaturizer[Mol]):
    """Fixed molecular descriptors from MolFeat's pretrained MolT5 transformer."""

    def __init__(
        self,
        notation: str = "smiles",
        hf_model_name: str = "laituan245/molt5-base",
        fallback_to_hf: bool = True,
    ):
        self.transformer = None
        self._hf_tokenizer = None
        self._hf_model = None
        self._hf_device = torch.device("cpu")

        try:
            from molfeat.trans.pretrained.hf_transformers import PretrainedHFTransformer

            self.transformer = PretrainedHFTransformer(kind="MolT5", notation=notation, dtype=float)
        except Exception as e:
            if not fallback_to_hf:
                raise
            logger.warning(
                "MolFeat MolT5 loader failed (%s). Falling back to direct Hugging Face model '%s'.",
                str(e),
                hf_model_name,
            )
            try:
                from transformers import AutoModel, AutoTokenizer
            except ImportError as e2:
                raise ImportError(
                    "MolT5 featurizer requires either molfeat model-store access or "
                    "`transformers`/`tokenizers` for direct HF fallback."
                ) from e2

            self._hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
            self._hf_model = AutoModel.from_pretrained(hf_model_name).to(self._hf_device).eval()

        self._size: int | None = None

    def __len__(self) -> int:
        if self._size is None and self._hf_model is not None:
            d = getattr(self._hf_model.config, "hidden_size", None)
            if d is None:
                d = getattr(self._hf_model.config, "d_model", None)
            if d is not None:
                self._size = int(d)
        if self._size is None:
            # Infer descriptor width lazily from a valid test molecule.
            self._size = len(self(Chem.MolFromSmiles("C")))
        return self._size

    def __call__(self, mol: Mol) -> np.ndarray:
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        if self.transformer is not None:
            feat = self.transformer([smiles])
            feat = np.asarray(feat, dtype=float)
            if feat.ndim == 2:
                feat = feat[0]
            elif feat.ndim > 2:
                feat = feat.reshape(-1)
        else:
            toks = self._hf_tokenizer([smiles], return_tensors="pt", padding=True, truncation=True).to(
                self._hf_device
            )
            with torch.no_grad():
                if hasattr(self._hf_model, "encoder"):
                    out = self._hf_model.encoder(
                        input_ids=toks["input_ids"],
                        attention_mask=toks["attention_mask"],
                    )
                else:
                    out = self._hf_model(**toks)
                hidden = out.last_hidden_state  # [1, L, D]
                mask = toks["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            feat = pooled.squeeze(0).cpu().numpy().astype(float, copy=False)

        if self._size is None:
            self._size = int(feat.shape[0])

        return feat


@MoleculeFeaturizerRegistry("charge")
class ChargeFeaturizer(VectorFeaturizer[Mol]):
    def __call__(self, mol: Chem.Mol) -> np.ndarray:
        return np.array([Chem.GetFormalCharge(mol)])

    def __len__(self) -> int:
        return 1
