import torch
from torch import Tensor, nn

from model_core.data import BatchMolGraph
from model_core.nn.message_passing.proto import MessagePassing
from model_core.nn.transforms import GraphTransform, ScaleTransform


class DescriptorOnlyMessagePassing(MessagePassing):
    """No-op message passing used for descriptor-only FFN models."""

    def __init__(
        self,
        output_dim: int = 0,
        V_d_transform: ScaleTransform | None = None,
        graph_transform: GraphTransform | None = None,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.V_d_transform = V_d_transform if V_d_transform is not None else nn.Identity()
        self.graph_transform = graph_transform if graph_transform is not None else nn.Identity()
        self.hparams = {
            "cls": self.__class__,
            "output_dim": output_dim,
            "V_d_transform": V_d_transform,
            "graph_transform": graph_transform,
        }

    def forward(self, bmg: BatchMolGraph, V_d: Tensor | None = None) -> Tensor:
        bmg = self.graph_transform(bmg)
        n_vertices = 0 if bmg.V is None else len(bmg.V)

        if bmg.V is not None:
            device = bmg.V.device
            dtype = bmg.V.dtype
        elif V_d is not None:
            device = V_d.device
            dtype = V_d.dtype
        else:
            device = torch.device("cpu")
            dtype = torch.float32

        return torch.zeros((n_vertices, self.output_dim), dtype=dtype, device=device)
