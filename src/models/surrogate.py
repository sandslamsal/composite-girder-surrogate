"""Residual-MLP surrogate for composite-girder moment-curvature response.

Predicts the two non-trivial section quantities (y_na, curvature) given a
17-dim normalised feature vector (14 continuous + 3 one-hot section-type
indicators). Width 256, 5 residual blocks, GELU + dropout 0.1.
"""
from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.lin1 = nn.Linear(width, width)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.lin2 = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lin1(x)
        h = self.act(h)
        h = self.drop(h)
        h = self.lin2(h)
        return x + h


class CompositeGirderSurrogate(nn.Module):
    """Outputs (y_na, curvature), both >= 0 via Softplus."""

    def __init__(
        self,
        input_dim: int = 17,
        output_dim: int = 2,
        width: int = 256,
        n_blocks: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.stem = nn.Linear(input_dim, width)
        self.stem_act = nn.GELU()
        self.blocks = nn.ModuleList(
            [ResidualBlock(width, dropout) for _ in range(n_blocks)]
        )
        self.head = nn.Linear(width, output_dim)
        # Softplus keeps both outputs non-negative; predictions live in the
        # normalised [0, 1]-ish range. Default beta=1 is more numerically stable
        # on MPS than beta=2 (which can yield NaNs in mixed-precision-ish paths).
        self.out_act = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem_act(self.stem(x))
        for blk in self.blocks:
            h = blk(h)
        raw = self.head(h)
        return self.out_act(raw)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
