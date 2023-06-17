import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Union


def exponential_scaled_dot_product(query: Tensor, key: Tensor) -> Tensor:
    attention = query @ key.permute(0, 2, 1)
    seq_len, n_features = query.shape[1:]
    correction = (
        query[..., :seq_len] *\
        torch.diagonal(key, dim1=1, dim2=2).unsqueeze(-1).permute(0, 2, 1)
    )
    attention -= correction
    attention = torch.exp(attention / np.sqrt(n_features))
    
    divider = (
        attention.sum(axis=1, keepdim=True) -\
        torch.diagonal(attention, dim1=1, dim2=2).unsqueeze(-1).permute(0, 2, 1)
    ).permute(0, 2, 1)

    return attention / divider


class SharedAttention(nn.Module):
    def __init__(self, n_features: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        
        self.query = nn.Linear(
            in_features=n_features,
            out_features=n_features,
        )
        self.key = nn.Linear(
            in_features=n_features,
            out_features=n_features,
        )

    def forward(
        self, pattern: Tensor, value: Tensor
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Shared Attention

        Args:
            pattern (Tensor): pattern embeddings with shape (batch, seq_len, n_faetures).
            value (Tensor): values with shape (batch, seq_len, n_features).

        Returns:
            Tuple[Tensor, Tuple[Tensor, Tensor]]:
                - representations with shape (batch, seq_len + 1, n_features).
                - queries and keys with shape (batch, seq_len, n_features).
        """
        query, key = self.query(pattern), self.key(pattern)


        # interpolation mode
        rep_in = self.mlp_o(self.calc_attn(query, key, value))
        # extrapolation mode
        rep_ex = self.mlp_o(
            self.calc_attn(query, key, value, kernel_size=self.kernel_size)
        )
        return torch.cat([rep_in, rep_ex], dim=-1), (query, key)

    def interpolation(self, query, key):
        return exponential_scaled_dot_product(query, key)
    
    def extrapolation(self, query, key):




class SequenceGenerator(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
