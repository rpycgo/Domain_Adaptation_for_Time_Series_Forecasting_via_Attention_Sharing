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
    def __init__(self, feat_dim: int, hidden_dim: int, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size

        self.query = nn.Linear(
            in_features=feat_dim, out_features=feat_dim
        )
        self.key = nn.Linear(
            in_features=feat_dim, out_features=feat_dim
        )

    def forward(
        self, pattern: Tensor, value: Tensor
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Shared Attention

        Args:
            pattern (Tensor): pattern embeddings with shape `(B, M, T)`.
            value (Tensor): values with shape `(B, D, T)`.

        Returns:
            Tuple[Tensor, Tuple[Tensor, Tensor]]:
                - representations with shape `(B, D, T + 1)`.
                - queries and keys with shape `(B, D, T)`.
        """
        query, key = self.query(pattern), self.key(pattern)


        # interpolation mode
        rep_in = self.mlp_o(self.calc_attn(query, key, value))
        # extrapolation mode
        rep_ex = self.mlp_o(
            self.calc_attn(query, key, value, kernel_size=self.kernel_size)
        )
        return torch.cat([rep_in, rep_ex], dim=-1), (query, key)

    def calc_attn(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        kernel_size: Union[int, Tuple[int, int]] = None,
    ) -> Tensor:
        """Calculate attention value

        Args:
            query (Tensor): queries with shape `(B, D, T)`.
            key (Tensor): keys with shape `(B, D, T)`.
            value (Tensor): values with shape `(B, D, T)`.
            kernel_size (Union[int, Tuple[int, int]], optional): kernel size used in encoder,
                required in extrapolation mode.

        Returns:
            Tensor: attention values with shape `(B, D, T + 1)`.
        """
        if isinstance(kernel_size, tuple):
            kernel_size = max(kernel_size)
        if kernel_size:
            unit_len = (kernel_size - 1) // 2
            query = query[..., -1 - unit_len : -1 - unit_len + 1]
            # ? why k, v shape = [..., 96]..?
            key = key[..., kernel_size - unit_len - 1 : -1 - unit_len - 1]
            value = value[..., kernel_size:-1]
        attn_logits = torch.exp(
            query.transpose(1, 2) @ key
            - (query.transpose(1, 2) @ key)
            * torch.eye(query.shape[-1], device=query.device)
            / math.sqrt(query.shape[-1])
        )
        attn_scores = torch.softmax(attn_logits, dim=-1)
        attn_values = (attn_scores @ value.transpose(1, 2)).transpose(1, 2)
        return attn_values


class SequenceGenerator(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size