import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Union


class DomainDiscriminator:
    def __init__(self, config: dict):
        super().__init__()

        self.loss = nn.CrossEntropyLoss()

        self.query = nn.Linear(
            in_features=config.get('n_features'),
            out_features=config.get('n_features'),
        )
        self.key = nn.Linear(
            in_features=config.get('n_features'),
            out_features=config.get('n_features'),
        )

    def forward(self, query: Tensor, key: Tensor) -> Tuple[Tensor, Tensor]:
        domain_query = self.query(query)
        domain_key = self.key(key)
        loss = self.loss(domain_query, domain_key)

        return loss
