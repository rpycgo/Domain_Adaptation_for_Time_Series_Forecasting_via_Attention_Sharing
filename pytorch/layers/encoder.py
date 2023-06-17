import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Union


class PrivateEncoders(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.encoder_config = config.get('encoder')

        self.temporal_convolutions = nn.ModuleList()
        for kernel_size in self.encoder_config.get('temporal_convolutions').get('kernel_sizes'):
            self.temporal_convolutions.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=self.encoder_config.get('temporal_convolutions').get('out_channels'),
                    kernel_size=kernel_size,
                    stride=self.encoder_config.get('temporal_convolutions').get('stride'),
                    dilation=self.encoder_config.get('temporal_convolutions').get('dilation'),
                )
            )

        self.value = nn.Linear(
            in_features=in_features,
            out_features=sum([
                ((in_channels - kernel_size) + 1) // self.encoder_config.get('temporal_convolutions').get('stride')
                for kernel_size
                in self.encoder_config.get('temporal_convolutions').get('kernel_sizes')
            ])
        )

    def forward(self, x):
        temporal_convolution_output = torch.concat(
            [layer(x.permute(0, 2, 1)) for layer in self.temporal_convolutions],
            axis=-1
        )
        value = self.value(x)

        return temporal_convolution_output.permute(0, 2, 1), value
