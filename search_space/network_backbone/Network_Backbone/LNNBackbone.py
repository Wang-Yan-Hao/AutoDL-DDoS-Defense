from typing import Any, Dict, List, Optional, Tuple

import ConfigSpace as CS
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformIntegerHyperparameter
)

import torch
from torch import nn

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.network_backbone.base_network_backbone import NetworkBackboneComponent
from autoPyTorch.pipeline.components.setup.network_backbone.utils import _activations
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter


class LNNBackbone(NetworkBackboneComponent):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def build_backbone(self, input_shape: Tuple[int, ...]):
        return LightweightNN(self.config)
    
    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None) -> Dict[str, Any]:
        return {
            'shortname': 'LNNBackbone',
            'name': 'LNNBackbone',
            'handles_tabular': True,
            'handles_image': False,
            'handles_time_series': False,
        }

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        num_kernels: HyperparameterSearchSpace = HyperparameterSearchSpace(
                                                        hyperparameter='num_kernels',
                                                        value_range=(8, 32),
                                                        default_value=16,
                                                        ),
        kernel_size: HyperparameterSearchSpace = HyperparameterSearchSpace(
                                                    hyperparameter='kernel_size',
                                                    value_range=tuple([3, 5, 7]),
                                                    default_value=3,
                                                    ),
        kernel_growth_ratio: HyperparameterSearchSpace = HyperparameterSearchSpace(
                                                            hyperparameter='kernel_growth_ratio',
                                                            value_range=tuple([2, 4, 8]),
                                                            default_value=4,
                                                            ),
        num_unitb: HyperparameterSearchSpace = HyperparameterSearchSpace(
                                                    hyperparameter='num_unitb',
                                                    value_range=(1,4),
                                                    default_value=2,
                                                    ),                                                
    ) -> ConfigurationSpace:
        cs = CS.ConfigurationSpace()

        add_hyperparameter(cs, num_kernels, UniformIntegerHyperparameter)
        add_hyperparameter(cs, kernel_size, CategoricalHyperparameter)
        add_hyperparameter(cs, kernel_growth_ratio, CategoricalHyperparameter)
        add_hyperparameter(cs, num_unitb, UniformIntegerHyperparameter)

        return cs
    
class UnitB(nn.Module):
    def __init__(self, num_channels: int, kernel_size: int):
        super().__init__()

        branch_channel = num_channels//2

        self.branch1 = nn.Sequential(
            # Point-wise conv
            nn.Conv1d(branch_channel, num_channels, 1),
            nn.BatchNorm1d(num_channels),
            nn.SELU(inplace=True),

            # Depth-wise conv
            nn.Conv1d(num_channels, num_channels, kernel_size, padding="same",
                      groups=num_channels, bias=False),
            nn.BatchNorm1d(num_channels),

            # standard conv
            nn.Conv1d(num_channels, branch_channel, 1, padding="same"),
            nn.BatchNorm1d(branch_channel),
            nn.SELU(inplace=True),
        )
          
    def _concat(self, branch1: torch.FloatTensor, 
                branch2: torch.FloatTensor) -> torch.FloatTensor:
        # Concatenate along channel axis
        return torch.cat((branch1, branch2), 1)

    def _channel_shuffle(self, x: torch.FloatTensor, 
                         groups: int) -> torch.FloatTensor:
        batchsize, num_channels, features = x.shape

        channels_per_group = num_channels // groups
        
        # reshape
        x = x.view(batchsize, groups, channels_per_group, features)

        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        x = x.view(batchsize, -1, features)

        return x
    
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x1 = x[:, :(x.shape[1]//2), :]
        x2 = x[:, (x.shape[1]//2):, :]
        out = self._concat(x1, self.branch1(x2))

        return self._channel_shuffle(out, 2)

class LightweightNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_grown_kernels = self.config["num_kernels"] * self.config["kernel_growth_ratio"]

        self.conv1 = nn.Conv1d(1, self.config["num_kernels"], self.config["kernel_size"], padding="same")
        self.bn1 = nn.BatchNorm1d(self.config["num_kernels"])
        self.selu1 = nn.SELU(inplace=True)
        
        self.unit_a = self._unit_a(
            self.config["num_kernels"], 
            self.num_grown_kernels
        )

        unit_bi = list()
        for i in range(self.config["num_unitb"]):
            unit_bi.append(
                UnitB(self.num_grown_kernels, self.config["kernel_size"])
            )
        self.unit_b = nn.Sequential(*unit_bi)

        self.globalpool = nn.AdaptiveAvgPool1d(1)

    def _unit_a(self, in_channel: int, out_channel: int) -> nn.Sequential:
        return nn.Sequential(
            # Point-wise conv
            nn.Conv1d(in_channel, out_channel, 1),
            nn.BatchNorm1d(out_channel),
            nn.SELU(inplace=True),

            # Depth-wise conv
            nn.Conv1d(out_channel, out_channel, self.config["kernel_size"], 
                      stride=2, groups=out_channel, bias=False),
            nn.BatchNorm1d(out_channel),

            # standard conv
            nn.Conv1d(out_channel, out_channel, 1),
            nn.BatchNorm1d(out_channel),
            nn.SELU(inplace=True),
        )

    def forward(self, x: torch.FloatTensor):
        # Due to the input shape would be the tabular task format
        # we may need to refit the shape of input
        batch, features = x.size()
        x = x.view(batch, 1, features)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.selu1(x)
        x = self.unit_a(x)
        x = self.unit_b(x)
        x = self.globalpool(x)

        x = torch.flatten(x, start_dim=1)
        return x