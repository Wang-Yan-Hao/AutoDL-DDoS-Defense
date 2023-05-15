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


class CDCNNBackbone(NetworkBackboneComponent):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def build_backbone(self, input_shape: Tuple[int, ...]):
        return CDCNN(self.config)
    
    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None) -> Dict[str, Any]:
        return {
            'shortname': 'CDCNNBackbone',
            'name': 'CDCNNBackbone',
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
        num_gconv: HyperparameterSearchSpace = HyperparameterSearchSpace(
                                                    hyperparameter='num_gconv',
                                                    value_range=(1,4),
                                                    default_value=2,
                                                    ),                                              
        reduction_ratio: HyperparameterSearchSpace = HyperparameterSearchSpace(
                                                    hyperparameter='reduction_ratio',
                                                    value_range=(8,32),
                                                    default_value=16,
                                                    ), 
    ) -> ConfigurationSpace:
        cs = CS.ConfigurationSpace()

        add_hyperparameter(cs, num_kernels, UniformIntegerHyperparameter)
        add_hyperparameter(cs, kernel_size, CategoricalHyperparameter)
        add_hyperparameter(cs, kernel_growth_ratio, CategoricalHyperparameter)
        add_hyperparameter(cs, num_gconv, UniformIntegerHyperparameter)
        add_hyperparameter(cs, reduction_ratio, UniformIntegerHyperparameter)

        return cs


class SELayer(nn.Module):
    def __init__(self, channel, reduction):
        super().__init__()
        hidden_neurons = channel // 2 if channel < reduction else channel // reduction

        self.globalpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, hidden_neurons, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_neurons, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.globalpool(x).view(b, -1)
        y = self.fc(y).view(b, c, -1)
        return x * y.expand_as(x)


class CDCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_grown_kernels = self.config["num_kernels"] * self.config["kernel_growth_ratio"]

        self.conv1 = nn.Conv1d(1, self.config["num_kernels"], 
                               self.config["kernel_size"],
                               padding="same")
        self.bn1 = nn.BatchNorm1d(self.config["num_kernels"])
        self.selu1 = nn.SELU(inplace=True)
        self.SE1 = SELayer(self.config["num_kernels"], self.config["reduction_ratio"])
        
        self.conv2 = nn.Conv1d(self.config["num_kernels"], 
                               self.config["num_kernels"], 
                               self.config["kernel_size"],
                               padding="same")
        self.bn2 = nn.BatchNorm1d(self.config["num_kernels"])
        self.selu2 = nn.SELU(inplace=True)
        self.SE2 = SELayer(self.config["num_kernels"], self.config["reduction_ratio"])
        
        self.gconv1 = self._gconv(self.config["num_kernels"], self.num_grown_kernels)
        self.SE3 = SELayer(self.num_grown_kernels, self.config["reduction_ratio"])

        gconvi = list()
        for i in range(1, self.config["num_gconv"]):
            gconvi.append(self._gconv(self.num_grown_kernels, 
                          self.num_grown_kernels))
            gconvi.append(
                SELayer(self.num_grown_kernels, self.config["reduction_ratio"])
            )
        self.gconv = nn.Sequential(*gconvi)

        self.globalpool = nn.AdaptiveAvgPool1d(1)

    def _gconv(self, in_channel: int, out_channel: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(in_channel, out_channel, self.config["kernel_size"], 
                      padding="same"),
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
        x = self.SE1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.selu2(x)
        x = self.SE2(x)
        x = self.gconv1(x)
        x = self.SE3(x)
        x = self.gconv(x)
        x = self.globalpool(x)

        x = torch.flatten(x, start_dim=1)
        return x