import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import pprint
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from ray.tune.examples.mnist_pytorch import get_data_loaders

from ray import air, tune
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.schedulers.pb2 import PB2
from ray.tune import CLIReporter


search_config = {
    "data_loader" : {"batch_size": tune.randint(32,320),},
    "coalescer:__choice__" : "NoCoalescer",
    "feature_preprocessor:__choice__" : tune.grid_search
    ([
        {"ExtraTreesPreprocessorClassification" : 
        {
            "bootstrap" : tune.choice(["True", "False"]),
            "criterion" : tune.choice(["gini", "entropy"]),
            "max_depth" : None,
            "max_features" : tune.uniform(0,1),
            "max_leaf_nodes" : None,
            "min_impurity_decrease" : None,
            "min_samples_leaf" : tune.randint(1,20),
            "min_samples_split" : tune.randint(2,20),
            "min_weight_fraction_leaf" : 0,
            "n_estimators" : tune.randint(10,100),
        }},
        {"FastICA" : 
        {
            "algorithm" : tune.choice(["parallel", "deflation"]),
            "fun" : tune.choice(["logcosh", "exp", "cube"]),
            "n_components" : tune.randint(39,71),
            "whiten" : tune.choice(["True", "False"]),
        }},
        {"FeatureAgglomeration" : 
        {
            "affinity" : tune.choice(["euclidean","manhattan", "cosine"]),
            "linkage" : tune.choice(["ward", "complete", "average"]),
            "n_clusters" : tune.randint(39,71),
            "pooling_func" : tune.choice(["mean", "median", "max"]),
        }},
        {"KernelPCA" :
        {
            "coef0" : tune.uniform(-1,1),
            "degree" : tune.lograndint(2,5),
            "gamma" : tune.loguniform(3.0517578125e-05, 8),
            "kernel" : tune.choice(["poly", "rbf", "sigmoid", "cosine"]),
            "n_components" : tune.randint(39,71),
        }},
        {"LibLinearSVCPreprocessor" : 
        {
            "C" : tune.loguniform(0.03125, 32768),
            "dual" : tune.choice(["False"]),
            "fit_intercept" : tune.choice(["True"]),
            "intercept_scaling" : 1,
            "loss" : tune.choice(["squared_hinge", "hinge"]),
            "multi_class" : tune.choice(["ovr"]),
            "penalty" : tune.choice(["1l"]),
            "tol" : tune.loguniform(1e-5,0.1),
        }},
        {"Nystroem" :
        {
            "coef0" : tune.uniform(-1,1),
            "degree" : tune.lograndint(2,5),
            "gamma" : tune.loguniform(3.0517578125e-5,8),
            "kernel" : tune.choice(["poly", "rbf", "sigmoid", "cosine"]),
            "n_components" : tune.randint(39,71),
        }},
        {"PCA" : 
        {
            "keep_variance" : tune.uniform(0.5,0.9999),
            "whiten" : tune.choice(["True", "False"]),
        }},
        {"PolynomialFeatures" :
        {
            "degree" : tune.randint(2,3),
            "include_bias" : tune.choice(["True", "False"]),
            "interaction_only" : tune.choice(["True", "False"]),
        }},
        {"RandomKitchenSinks" : 
        {
            "gamma" : tune.loguniform(3.0517578125e-5,8),
            "n_components" : tune.randint(39,71),
        }},
        {"RandomTreesEmbedding" :
        {
            "max_depth" : tune.randint(2,10),
            "max_leaf_nodes" : None,
            "min_samples_leaf" : tune.randint(1,20),
            "min_samples_split" : tune.randint(2,20),
            "n_estimators" : tune.randint(10,100),
        }},
        {"SelectPercentileClassification" : 
        {
            "percentile" : tune.randint(1,99),
            "score_func" : tune.choice(["f_classif"]),
        }},
        {"SelectRatesClassification" : 
        {
            "alpha" : tune.uniform(0.01, 0.5),
            "mode" : tune.choice(["fpr", "fdr", "fwe", "percentile"]),
            "score_func" : tune.choice(["f_classif"]),
        }},
        {"TruncatedSVD" : {"target_dim" : tune.randint(39,71),}},
    ]),
    "encoder:__choice__" : "NoEncoder",
    "imputer" : {"numerical_strategy" : tune.choice(["mean", "median", "most_frequent", "constant_zero",])},
    "lr_scheduler:__choice__" : tune.grid_search
    ([
        {"CosineAnnealingLR" : {"T_max" : tune.randint(10,500)}},
        {"CosineAnnealingWarmRestarts" :
        {
            "T_0" : tune.randint(1,20),
            "T_mult" : tune.uniform(1,2),
        }},
        {"CyclicLR" : 
        {
            "base_lr" : tune.uniform(1e-6, 0.1),
            "max_lr" : tune.uniform(0.001, 0.1),
            "mode" : tune.choice(["triangular", "triangular2", "exp_range"]),
            "step_size_up" : tune.randint(1000,4000),
        }},
        {"ExponentialLR" : {"gamma" : tune.uniform(0.7,0.9999)}},
        {"ReduceLROnPlateau" :
        {
            "factor" : tune.uniform(0.01, 0.9),
            "mode" : tune.choice(["min", "max"]),
            "patience" : tune.randint(5,20),
        }},
        {"StepLR" :
        {
            "gamma" : tune.uniform(0.001, 0.9),
            "step_size" : tune.randint(1, 10),
        }},
    ]),
    "network_backbone:__choice__" : tune.grid_search
    ([
        {"CDCNNBackbone" : 
        {
            "kernel_growth_ratio" : tune.choice([2,4,8]),
            "kernel_size" : tune.choice([3,5,7]),
            "num_gconv" : tune.randint(1,4),
            "num_kernels" : tune.randint(8,32),
            "reduction_ratio" : tune.randint(8,32),
        }},
        {"CLNNBackbone" : 
        {
            "kernel_growth_ratio" : tune.choice([2,4,8]),
            "kernel_size" : tune.choice([3,5,7]),
            "num_kernels" : tune.randint(8,32),
            "num_unitb" : tune.randint(1,4),
        }}, 
        {"DCNNBackbone" : 
        {
            "kernel_growth_ratio" : tune.choice([2,4,8]),
            "kernel_size" : tune.choice([3,5,7]),
            "num_kernels" : tune.randint(8,32),
            "num_sconv" : tune.randint(1,4),
            "reduction_ratio" : tune.randint(8,32),
        }},
        {"LNNBackbone" : 
        {
            "kernel_growth_ratio" : tune.choice([2,4,8]),
            "kernel_size" : tune.choice([3,5,7]),
            "num_kernels" : tune.randint(8,32),
            "num_unitb" : tune.randint(1,4),
        }},
        {"MLPBackbone" : 
        {
            "activation" : tune.choice(["relu", "tanh", "sigmoid"]),
            "dropout_1" : tune.uniform(0, 0.8),
            "dropout_2" : tune.uniform(0, 0.8),
            "dropout_3" : tune.uniform(0, 0.8),
            "dropout_4" : tune.uniform(0, 0.8),
            "dropout_5" : tune.uniform(0, 0.8),
            "dropout_6" : tune.uniform(0, 0.8),
            "dropout_7" : tune.uniform(0, 0.8),
            "dropout_8" : tune.uniform(0, 0.8),
            "dropout_9" : tune.uniform(0, 0.8),
            "dropout_10" : tune.uniform(0, 0.8),
            "dropout_11" : tune.uniform(0, 0.8),
            "dropout_12" : tune.uniform(0, 0.8),
            "dropout_13" : tune.uniform(0, 0.8),
            "dropout_14" : tune.uniform(0, 0.8),
            "dropout_15" : tune.uniform(0, 0.8),
            "num_groups" : tune.randint(1, 15),
            "num_units_1" : tune.randint(10, 1024),
            "num_units_2" : tune.randint(10, 1024),
            "num_units_3" : tune.randint(10, 1024),
            "num_units_4" : tune.randint(10, 1024),
            "num_units_5" : tune.randint(10, 1024),
            "num_units_6" : tune.randint(10, 1024),
            "num_units_7" : tune.randint(10, 1024),
            "num_units_8" : tune.randint(10, 1024),
            "num_units_9" : tune.randint(10, 1024),
            "num_units_10" : tune.randint(10, 1024),
            "num_units_11" : tune.randint(10, 1024),
            "num_units_12" : tune.randint(10, 1024),
            "num_units_13" : tune.randint(10, 1024),
            "num_units_14" : tune.randint(10, 1024),
            "num_units_15" : tune.randint(10, 1024),
            "use_dropout" : tune.choice(["True", "False"]),
        }},
        {"ResNetBackbone" : 
        {
            "activation" : tune.choice(["relu", "tanh", "sigmoid"]),
            "blocks_per_group_0" : tune.randint(1,4),
            "blocks_per_group_1" : tune.randint(1,4),
            "blocks_per_group_2" : tune.randint(1,4),
            "blocks_per_group_3" : tune.randint(1,4),
            "blocks_per_group_4" : tune.randint(1,4),
            "blocks_per_group_5" : tune.randint(1,4),
            "blocks_per_group_6" : tune.randint(1,4),
            "blocks_per_group_7" : tune.randint(1,4),
            "blocks_per_group_8" : tune.randint(1,4),
            "blocks_per_group_9" : tune.randint(1,4),
            "blocks_per_group_10" : tune.randint(1,4),
            "blocks_per_group_11" : tune.randint(1,4),
            "blocks_per_group_12" : tune.randint(1,4),
            "blocks_per_group_13" : tune.randint(1,4),
            "blocks_per_group_14" : tune.randint(1,4),
            "blocks_per_group_15" : tune.randint(1,4),
            "dropout_0" : tune.uniform(0, 0.8),
            "dropout_1" : tune.uniform(0, 0.8),
            "dropout_2" : tune.uniform(0, 0.8),
            "dropout_3" : tune.uniform(0, 0.8),
            "dropout_4" : tune.uniform(0, 0.8),
            "dropout_5" : tune.uniform(0, 0.8),
            "dropout_6" : tune.uniform(0, 0.8),
            "dropout_7" : tune.uniform(0, 0.8),
            "dropout_8" : tune.uniform(0, 0.8),
            "dropout_9" : tune.uniform(0, 0.8),
            "dropout_10" : tune.uniform(0, 0.8),
            "dropout_11" : tune.uniform(0, 0.8),
            "dropout_12" : tune.uniform(0, 0.8),
            "dropout_13" : tune.uniform(0, 0.8),
            "dropout_14" : tune.uniform(0, 0.8),
            "dropout_15" : tune.uniform(0, 0.8),
            "max_shake_drop_probability" : tune.uniform(0,1),
            "num_groups" : tune.randint(1,15),
            "num_units_0" : tune.randint(10,1024),
            "num_units_1" : tune.randint(10,1024),
            "num_units_2" : tune.randint(10,1024),
            "num_units_3" : tune.randint(10,1024),
            "num_units_4" : tune.randint(10,1024),
            "num_units_5" : tune.randint(10,1024),
            "num_units_6" : tune.randint(10,1024),
            "num_units_7" : tune.randint(10,1024),
            "num_units_8" : tune.randint(10,1024),
            "num_units_9" : tune.randint(10,1024),
            "num_units_10" : tune.randint(10,1024),
            "num_units_11" : tune.randint(10,1024),
            "num_units_12" : tune.randint(10,1024),
            "num_units_13" : tune.randint(10,1024),
            "num_units_14" : tune.randint(10,1024),
            "num_units_15" : tune.randint(10,1024),
            "use_dropout" : tune.choice(["True", "False"]),
            "use_shake_drop" : tune.choice(["True", "False"]),
            "use_shake_shake" : tune.choice(["True", "False"]),
        }},
        {"ShapedMLPBackbone" : 
        {
            "activation" : tune.choice(["relu", "tanh", "sigmoid"]),
            "max_dropout" : tune.uniform(0, 1),
            "max_unit" : tune.randint(10, 1024),
            "mlp_shape" : tune.choice(["funnel", "long_funnel", "diamond", "hexagon", "brick", "triangle", "stairs"]),
            "num_groups" : tune.randint(1,15),
            "output_dim" : tune.randint(10,1024),
            "use_dropout" : tune.choice(["True", "False"]),
        }},
        {"ShapedResNetBackbone" : 
        {
            "activation" : tune.choice(["relu", "tanh", "sigmoid"]),
            "blocks_per_group" : tune.randint(1, 4),
            "max_dropout" : tune.uniform(0, 0.8),
            "max_shake_drop_probability" : tune.uniform(0,1),
            "max_unit" : tune.randint(10, 1024),
            "num_groups" : tune.randint(1,15),
            "output_dim" : tune.randint(10,1024),
            "resnet_shape" : tune.choice(["funnel", "long_funnel", "diamond", "hexagon", "brick", "triangle", "stairs"]),
            "use_dropout" : tune.choice(["True", "False"]),
            "use_shake_drop" : tune.choice(["True", "False"]),
            "use_shake_shake" : tune.choice(["True", "False"]),
        }},
    ]),
    "network_embedding:__choice__" : "NoEmbedding",
    "network_head:__choice__" : tune.grid_search
    ([
        {"fully_connected" : 
        {
            "activation" : tune.choice(["relu", "tanh", "sigmoid"]),
            "num_layers" : tune.randint(1,4),
            "units_layer_1" : tune.randint(64,512),
            "units_layer_2" : tune.randint(64,512),
            "units_layer_3" : tune.randint(64,512),
            "units_layer_4" : tune.randint(64,512),
        }},
    ]),
    "network_init:__choice__" : tune.grid_search
    ([
        {"NoInit" : {"bias_strategy" : tune.choice(["Zero", "normal"])}}
    ]),
    "optimizer:__choice__" : tune.grid_search
    ([
        {"AdamOptimizer" : 
        {
            "beta1" : tune.uniform(0.85, 0.999),
            "beta2" : tune.uniform(0.9, 0.999),
            "lr" : tune.loguniform(1e-5, 0.1),
            "weight_decay" : tune.uniform(0, 0.1),
        }},
        {"AdamWOptimizer" : 
        {
            "beta1" : tune.uniform(0.85, 0.999),
            "beta2" : tune.uniform(0.9, 0.999),
            "lr" : tune.loguniform(1e-5, 0.1),
            "weight_decay" : tune.uniform(0, 0.1),
        }},
        {"RMSpropOptimizer" : 
        {
            "alpha" : tune.uniform(0.1, 0.99),
            "lr" : tune.loguniform(1e-5, 0.1),
            "momentum" : tune.uniform(0, 0.99),
            "weight_decay" : tune.uniform(0, 0.1),
        }},
        {"SGDOptimizer" : 
        {
            "lr" : tune.loguniform(1e-5, 0.1),
            "momentum" : tune.uniform(0, 0.99),
            "weight_decay" : tune.uniform(0, 0.1),
        }}, 
    ]), 
    "scaler:__choice__" : tune.grid_search
    ([
        {"MinMaxScaler"},
        {"NoScaler"},
        {"Normalizer" : 
        {
            "norm" : tune.choice(["mean_abs", "mean_squared", "max"]),
        }},
        {"PowerTransformer"},
        {"QuantileTransformer" : 
        {
            "n_quantiles" : tune.randint(10,2000),
            "output_distribution" : tune.choice(["uniform", "normal"]),   
        }},
        {"RobustScaler" :
        {
            "q_max" : tune.uniform(0.7, 0.999),
            "q_min" : tune.uniform(0.001, 0.3),
        }},
        {"StandardScaler"},

    ]),
    "trainer:__choice__" : tune.grid_search
    ([
        {"StandardTrainer" : 
        {
            "weighted_loss" : tune.choice(["True", "False"]),
        }},
    ]),
}