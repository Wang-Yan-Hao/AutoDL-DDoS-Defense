from autoPyTorch.datasets.resampling_strategy import CrossValTypes, HoldoutValTypes

paper_config = {
    "total_walltime_limit": 259200, # 259200 seconds
    "func_eval_time_limit_secs": 259200,
    "include_components": {
        'trainer': ['StandardTrainer',], # Weight loss function
        'network_backbone': ['CDCNNBackbone'],
        'network_init': ["KaimingInit", "NoInit", "XavierInit"] # Basic components
    },
    "resampling_strategy": HoldoutValTypes.holdout_validation,
    "resampling_strategy_args": {'val_share': 0.2},
    "ensemble_learning": False,
    "enable_traditional_pipeline": False,
}

run_config = {
    "seed": 42,
    "memory_limit_MB": 4096*45,
}


