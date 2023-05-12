import numpy as np
from autoPyTorch.datasets.resampling_strategy import CrossValTypes, HoldoutValTypes

paper_config = {
    "total_walltime_limit": 1200 ,
    "func_eval_time_limit_secs": 1200,
    "include_components": {
        'trainer': ['StandardTrainer',], # weight loss function
        'network_init': ["NoInit"] # basic comoponents from 皓評
    },
    "resampling_strategy": HoldoutValTypes.holdout_validation,
    "resampling_strategy_args": {'val_share': 0.2},
    "ensemble_learning": False,
    "enable_traditional_pipeline": False,
    "training_dataset": "10-percent-dataset",
    "testing_dataset": "10-percent-dataset",
}

run_config = {
    "cpu_process_nun": 1,
    "seed": 42,
    "memory_limit_MB": 4096*45,
    "use_gpu": True,
    "gpu_index": "2", # index from zero
}


