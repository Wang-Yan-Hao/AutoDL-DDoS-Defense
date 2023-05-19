from autoPyTorch.datasets.resampling_strategy import CrossValTypes, HoldoutValTypes

paper_config = {
    'budget_type': 'epochs',
    'budget': 50, # Refit 50 epochs
    'resampling_strategy': HoldoutValTypes.holdout_validation,
    'resampling_strategy_args': {'val_share': 0.2},
    'ensemble_learning': 0,
    'enable_traditional_pipeline': False,
}

run_config = {
    'seed': 42,
    'memory_limit_MB': 4096*45,
    'process_num': 1,
    'DC_using_GPU' : '0', # Default using first GPU
    'DCwF_using_GPU': '0',
    'DCwM_using_GPU': '0',
    'DCwM_using_GPU': '0',
    'DCwP_using_GPU': '0',
    'DCwR_using_GPU': '0',
    'LW1_using_GPU': '0',
    'LW2_using_GPU': '0',
    'T1_using_GPU': '0',
    'T2_using_GPU': '0'
}


