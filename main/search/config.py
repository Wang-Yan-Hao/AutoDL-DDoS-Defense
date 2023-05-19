from autoPyTorch.datasets.resampling_strategy import CrossValTypes, HoldoutValTypes

paper_config = {
    'total_walltime_limit': 259200, # 259200 seconds
    'func_eval_time_limit_secs': 259200,
    'DC_include_components': {
        'trainer': ['StandardTrainer',], # Weight loss function
        'network_init': ['KaimingInit', 'NoInit', 'XavierInit'] # Basic components
    },
    'DCwF_include_components': {
        'trainer': ['StandardTrainer',], # Weight loss function
        'feature_preprocessor': ['FeatureAgglomeration'],
        'network_init': ['KaimingInit', 'NoInit', 'XavierInit'] # Basic components
    },
    'DCwM_include_components': {
        'trainer': ['StandardTrainer',], # Weight loss function
        'scaler': ['MinMaxScaler'],
        'network_init': ['KaimingInit', 'NoInit', 'XavierInit'] # Basic components
    },
    'DCwP_include_components': {
        'trainer': ['StandardTrainer',], # Weight loss function
        'feature_preprocessor': ['PCA'],
        'network_init': ['KaimingInit', 'NoInit', 'XavierInit'] # Basic components
    },
    'DCwR_include_components': {
        'trainer': ['StandardTrainer',], # Weight loss function
        'scaler': ['RobustScaler'],
        'network_init': ['KaimingInit', 'NoInit', 'XavierInit'] # Basic components
    },
    'LW1_include_components': {
        'trainer': ['StandardTrainer',], # Weight loss function
        'network_backbone': ['LW1_Backbone'],
        'network_init': ['KaimingInit', 'NoInit', 'XavierInit'] # Basic components
    },
    'LW2_include_components': {
        'trainer': ['StandardTrainer',], # Weight loss function
        'network_backbone': ['LW2_Backbone'],
        'network_init': ['KaimingInit', 'NoInit', 'XavierInit'] # Basic components
    },
    'T1_include_components': {
        'trainer': ['StandardTrainer',], # Weight loss function
        'network_backbone': ['T1_Backbone'],
        'network_init': ['KaimingInit', 'NoInit', 'XavierInit'] # Basic components
    },
    'T2_include_components': {
        'trainer': ['StandardTrainer',], # Weight loss function
        'network_backbone': ['T2_Backbone'],
        'network_init': ['KaimingInit', 'NoInit', 'XavierInit'] # Basic components
    },    
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