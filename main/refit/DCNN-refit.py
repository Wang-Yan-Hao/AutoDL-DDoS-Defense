import os
import tempfile as tmp
import warnings

os.environ['JOBLIB_TEMP_FOLDER'] = tmp.gettempdir()
os.environ['OMP_NUM_THREADS'] = '10'
os.environ['OPENBLAS_NUM_THREADS'] = '10'
os.environ['MKL_NUM_THREADS'] = '10'

#warnings.simplefilter(action='ignore', category=UserWarning)
#warnings.simplefilter(action='ignore', category=FutureWarning)

import sklearn.datasets
import pandas as pd
import sklearn.metrics

from autoPyTorch.api.tabular_classification import TabularClassificationTask
from autoPyTorch.datasets.resampling_strategy import HoldoutValTypes
from ConfigSpace.configuration_space import Configuration

current_file_path = os.path.abspath(__file__) # Get the path of the current Python file
current_file_name = os.path.splitext(os.path.basename(current_file_path))[0] # Get the filename without the path or extension
script_dir = os.path.dirname(os.path.abspath(__file__))

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

estimator = TabularClassificationTask(
    temporary_directory= os.path.join(script_dir, './tmp/autoPyTorch_example_tmp'),
    output_directory= os.path.join(script_dir, './tmp/autoPyTorch_example_out'),
    delete_tmp_folder_after_terminate=False,
    delete_output_folder_after_terminate=False,
    ensemble_size = 0, # Not ensemble
    resampling_strategy=HoldoutValTypes.holdout_validation,
    resampling_strategy_args={'val_share': 0.2},
    seed=42,
    n_jobs=1,
)

X_train = pd.read_csv('../../01-12-5-percentclean-feature.csv')
y_train = pd.read_csv('../../01-12-5-percentclean-label.csv')
X_test = pd.read_csv('../../03-11-5-percentclean-feature.csv')
y_test = pd.read_csv('../../03-11-5-percentclean-label.csv')


dataset = estimator.get_dataset(X_train=X_train,
                                y_train=y_train,
                                X_test=X_test,
                                y_test=y_test)
space = estimator.get_search_space(dataset)

configuration = Configuration(configuration_space=space, values={
  'coalescer:__choice__': 'NoCoalescer',
  'data_loader:batch_size': 64,
  'encoder:__choice__': 'NoEncoder',
  'feature_preprocessor:__choice__': 'NoFeaturePreprocessor',
  'imputer:numerical_strategy': 'mean',
  'lr_scheduler:ReduceLROnPlateau:factor': 0.1,
  'lr_scheduler:ReduceLROnPlateau:mode': 'min',
  'lr_scheduler:ReduceLROnPlateau:patience': 10,
  'lr_scheduler:__choice__': 'ReduceLROnPlateau',
  'network_backbone:DCNNBackbone:kernel_growth_ratio': 4,
  'network_backbone:DCNNBackbone:kernel_size': 3,
  'network_backbone:DCNNBackbone:num_kernels': 16,
  'network_backbone:DCNNBackbone:num_sconv': 2,
  'network_backbone:DCNNBackbone:reduction_ratio': 16,
  'network_backbone:__choice__': 'DCNNBackbone',
  'network_embedding:__choice__': 'NoEmbedding',
  'network_head:__choice__': 'fully_connected',
  'network_head:fully_connected:activation': 'relu',
  'network_head:fully_connected:num_layers': 2,
  'network_head:fully_connected:units_layer_1': 128,
  'network_init:XavierInit:bias_strategy': 'Normal',
  'network_init:__choice__': 'XavierInit',
  'optimizer:AdamOptimizer:beta1': 0.9,
  'optimizer:AdamOptimizer:beta2': 0.9,
  'optimizer:AdamOptimizer:lr': 0.01,
  'optimizer:AdamOptimizer:weight_decay': 0.0,
  'optimizer:__choice__': 'AdamOptimizer',
  'scaler:__choice__': 'StandardScaler',
  'trainer:StandardTrainer:weighted_loss': True,
  'trainer:__choice__': 'StandardTrainer',
})

print("Passed Configuration:", configuration)

pipeline, run_info, run_value, dataset = estimator.fit_pipeline(X_train=X_train,
                                                                y_train=y_train,
                                                                X_test=X_test.copy(),
                                                                y_test=y_test.copy(),
                                                                configuration=configuration,
                                                                budget_type='epochs',
                                                                budget=50,
                                                                run_time_limit_secs=1000000,
                                                                pipeline_options={'device': 'cuda'},
                                                                resampling_strategy=HoldoutValTypes.holdout_validation,
                                                                resampling_strategy_args={'val_share': 0.2},
                                                                )

# The fit_pipeline command also returns a named tuple with the pipeline constraints
print(run_info)

# The fit_pipeline command also returns a named tuple with train/test performance
print(run_value)

# This object complies with Scikit-Learn Pipeline API.
# https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
print(pipeline.named_steps)