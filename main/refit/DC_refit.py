import os
import tempfile as tmp
import pandas as pd
import sklearn.metrics

os.environ['JOBLIB_TEMP_FOLDER'] = tmp.gettempdir()
os.environ['OMP_NUM_THREADS'] = '10'
os.environ['OPENBLAS_NUM_THREADS'] = '10'
os.environ['MKL_NUM_THREADS'] = '10'
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Using first GPU 

from autoPyTorch.api.tabular_classification import TabularClassificationTask
from autoPyTorch.datasets.resampling_strategy import HoldoutValTypes
from ConfigSpace.configuration_space import Configuration

current_file_path = os.path.abspath(__file__) # Get the path of the current Python file
current_file_name = os.path.splitext(os.path.basename(current_file_path))[0] # Get the filename without the path or extension
script_dir = os.path.dirname(os.path.abspath(__file__))

estimator = TabularClassificationTask(
    temporary_directory= os.path.join(script_dir, './tmp_DC/autoPyTorch_example_tmp'),
    output_directory= os.path.join(script_dir, './tmp_DC/autoPyTorch_example_out'),
    delete_tmp_folder_after_terminate=False,
    delete_output_folder_after_terminate=False,
    ensemble_size = 0, # Not ensemble
    resampling_strategy=HoldoutValTypes.holdout_validation,
    resampling_strategy_args={'val_share': 0.2},
    seed=42,
    n_jobs=1,
)

X_train = pd.read_csv('data/output/01-12-five-percent-clean-feature.csv')
y_train = pd.read_csv('data/output/01-12-five-percent-clean-label.csv')
X_test = pd.read_csv('data/output/03-11-five-percent-clean-feature.csv')
y_test = pd.read_csv('data/output/03-11-five-percent-clean-label.csv')

dataset = estimator.get_dataset(X_train=X_train,
                                y_train=y_train,
                                X_test=X_test,
                                y_test=y_test)

space = estimator.get_search_space(dataset)

configuration = Configuration(configuration_space=space,values={ # Set to the best configure find in the search
  'coalescer:__choice__': 'NoCoalescer',
  'data_loader:batch_size': 64,
  'encoder:__choice__': 'NoEncoder',
  'feature_preprocessor:__choice__': 'NoFeaturePreprocessor',
  'imputer:numerical_strategy': 'mean',
  'lr_scheduler:ReduceLROnPlateau:factor': 0.1,
  'lr_scheduler:ReduceLROnPlateau:mode': 'min',
  'lr_scheduler:ReduceLROnPlateau:patience': 10,
  'lr_scheduler:__choice__': 'ReduceLROnPlateau',
  'network_backbone:ShapedMLPBackbone:activation': 'relu',
  'network_backbone:ShapedMLPBackbone:max_units': 200,
  'network_backbone:ShapedMLPBackbone:mlp_shape': 'funnel',
  'network_backbone:ShapedMLPBackbone:num_groups': 5,
  'network_backbone:ShapedMLPBackbone:output_dim': 200,
  'network_backbone:ShapedMLPBackbone:use_dropout': False,
  'network_backbone:__choice__': 'ShapedMLPBackbone',
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

# Start training
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