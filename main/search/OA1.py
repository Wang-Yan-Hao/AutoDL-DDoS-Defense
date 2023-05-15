import config
import torch
import os
import tempfile as tmp
import pandas as pd
import logging
import sys
import warnings
from autoPyTorch.api.tabular_classification import TabularClassificationTask
from autoPyTorch.datasets.resampling_strategy import CrossValTypes, HoldoutValTypes
import tracemalloc

tracemalloc.start()
def get_memory_info():
    # Get the current total memory usage
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    # Convert the memory usage to human-readable format
    current_memory_mb = current_memory / (1024 * 1024)
    peak_memory_mb = peak_memory / (1024 * 1024)
    # Print the current total memory usage
    print(f"Current Memory Usage: {current_memory_mb:.2f} MB")
    print(f"Peak Memory Usage: {peak_memory_mb:.2f} MB")

#warnings.simplefilter(action='ignore', category=UserWarning)
#warnings.simplefilter(action='ignore', category=FutureWarning)

# Log setting
current_file_path = os.path.abspath(__file__) # Get the path of the current Python file
current_file_name = os.path.splitext(os.path.basename(current_file_path))[0] # Get the filename without the path or extension
script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f'./{current_file_name}.log')
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout) # also output to the console
    ],
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# This environment variable is used by the Joblib library in Python to specify the temporary folder location where it stores data during parallel computations. 
# Joblib is a library used for performing parallel computations in Python, and it is often used in conjunction with scikit-learn, a popular machine learning library.
os.environ['JOBLIB_TEMP_FOLDER'] = tmp.gettempdir() # /tmp
# This environment variable is used by the OpenMP library to specify the maximum number of threads that can be used by a parallel region of code. 
# OpenMP is a library for parallel programming in C, C++, and Fortran, and it is used to write programs that can take advantage of multi-core processors.
os.environ['OMP_NUM_THREADS'] = '10'
# This environment variable is used by the OpenBLAS library to specify the number of threads that can be used by the library when performing linear algebra operations. 
# OpenBLAS is an optimized implementation of the Basic Linear Algebra Subprograms (BLAS) library, which is used for performing linear algebra operations in scientific computing and machine learning.
os.environ['OPENBLAS_NUM_THREADS'] = '10'
# This environment variable is used by the Intel Math Kernel Library (MKL) to specify the maximum number of threads that can be used by the library when performing linear algebra operations.
# MKL is a library of highly optimized mathematical functions for scientific computing, including linear algebra, Fourier transforms, and random number generation.
os.environ['MKL_NUM_THREADS'] = '10'

# 設定跑在第三個GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

logging.info(f'Execution path is {current_file_path}')
logging.info('Read data')
# Data
""" import sklearn.model_selection
X = pd.read_csv(os.path.join(script_dir, '../../sub_data/feature.csv'))
y = pd.read_csv(os.path.join(script_dir, '../../sub_data/label.csv'))
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split( # default number ratio, train:test = 3:1
    X,
    y,
    random_state=1,
) """
X_train = pd.read_csv('../01-12-5-percentclean-feature.csv')
y_train = pd.read_csv('../01-12-5-percentclean-label.csv')
X_test = pd.read_csv('../03-11-5-percentclean-feature.csv')
y_test = pd.read_csv('../03-11-5-percentclean-label.csv')

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

logging.info('Start Process')
api = TabularClassificationTask(
    temporary_directory= os.path.join(script_dir, './tmp/autoPyTorch_example_tmp'),
    output_directory= os.path.join(script_dir, './tmp/autoPyTorch_example_out'),
    delete_tmp_folder_after_terminate=False,
    delete_output_folder_after_terminate=False,
    ensemble_size = 0, # Not ensemble
    include_components=config.paper_config['include_components'],
    resampling_strategy=config.paper_config['resampling_strategy'],
    resampling_strategy_args=config.paper_config['resampling_strategy_args'],
    seed=config.run_config['seed'],
    n_jobs=1,
)

get_memory_info()

logging.info('Starting the search for the best model')
api.set_pipeline_options(device="cuda")
api.search(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test.copy(),
    y_test=y_test.copy(),
    optimize_metric='accuracy', # 我們想要其他指標
    memory_limit=config.run_config['memory_limit_MB'],
    total_walltime_limit=config.paper_config['total_walltime_limit'],
    func_eval_time_limit_secs=config.paper_config['func_eval_time_limit_secs'], # maximum is total_walltime_limit/2
    enable_traditional_pipeline=config.paper_config['enable_traditional_pipeline'],
    # dataset_compression={
    #     "memory_allocation": 500, "methods": ["precision", "subsample"]
    # },
)
get_memory_info()
logging.info('Finished the search for the best model')

# Get the DataFrame info as a string
import io
with io.StringIO() as buffer:
    X_test.info(memory_usage='deep',buf=buffer)
    X_test_info_str = buffer.getvalue()
logging.info(f'X_test_info: {X_test_info_str}')

logging.info('Starting the predict for the best model')
y_pred = api.predict(X_test)
logging.info('Finished the predict for the best model')
score = api.score(y_pred, y_test)
logging.info(f'Predict score_search: {score}')

sprint_statistics = api.sprint_statistics()
logging.info(f'Model statistics: {sprint_statistics}')
show_models = api.show_models()
logging.info(f'Show model_search: {show_models}')

#########
""" logging.info('Starting refit')
api.refit(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    total_walltime_limit=1800,
    run_time_limit_secs=1800,
)

logging.info('Starting the predict for the refit model')
y_pred = api.predict(X_test)
logging.info('Finished the predict for the refit model')
score = api.score(y_pred, y_test)
logging.info(f'Predict score_refit: {score}')
show_models = api.show_models()
logging.info(f'Show model_refit: {show_models}') """
#########
logging.info('End')
