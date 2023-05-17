import config
import torch
import os
import tempfile as tmp
import pandas as pd
import logging
import sys
from autoPyTorch.api.tabular_classification import TabularClassificationTask
from autoPyTorch.datasets.resampling_strategy import CrossValTypes, HoldoutValTypes
import sklearn.model_selection
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

logging.info(f'Execution path is {current_file_path}')
# This environment variable is used by the Joblib library in Python to specify the temporary folder location where it stores data during parallel computations. 
# Joblib is a library used for performing parallel computations in Python, and it is often used in conjunction with scikit-learn, a popular machine learning library.
os.environ['JOBLIB_TEMP_FOLDER'] = tmp.gettempdir() # /tmp


logging.info('Read data')

X = pd.read_csv(os.path.join(script_dir, '../../sub_data/feature.csv'))
y = pd.read_csv(os.path.join(script_dir, '../../sub_data/label.csv'))
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split( # default number ratio, train:test = 3:1
    X,
    y,
    random_state=1,
)
print("##########", y_test.shape)
api = TabularClassificationTask(
    # temporary_directory= os.path.join(script_dir, './tmp_OA1/autoPyTorch_example_tmp'),
    # output_directory= os.path.join(script_dir, './tmp_OA1/autoPyTorch_example_out'),
    # delete_tmp_folder_after_terminate=False,
    # delete_output_folder_after_terminate=False,
    ensemble_size = 0, # Not ensemble
    include_components=config.paper_config['include_components'],
    resampling_strategy=config.paper_config['resampling_strategy'],
    resampling_strategy_args=config.paper_config['resampling_strategy_args'],
    seed=config.run_config['seed'],
    n_jobs=1
)
api.set_pipeline_options(device="cuda")

get_memory_info()

logging.info('Starting the search for the best model')
api.search(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test.copy(),
    y_test=y_test.copy(),
    optimize_metric='accuracy',
    memory_limit=config.run_config['memory_limit_MB'],
    total_walltime_limit=config.paper_config['total_walltime_limit'],
    func_eval_time_limit_secs=config.paper_config['func_eval_time_limit_secs'], # maximum is total_walltime_limit/2
    enable_traditional_pipeline=config.paper_config['enable_traditional_pipeline'],
)
logging.info('Finished the search for the best model')
get_memory_info()

# Get the DataFrame info as a string
import io
with io.StringIO() as buffer:
    X_test.info(memory_usage='deep',buf=buffer)
    X_test_info_str = buffer.getvalue()
logging.info(f'X_test_info: {X_test_info_str}')

logging.info('Starting the predict for the best model')
y_pred = api.predict(X_test)


print("#####################", y_pred.shape)

logging.info('Finished the predict for the best model')
score = api.score(y_pred, y_test)
logging.info(f'Predict score: {score}')

sprint_statistics = api.sprint_statistics()
logging.info(f'Model statistics: {sprint_statistics}')

logging.info('Process end')
