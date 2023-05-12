import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import time
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
import joblib
from autoPyTorch.optimizer.conver_condition import convert_config

class Search_config():
    def __init__(self):
        self.cs = cs
        self.dataset_properties_2 = dataset_properties_2
        self.include_2 = include_2
        self.exclude_2 = exclude_2
        self.X_train_2 = X_train_2
        self.y_train_2 = y_train_2
        self.X_test_2 = X_test_2
        self.y_test_2 = y_test_2

def train_pipeline(config):
    print(config)
    # 必須call這個function


    def translate_ray_to_autopytorch(autopytorch_cs):
        
        return 0
    
    search_config = joblib.load("/tmp/search_config.joblib")
    from autoPyTorch.pipeline.tabular_classification import TabularClassificationPipeline
    from ConfigSpace.configuration_space import Configuration

    ray_cs = translate_ray_to_autopytorch(config)
    current_config = Configuration(configuration_space=search_config.cs, values=ray_cs)
    current_pipeline = TabularClassificationPipeline(config=current_config, dataset_properties=search_config.dataset_properties_2, include=search_config.include_2, exclude=search_config.exclude_2) # 現在的 pipeline

    step = 0
    step_path = "my_model/step.txt"

    # If `session.get_checkpoint()` is not None, then we are resuming from a checkpoint.
    # Load model state and iteration step from checkpoint.
    if session.get_checkpoint():
        print("Loading from checkpoint.")
        loaded_checkpoint = session.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            path = os.path.join(loaded_checkpoint_dir, "checkpoint.pkl")
            step_file = os.path.join(loaded_checkpoint_dir, step_path)
            pipeline = joblib.load(path)
            with open(step_file, "r") as f:
                step = int(f.read())
    while True:
        #this train modle from data process to caculus loss
        current_pipeline.fit(search_config.X_train_2, search_config.y_train_2)
        #use test funtion to find acc from test_dataset
        acc = current_pipeline.score(search_config.X_test_2, search_config.y_test_2)
        checkpoint = None
        if step % 5 == 0:
            # Every 5 steps, checkpoint our current state.
            # First get the checkpoint directory from tune.
            # Need to create a directory under current working directory
            # to construct an AIR Checkpoint object from.
            os.makedirs("my_model", exist_ok=True)
            # Save the pipeline as a checkpoint
            joblib.dump(pipeline, "my_model/checkpoint.pkl")
            with open(step_path, "w") as f:
                f.write(str(step))
            checkpoint = Checkpoint.from_directory("my_model")

        step += 1
        session.report({"mean_accuracy": acc}, checkpoint=checkpoint)



# 我們創建專為 feature and label datafram 資料的 dataloader
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def pb2_optimize(X_train, y_train, X_test, y_test, dataset_properties, include, exclude, scenario_dict, ta, ta_kwargs, n_jobs, initial_configurations):

    print("pb2.optimize")
    # 資料處理, 資料變成 torch data loader
    # train_dataset = CustomDataset(X_train, y_train)
    # train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # global test_dataset
    # test_dataset = CustomDataset(X_test, y_test)
    # global test_dataloader 
    # test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    global X_train_2
    X_train_2 = X_train
    global y_train_2
    y_train_2 = y_train  
    global X_test_2
    X_test_2 = X_test
    global y_test_2
    y_test_2 = y_test
    global include_2
    include_2 = include
    global exclude_2
    exclude_2 = exclude
    global dataset_properties_2
    dataset_properties_2 = dataset_properties


    # cs (ConfigurationSpace) 轉成 PB2這邊用的 config (字典形式)，cs 為 ConfigurationSpace class, source: https://automl.github.io/ConfigSpace/main/index.html
    global cs
    cs = scenario_dict['cs']
    # cs_dict = cs.get_hyperparameters_dict()
    default_config = cs.get_default_configuration() # default config , Configuration class
    # print(default_config)
    # default_config_dic = default_config.get_dictionary()
    
    search_config = Search_config()
    joblib.dump(search_config, '/tmp/search_config.joblib')

    # __pb2_begin__
    scheduler = PB2(
        time_attr="training_iteration",
        perturbation_interval=5,
        hyperparam_bounds={
            # distribution for resampling
            "lr": [0.1, 1],
            # allow perturbations within this set of categorical values
            # "momentum": [0.8, 0.99],
        },
    )

    # __pb2_end__


    #end condiction
    class TimeStopper(tune.Stopper):
        def __init__(self):
            self._start = time.time()
            self._deadline = 600  # Stop all trials after 60 seconds

        def __call__(self, trial_id, result):
            return False

        def stop_all(self):
            return time.time() - self._start > self._deadline

    stopper = TimeStopper()

    reporter = CLIReporter(max_progress_rows=10)

    tuner = tune.Tuner(
        train_pipeline,
        run_config=air.RunConfig(
            name="1",
            stop=stopper,
            progress_reporter=reporter,
            verbose=1,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_score_attribute="mean_accuracy",
                num_to_keep=4,
            ),
        ),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            metric="mean_accuracy",
            mode="max",
            num_samples=5,
        ),
        param_space=convert_config,
    )
    results = tuner.fit()
    
    # __tune_end__
    print("Best config is:", results.get_best_result().config)