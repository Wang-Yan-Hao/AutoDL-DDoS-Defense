import os
import sys
import time
import psutil
import logging
import pandas as pd
from joblib import load

# Log setting
current_file_path = os.path.abspath(__file__) # Get the path of the current Python file
current_file_name = os.path.splitext(os.path.basename(current_file_path))[0] # Get the filename without the path or extension
script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join('predict.log')
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout) # Also output to the console
    ],
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Load the model, you should put the model path here, model is put on main/refit/tmp/ folder
model_path = '42.2.50.0.model'
mem_before = psutil.Process().memory_info().rss / 1024 / 1024  # memory usage in MB

model = load(model_path)
# model = load(model_path)
mem_after = psutil.Process().memory_info().rss / 1024 / 1024  # memory usage in MB
mem_usage = mem_after - mem_before  # memory usage in MB
logging.info(f"Memory usage of model {model_path}: {mem_usage:.2f} MB")

# Load the CSV file as a pandas dataframe
csv_path = 'data/predict_data/output/predict_data.csv'
df = pd.read_csv(csv_path)
logging.info(f"Predict number of packets: {df.shape[0]}")
df_numpy = df.to_numpy()

# Measure memory usage during prediction
mem_usage_start = df.memory_usage(deep=True).sum() / 1024 / 1024  # memory usage in MB
cpu_usage_start = psutil.cpu_percent()

total_time = 0.0
for i in range(0,10):
    start_time = time.process_time()
    model.predict(df_numpy)
    end_time = time.process_time()
    total_time += end_time - start_time


cpu_usage_end = psutil.cpu_percent()
mem_usage_end = df.memory_usage(deep=True).sum() / 1024 / 1024  # memory usage in MB
mem_usage_prediction = mem_usage_end - mem_usage_start  # memory usage in MB
cpu_usage_prediction = cpu_usage_end - cpu_usage_start  # CPU usage in percent

# Print the predictions, prediction time, memory usage, and CPU usage
logging.info(f"Prediction time: {total_time/10/1000:f} million seconds")
logging.info(f"Memory usage of {csv_path}: {mem_usage_prediction:.2f} MB")
logging.info(f"CPU usage during prediction: {cpu_usage_prediction:.2f} %")
