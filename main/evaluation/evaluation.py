import os
import sys
import time
import joblib
import logging
import pandas as pd
from sklearn.metrics import confusion_matrix
# Log setting
current_file_path = os.path.abspath(__file__) # Get the path of the current Python file
current_file_name = os.path.splitext(os.path.basename(current_file_path))[0] # Get the filename without the path or extension
script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f'./{current_file_name}.log')
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
model = joblib.load('42.2.50.0.model')
logging.info(model)

# Load the feature and label data
features = pd.read_csv('data/evaluation_data/output/03-11_five_percent_clean_feature.csv')
labels = pd.read_csv('data/evaluation_data/output/03-11_five_percent_clean_label.csv')

logging.info(f'feature shape: {features.shape}')
logging.info(f'labels shape: {labels.shape}')
# Calculate the number of 0's and 1's
num_zeros = (labels[' Label'] == 0).sum()
num_ones = (labels[' Label'] == 1).sum()
logging.info(f'0 Label number: {num_zeros}')
logging.info(f'1 Label number: {num_ones}')

# Record the start time
start_time = time.time()

# Make predictions on the testing data
y_pred = model.predict(features, batch_size=32)

# Record the end time
end_time = time.time()
# Calculate the total execution time
exec_time = end_time - start_time
# Print the execution time
logging.info('Execution time: {:.2f} seconds'.format(exec_time))

# Convert the predicted values to a DataFrame
Y_df = pd.DataFrame(y_pred, columns=['Negative Label', 'True Lable'])

# 0.5 threshold
Y_df['True Lable'] = Y_df['True Lable'].apply(lambda x: 1 if x >= 0.5 else 0)

# Save the predicted values to a CSV file
Y_df.to_csv('main/evaluation/predict_label.csv', index=False)

# Calculate the confusion matrix
cm = confusion_matrix(labels[' Label'], Y_df['True Lable'])

logging.info(f'confusion matrix: {cm}')

tn, fp, fn, tp = cm.ravel()

logging.info(f'truth negative: {tn}')
logging.info(f'false positive: {fp}')
logging.info(f'false negative: {fn}')
logging.info(f'truth positive: {tp}')

