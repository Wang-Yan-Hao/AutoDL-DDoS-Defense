import pandas as pd
import time

print('## divide.py result: No content')
# Start the timer
start_time = time.time()

df = pd.read_csv('data/output/01-12_five_percent_clean.csv')
label = df.loc[:, [' Label']] # Only get label name
feature = df.drop(columns=[' Label']) # Get other feature
label.to_csv('data/output/01-12_five_percent_clean_label.csv', index = False)
feature.to_csv('data/output/01-12_five_percent_clean_feature.csv', index = False)

df = pd.read_csv('data/output/03-11_five_percent_clean.csv')
label = df.loc[:, [' Label']]
feature = df.drop(columns=[' Label'])
label.to_csv('data/output/03-11_five_percent_clean_label.csv', index = False)
feature.to_csv('data/output/03-11_five_percent_clean_feature.csv', index = False)

elapsed_time = time.time() - start_time
print("Time taken:", elapsed_time, "seconds")