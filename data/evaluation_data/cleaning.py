import pandas as pd
import numpy as np
import time

print('## cleaning.py result:')
# Start the timer
start_time = time.time()

pd.set_option('mode.use_inf_as_na', True)

def handle_func(csv_path: str, output_file: str):
    # Remove irrelevant features: because these are not relevant to the data
    df = pd.read_csv(csv_path, usecols=usecols)

    # Change label with 0 to bening and 1 for ddos packets
    df[' Label'] = df[' Label'].replace({'BENIGN': int(0)})
    df[' Label'] = df[' Label'].apply(lambda x: int(1) if x != 0 else x)

    # Fit AutoPytorch asked type
    df[' Protocol'] = df[' Protocol'].astype('category')
    df[' Inbound'] = df[' Inbound'].astype('category')

    # Drop NaN, inf, blank value
    # Rather than returning a new DataFrame. If inplace=True, the original DataFrame is modified, and the method does not return anything.
    df.dropna(inplace=True)
    df.to_csv(output_file, index=False)
    
    print(df.info(verbose=True))
    print('')

# Read column names from file
cols = list(pd.read_csv('data/evaluation_data/output/03-11_five_percent.csv', nrows =1))
print(f'Number of origin col number: {len(cols)}')

usecols = [i for i in cols if i != 'Unnamed: 0'and i != 'Unnamed: 0.1' and i != 'Flow ID' and i != ' Source IP' and i != ' Source Port' and i != ' Destination IP' and i != ' Destination Port' and i != ' Timestamp' and i != 'SimillarHTTP']
print(f'Number of our extract col number: {len(usecols)}')
print(f'The extract col: {usecols}')

print('')

print('Training Day')
handle_func('data/evaluation_data/output/01-12_five_percent.csv', 'data/evaluation_data/output/01-12_five_percent_clean.csv')
print('Testing Day')
handle_func('data/evaluation_data/output/03-11_five_percent.csv', 'data/evaluation_data/output/03-11_five_percent_clean.csv')

elapsed_time = time.time() - start_time

print("Time taken:", elapsed_time, "seconds")

print('')