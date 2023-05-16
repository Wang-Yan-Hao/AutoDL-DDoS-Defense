# https://stackoverflow.com/questions/23296282/what-rules-does-pandas-use-to-generate-a-view-vs-a-copy
# https://blog.csdn.net/u013481793/article/details/127069845
import pandas as pd
import numpy as np
from os import listdir

pd.set_option('mode.use_inf_as_na', True)

def handle_func(csv_path: str, output_file: str):
    # Remove irrelevant features: because these are not relevant to the data
    df = pd.read_csv(csv_path, usecols=usecols)
    # Change label with 0 to bening and 1 for ddos packets
    df[' Label'] = df[' Label'].replace({'BENIGN': 0}).astype(int)
    df[' Label'] = df[' Label'].apply(lambda x: 1 if x != 0 else x)
    # Fit AutoPytorch asked type
    df[' Protocol'] = df[' Protocol'].astype('category')
    df[' Inbound'] = df[' Inbound'].astype('category')
    # Drop NaN, inf, blank value
    # Rather than returning a new DataFrame. If inplace=True, the original DataFrame is modified, and the method does not return anything.
    df.dropna(inplace=True)
    df.to_csv(output_file, index=False)
    print('One time end \n')
    # Read column names from file

# Read column names from file
cols = list(pd.read_csv('./output/03-11-five-percent.csv', nrows =1))
print(f"Number of origin features: {len(cols)}")
usecols = [i for i in cols if i != 'Unnamed: 0'and i != 'Unnamed: 0.1' and i != 'Flow ID' and i != ' Source IP' and i != ' Source Port' and i != ' Destination IP' and i != ' Destination Port' and i != ' Timestamp' and i != 'SimillarHTTP']

print(f"Our extract features: {len(usecols)}")
print(f"The extract feature: {usecols}")

handle_func('./output/01-12-five-percent.csv', './output/01-12-five-percent-clean.csv')
handle_func('./output/03-11-five-percent.csv', './output/03-11-five-percent-clean.csv')