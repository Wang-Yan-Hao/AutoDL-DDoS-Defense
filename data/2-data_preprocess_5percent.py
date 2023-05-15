# https://stackoverflow.com/questions/23296282/what-rules-does-pandas-use-to-generate-a-view-vs-a-copy
# https://blog.csdn.net/u013481793/article/details/127069845
import pandas as pd
import numpy as np
chunksize = 100000

pd.set_option('mode.use_inf_as_na', True)

# Read column names from file
cols = list(pd.read_csv('seed=12=03-11-5-percent.csv', nrows =1))
usecols = [i for i in cols if i != 'Unnamed: 0'and i != 'Unnamed: 0.1' and i != 'Flow ID' and i != ' Source IP' and i != ' Source Port' and i != ' Destination IP' and i != ' Destination Port' and i != ' Timestamp' and i != 'SimillarHTTP']

print(f"總共使用 {len(usecols)} 個cols")
print(usecols)
def handle_func(csv_path: str):
    # Remove irrelevant features: because these are not relevant to the data. Change label to 0(not DDoS) and 1(DDoS)
    df = pd.read_csv(csv_path, usecols=usecols)
    i = 0
    # Change label
    df.loc[df[' Label'] == 'BENIGN', ' Label'] = 0 # replace 'BENIGN' with 0
    df.loc[df[' Label'] != 0, ' Label'] = 1 # replace all other values with 1
    # Fit AutoPytorch asked type
    df[' Label'] = df[' Label'].astype(int) # Change label dtype from object to int
    df[' Protocol'] = df[' Protocol'].astype('category')
    df[' Inbound'] = df[' Inbound'].astype('category')
    # Drop NaN, inf, blank value
    # rather than returning a new DataFrame. If inplace=True, the original DataFrame is modified, and the method does not return anything.
    df.dropna(inplace=True)
    # df = df[np.isfinite(df).all(1)]
    outputfile = csv_path.split('.')[0]+'clean.csv'
    df.to_csv(outputfile,index=False)
    # df.to_csv(output_path, mode='a', header=not os.path.exists(output_path))
    # https://stackoverflow.com/questions/17530542/how-to-add-pandas-data-to-an-existing-csv-file

    print('One time end \n')
    # put in output

    from os import listdir


files_list = ['seed=12=01-12-5-percent.csv', 'seed=12=03-11-5-percent.csv']

for file in files_list:
    print(file)
    handle_func(file)