from os import listdir
from os.path import isfile, join
import pandas as pd
Training_dataset_path = '/home/EA301B/CBS3/training_server/origin_dataset/CSV-01-12/01-12/'
Testing_dataset_path = '/home/EA301B/CBS3/training_server/origin_dataset/CSV-03-11/03-11/'
Training_dataset_files_list = [Training_dataset_path + f for f in listdir(Training_dataset_path) if isfile(join(Training_dataset_path, f))]
Testing_dataset_files_list = [Testing_dataset_path + f for f in listdir(Testing_dataset_path) if isfile(join(Testing_dataset_path, f))]

Ten_percent_training_dict = {'csv_path': '01-12-output.csv'} # 計算總數 每個類別的10%

print(Training_dataset_files_list)
print(Testing_dataset_files_list)
print(f"Training data 有的 dataset 有這麼 {len(Training_dataset_files_list)} 多個 csv 檔案")
print(f"Testing data 有的 dataset 有這麼 {len(Testing_dataset_files_list)} 多個 csv 檔案")

# Read the first CSV file to get the header
first_csv_file = Training_dataset_files_list[0]
header_df = pd.read_csv(first_csv_file, nrows=0, dtype={'SimillarHTTP': str})
dfs = []

for csv_path in Training_dataset_files_list:
    # Read the CSV file into a dataframe
    df = pd.read_csv(csv_path, dtype={'SimillarHTTP': str})
    # Append the dataframe to the dfs list
    dfs.append(df)

# Concatenate the dataframes into a single dataframe
combined_df = pd.concat(dfs, ignore_index=True)

# Set the header of the combined dataframe to the header of the first dataframe
combined_df.columns = header_df.columns

# Assuming the label column is called "label"
# Group the combined dataframe by the label column
grouped_df = combined_df.groupby(" Label")

# Create an empty list to store the randomly selected rows
selected_rows = []

# Iterate over each group
for name, group in grouped_df:
    print(f"現在處理label {name}")
    if name == 'BENIGN':
        selected_rows.append(group)
        Ten_percent_training_dict[name] = len(group)
        continue
    # Calculate the number of rows to select (10% of the group size)
    n_rows_to_select = int(len(group) * 0.05)
    Ten_percent_training_dict[name] = n_rows_to_select

    # Randomly select n_rows_to_select rows from the group
    selected_group = group.sample(n=n_rows_to_select, random_state=12)
    
    # Append the selected rows to the selected_rows list
    selected_rows.append(selected_group)

# Concatenate the selected rows into a single dataframe
selected_df = pd.concat(selected_rows)

Ten_percent_training_dict

for key in Ten_percent_training_dict:
    print(f"{key} 這個類型的封包擁有 {Ten_percent_training_dict[key]} 這麼多個")

print('')

