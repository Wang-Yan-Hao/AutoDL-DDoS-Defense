import os
import pandas as pd
import time

print('## extract.py result:')
# Start the timer
start_time = time.time()

# Extract 5% data to a csv and return a list about dataset number info
def process_dataset(dataset_path, output_csv_path) -> dict:
    number_info = {}
    # Get the list of CSV files in the dataset directory
    csv_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.csv')]
    combined_df = pd.DataFrame()  # Initialize an empty dataframe

    for csv_file in csv_files:
        # Read the CSV file into a dataframe
        df = pd.read_csv(csv_file, dtype={'SimillarHTTP': str})
        combined_df = pd.concat([combined_df, df], ignore_index=True)  # Concatenate

    # Group the combined dataframe by the label column
    grouped_df = combined_df.groupby(' Label')

    selected_df = pd.DataFrame()  # Initialize an empty dataframe

    # Iterate over each group
    for name, group in grouped_df:
        if name == 'BENIGN': # If label is benign we leave all data
            selected_df = pd.concat([selected_df, group], ignore_index=True)  # Concatenate
            number_info[name] = len(group) # Store the extract dataset info
            number_info['origin' + name] = len(group) # Store the origin dataset info
            continue
        # Calculate the number of rows to select (5% of the group size)
        n_rows_to_select = int(len(group) * 0.05)
        number_info[name] = n_rows_to_select
        number_info['origin' + name] = len(group)

        # Randomly select n_rows_to_select rows from the group
        selected_group = group.sample(n=n_rows_to_select, random_state=42)
        
        selected_df = pd.concat([selected_df, selected_group], ignore_index=True)  # Concatenate

    # Reset the index of the selected dataframe
    selected_df = selected_df.reset_index(drop=True)
    # Save the five percent data to a csv file
    selected_df.to_csv(output_csv_path, index=False)

    return number_info

# Set the paths for training and testing datasets, you sholud put the data yourself
training_dataset_path = 'data/origin_data/CSV-01-12/01-12/'
testing_dataset_path = 'data/origin_data/CSV-03-11/03-11/'

output_folder = 'data/searching_data/output/'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# Process the training dataset
training_output_csv_path = 'data/searching_data/output/01-12_five_percent.csv'
training_selected_rows = process_dataset(training_dataset_path, training_output_csv_path)

# Process the testing dataset
testing_output_csv_path = 'data/searching_data/output/03-11_five_percent.csv'
testing_selected_rows = process_dataset(testing_dataset_path, testing_output_csv_path)

print('# Training Day')
# Print the selected rows for training dataset
for label, count in training_selected_rows.items():
    print(f'{label} has {count} packets')

print('')

print('# Testing Day')
# Print the selected rows for testing dataset
for label, count in testing_selected_rows.items():
    print(f'{label} has {count} packets')

# Calculate the elapsed time
elapsed_time = time.time() - start_time
print('')

print("Time taken:", elapsed_time, "seconds")

print('')
