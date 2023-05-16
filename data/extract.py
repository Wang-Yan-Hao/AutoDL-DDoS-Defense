import os
import pandas as pd

# Extract 5% data to a csv and return a list about dataset number info
def process_dataset(dataset_path, output_csv_path) -> dict:
    number_info = {}
    # Get the list of CSV files in the dataset directory
    csv_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.csv')]
    # Create an empty list to store the dataframe
    df_list = []
    # Create an empty list to store the randomly selected rows
    selected_rows = []

    for csv_file in csv_files:
        # Read the CSV file into a dataframe
        df = pd.read_csv(csv_file, dtype={'SimillarHTTP': str})
        df_list.append(df)

    # Concatenate the dataframes into a single dataframe
    combined_df = pd.concat(df, ignore_index=True)
    # Group the combined dataframe by the label column
    grouped_df = combined_df.groupby(" Label")

    # Iterate over each group
    for name, group in grouped_df:
        if name == 'BENIGN': # If label is benign we leave all data
            selected_rows.append(group)
            number_info[name] = len(group)
            continue
        # Calculate the number of rows to select (5% of the group size)
        n_rows_to_select = int(len(group) * 0.05)
        number_info[name] = n_rows_to_select

        # Randomly select n_rows_to_select rows from the group
        selected_group = group.sample(n=n_rows_to_select, random_state=12)
        
        # Append the selected rows to the selected_rows list
        selected_rows.append(selected_group)

    # Concatenate the selected rows into a single dataframe
    selected_df = pd.concat(selected_rows, ignore_index=True)
    # Reset the index of the selected dataframe
    selected_df = selected_df.reset_index(drop=True)
    # Save the five percent data to a csv file
    selected_df.to_csv(output_csv_path, index=False)

    return number_info

# Set the paths for training and testing datasets, you sholud put the data yourself
training_dataset_path = 'origin_dataset/CSV-01-12/01-12/'
testing_dataset_path = 'origin_dataset/CSV-03-11/03-11/'

# Process the training dataset
training_output_csv_path = './output/01-12-five-percent.csv'
training_selected_rows = process_dataset(training_dataset_path, training_output_csv_path)

# Process the testing dataset
testing_output_csv_path = './output/03-11-five-percent.csv'
testing_selected_rows = process_dataset(testing_dataset_path, testing_output_csv_path)

# Print the selected rows for training dataset
for label, count in training_selected_rows.items():
    print(f"{label} has {count} packets")

print('')

# Print the selected rows for testing dataset
for label, count in testing_selected_rows.items():
    print(f"{label} has {count} packets")

print('')
