import os
import random
import pandas as pd

df = pd.read_csv('data/evaluation_data/output/03-11_five_percent_clean_feature.csv')
# Get the number of rows in the DataFrame
num_rows = len(df)

# Calculate the number of rows to extract (1% of total rows)
num_rows_to_extract = int(num_rows * 0.01)

# Generate a random list of row indices to extract
indices_to_extract = random.sample(range(num_rows), num_rows_to_extract)
print(f"The predict data contain {len(indices_to_extract)} packets")

# Extract the rows and create a new DataFrame
df_extracted = df.iloc[indices_to_extract, :]

output_folder = 'data/predict_data/output/'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# Write the extracted data to a new CSV file
df_extracted.to_csv('data/predict_data/output/predict_data.csv', index=False)