import pandas as pd
import random

df = pd.read_csv('03-11-5-percentclean-feature.csv')

# Get the number of rows in the DataFrame
num_rows = len(df)

# Calculate the number of rows to extract (1% of total rows)
num_rows_to_extract = int(num_rows * 0.01)

# Generate a random list of row indices to extract
indices_to_extract = random.sample(range(num_rows), num_rows_to_extract)

# Extract the rows and create a new DataFrame
df_extracted = df.iloc[indices_to_extract, :]

# Write the extracted data to a new CSV file
df_extracted.to_csv('predict_data.csv', index=False)