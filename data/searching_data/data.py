import sys
import subprocess

# Open the log file in append mode
with open('data/data.log', 'a') as log_file:
    # Execute extract.py
    subprocess.call(['python', 'data/searching_data/extract.py'], stdout=log_file)

    # Execute cleaning.py
    subprocess.call(['python', 'data/searching_data/cleaning.py'], stdout=log_file)

    # Execute divide.py
    subprocess.call(['python', 'data/searching_data/divide.py'], stdout=log_file)