import sys
import subprocess

# Open the log file in append mode
with open('data/data.log', 'a') as log_file:
    # Execute extract.py
    subprocess.call(['python', 'data/evaluation/extract.py'], stdout=log_file)

    # Execute cleaning.py
    subprocess.call(['python', 'data/evaluation/cleaning.py'], stdout=log_file)

    # Execute divide.py
    subprocess.call(['python', 'data/evaluation/divide.py'], stdout=log_file)