import subprocess

# Open the log file in append mode
with open('data/evaluation_data/evaluation.log', 'a') as log_file:
    # Execute extract.py
    subprocess.call(['python', 'data/evaluation_data/extract.py'], stdout=log_file)

    # Execute cleaning.py
    subprocess.call(['python', 'data/evaluation_data/cleaning.py'], stdout=log_file)

    # Execute divide.py
    subprocess.call(['python', 'data/evaluation_data/divide.py'], stdout=log_file)