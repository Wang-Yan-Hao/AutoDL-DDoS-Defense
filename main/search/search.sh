#!/bin/bash

# List of Python files to execute
python_files=("main/search/DC.py" "main/search/DCwF.py" "main/search/DCwM.py" "main/search/DCwP.py" "main/search/DCwR.py" "main/search/LW1.py" "main/search/LW2.py" "main/search/T1.py" "main/search/T2.py")

# Create an array to store the process IDs
pids=()

# Execute each Python file in the background
for file in "${python_files[@]}"; do
    python "$file" &
    pids+=($!)
done

# Wait for all background processes to complete
for pid in "${pids[@]}"; do
    wait "$pid"
done
