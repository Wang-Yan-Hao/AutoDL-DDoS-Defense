#!/bin/bash

# List of Python files to execute
python_files=("DC.py" "DCwF.py" "DCwM.py" "DCwP.py" "DCwR.py", "LW1.py", "LW2.py", "T1.py", "T2.py")

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