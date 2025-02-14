#!/bin/bash

# Output Markdown file
output_file="concatenated.md"
# Clear or create the output file
> "$output_file"

# Find all info.log files in immediate subdirectories (adjust maxdepth if needed)
find ./../logs -maxdepth 2 -type f -name "info.log" | sort | while read -r file; do
    # Get the directory name (remove leading "./" if present)
    dir=$(dirname "$file")
    dir=${dir#./}

    # Extract the algorithm and dataset from the directory name.
    # Assuming format: <algorithm>_default_csuite_<dataset>
    algorithm=${dir%%_*}
    dataset=${dir#*_default_csuite_}

    # Write a Markdown header for the experiment.
    echo "## Algorithm: ${algorithm}, Dataset: ${dataset}" >> "$output_file"
    echo '```' >> "$output_file"
    
    # Filter out timestamps from the beginning of lines.
    # This sed command removes a pattern matching:
    # YYYY-MM-DD HH:MM:SS,mmm - INFO - 
    sed -E 's/^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]{3} - INFO - //' "$file" >> "$output_file"
    
    echo '```' >> "$output_file"
    echo >> "$output_file"
done
