#!/bin/bash

# Create an output directory for per-dataset files
output_dir="by_dataset"
mkdir -p "$output_dir"

# Find all info.log files in immediate subdirectories (adjust maxdepth if needed)
find ../logs -maxdepth 2 -type f -name "info.log" | sort | while read -r file; do
    # Get the parent directory name (remove leading "./" if present)
    dir=$(dirname "$file")
    dir=${dir#./}

    # Extract algorithm and dataset from the directory name.
    # Assuming the format: <algorithm>_default_csuite_<dataset>
    algorithm=${dir%%_*}
    dataset=${dir#*_default_csuite_}

    # Create an output file for the dataset (e.g., "cat_chain.md")
    output_file="${output_dir}/${dataset}.md"

    # If this is the first entry for the dataset, add a top-level header.
    if [ ! -f "$output_file" ]; then
        echo "# Dataset: ${dataset}" > "$output_file"
        echo >> "$output_file"
    fi

    # Append a header for the algorithm.
    echo "## Algorithm: ${algorithm}" >> "$output_file"
    echo '```' >> "$output_file"
    
    # Filter out timestamps from lines and append the content.
    sed -E 's/^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]{3} - INFO - //' "$file" >> "$output_file"
    
    echo '```' >> "$output_file"
    echo >> "$output_file"
done
