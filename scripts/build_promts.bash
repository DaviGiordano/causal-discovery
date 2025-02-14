#!/bin/bash

# Define the prompt text to prepend
read -r -d '' prompt_text <<'EOF'
# Introduction

EOF

# Create output directory if it doesn't exist
output_dir="prompt_by_dataset"
mkdir -p "$output_dir"

# Iterate over each markdown file in by_dataset/
for file in by_dataset/*.md; do
    # Extract the file name (e.g., cat_chain.md)
    filename=$(basename "$file")
    output_file="${output_dir}/${filename}"
    
    # Write the prompt text at the beginning of the new file
    {
        echo "$prompt_text"
        echo  # add an extra blank line
        cat "$file"
    } > "$output_file"
done

echo "Prompt files created in ${output_dir}/"
