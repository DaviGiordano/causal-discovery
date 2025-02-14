#!/bin/bash

# List of data tags
data_tags=("csuite_cat_chain" "csuite_cat_collider" "csuite_cat_to_cts" "csuite_cts_to_cat" "csuite_large_backdoor" "csuite_large_backdoor_binary_t" "csuite_linexp" "csuite_lingauss" "csuite_mixed_confounding" "csuite_mixed_simpson" "csuite_nonlingauss" "csuite_nonlin_simpson" "csuite_symprod_simpson" "csuite_weak_arrows" "csuite_weak_arrows_binary_t")

# List of algorithms
algorithms=("ges_default" "pc_default" "es_default" "fci_default" "directlingam_default" "icalingam_default" "grasp_default")  # Add your algorithms here

# Nested loop through each algorithm and data tag
for algorithm in "${algorithms[@]}"; do
    for data in "${data_tags[@]}"; do
        python3 main.py --algorithm "$algorithm" --data "$data"
    done
done
