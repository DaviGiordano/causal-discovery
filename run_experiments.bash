#!/bin/bash

# List of data tags
data_tags=("csuite_cat_chain" "csuite_cat_collider" "csuite_cat_to_cts" "csuite_cts_to_cat" "csuite_large_backdoor" "csuite_large_backdoor_binary_t" "csuite_linexp" "csuite_lingauss" "csuite_mixed_confounding" "csuite_mixed_simpson" "csuite_nonlingauss" "csuite_nonlin_simpson" "csuite_symprod_simpson" "csuite_weak_arrows" "csuite_weak_arrows_binary_t")


# Loop through each data tag and run the command
for data in "${data_tags[@]}"; do
    python3 main.py --algorithm ges_default --data "$data"
done
