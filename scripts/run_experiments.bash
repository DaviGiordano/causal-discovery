#!/bin/bash

# List of data tags
data_tags=("csuite_cat_collider" "csuite_cat_to_cts" "csuite_cts_to_cat" "csuite_large_backdoor" "csuite_large_backdoor_binary_t" "csuite_linexp" "csuite_lingauss" "csuite_mixed_confounding" "csuite_mixed_simpson" "csuite_nonlingauss" "csuite_nonlin_simpson" "csuite_symprod_simpson" "csuite_weak_arrows" "csuite_weak_arrows_binary_t" "ruta_synth_uniform" "ruta_synth_normal")
# List of algorithms
causallearn_algorithms=("pc_fisherz_005" "pc_fisherz_01" "pc_kcigaussian_005" "pc_kcigaussian_01" "fci_fisherz_005" "fci_fisherz_01" "fci_kcigaussian_005" "fci_kcigaussian_01" "ges_bic" "ges_bdeu" "ges_margigeneral" "ges_margimulti" "grasp_bic" "grasp_bdeu" "grasp_margigeneral" "grasp_margimulti" "directlingam_pwling" "directlingam_kernel")
castle_algorithms=("grandag_default" "dag_gnn_default" "corl_default" "notears_default")
# algorithms=("${castle_algorithms[@]}" "${causallearn_algorithms[@]}") 

# Nested loop through each algorithm and data tag
for algorithm_tag in "${causallearn_algorithms[@]}"; do
    for data_tag in "${data_tags[@]}"; do
        python3 main.py --algorithm_tag "$algorithm_tag" --dataset_tag "$data_tag"
    done
done
