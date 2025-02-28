#!/bin/bash
#SBATCH --job-name=causal-discovery-exp
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --array=1-18  # 2 batches for now, change to 18 for full experiments
#SBATCH --ntasks=16  # 2 algorithms running in parallel
#SBATCH --cpus-per-task=1  # Each job needs only 1 CPU per dataset
#SBATCH --mem=2G  # Allocate 4GB per task
#SBATCH --time=02:00:00
#SBATCH --nodelist=hpc05
 
mkdir -p logs  # Ensure log directory exists
 
# Define algorithms and datasets

algorithms=("pc_fisherz_005" "pc_fisherz_01" "pc_kcigaussian_005" "pc_kcigaussian_01" "fci_fisherz_005" "fci_fisherz_01" "fci_kcigaussian_005" "fci_kcigaussian_01" "ges_bic" "ges_bdeu" "ges_margigeneral" "ges_margimulti" "grasp_bic" "grasp_bdeu" "grasp_margigeneral" "grasp_margimulti" "directlingam_pwling" "directlingam_kernel")
datasets=("csuite_cat_collider" "csuite_cat_to_cts" "csuite_cts_to_cat" "csuite_large_backdoor" "csuite_large_backdoor_binary_t" "csuite_linexp" "csuite_lingauss" "csuite_mixed_confounding" "csuite_mixed_simpson" "csuite_nonlingauss" "csuite_nonlin_simpson" "csuite_symprod_simpson" "csuite_weak_arrows" "csuite_weak_arrows_binary_t" "ruta_synth_uniform" "ruta_synth_normal")

algorithm=${algorithms[$SLURM_ARRAY_TASK_ID-1]}
 
for dataset in "${datasets[@]}"; do
    python3 main.py --algorithm_tag "$algorithm" --dataset_tag "$dataset" &  # Parallel execution
done
 
wait  # Ensure all processes complete before the script exits
