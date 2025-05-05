#!/bin/bash
#SBATCH --job-name=causal-discovery
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err
#SBATCH --array=0-5
#SBATCH --cpus-per-task=16

CONFIG_FILES=(
    configs/boss.yaml
    configs/dagma.yaml
    configs/directlingam.yaml
    configs/fges.yaml
    configs/grasp.yaml
    configs/pc.yaml
)
OUTPUT_FILES=(
    output/adult_Xy_processed/boss/graph.txt
    output/adult_Xy_processed/dagma/graph.txt
    output/adult_Xy_processed/directlingam/graph.txt
    output/adult_Xy_processed/fges/graph.txt
    output/adult_Xy_processed/grasp/graph.txt
    output/adult_Xy_processed/pc/graph.txt
)

python3 main.py \
    --config ${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]} \
    --data data/adult/processed/Xy_train.csv \
    --output ${OUTPUT_FILES[$SLURM_ARRAY_TASK_ID]} \
    --knowledge data/adult/processed/knowledge.txt \
    --metadata data/adult/processed/metadata.json