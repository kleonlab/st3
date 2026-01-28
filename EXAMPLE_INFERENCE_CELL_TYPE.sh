#!/bin/bash
#
# Example: Run inference with cell type conditioning
# This script shows how to generate perturbed cells for specific perturbations and cell types
#

# Set your experiment directory (where your trained model is)
EXPERIMENT_DIR="experiments/perturbseq_diffusion"

# Set perturbations files
PERTURBATIONS_FILE="datasets/20M/test_perts.txt"  # 50 test perturbations
PERTURBATIONS_ALL_FILE="datasets/20M/all_perts.txt"  # All perturbations from training

# ============================================================================
# Example 1: Generate for hepg2 cell type using command line
# ============================================================================
echo "Example 1: Generating for hepg2 cell type..."

python scripts/inference_conditional.py \ 
    --experiment_dir ${EXPERIMENT_DIR} \
    --perturbations_file ${PERTURBATIONS_FILE} \
    --perturbations_all_file ${PERTURBATIONS_ALL_FILE} \
    --cell_type hepg2 \
    --num_samples_per_pert 100 \
    --num_steps 100 \
    --temperature 1.0

echo "Results saved to: ${EXPERIMENT_DIR}/inference_results/"
echo ""

# ============================================================================
# Example 2: Generate for hepg2 using bash wrapper script
# ============================================================================
echo "Example 2: Using bash wrapper script..."

EXPERIMENT_DIR=${EXPERIMENT_DIR} \
PERTURBATIONS_FILE=${PERTURBATIONS_FILE} \
CELL_TYPE=hepg2 \
NUM_SAMPLES_PER_PERT=100 \
NUM_STEPS=100 \
bash/inf_conditional.sh

echo ""

# ============================================================================
# Example 3: Generate for all three cell types (sequential)
# ============================================================================
echo "Example 3: Generating for all cell types..."

for CELL_TYPE in hepg2 jurkat rpe1; do
    echo "Generating for ${CELL_TYPE}..."
    
    python scripts/inference_conditional.py \
        --experiment_dir ${EXPERIMENT_DIR} \
        --perturbations_file ${PERTURBATIONS_FILE} \
        --perturbations_all_file ${PERTURBATIONS_ALL_FILE} \
        --cell_type ${CELL_TYPE} \
        --num_samples_per_pert 100 \
        --num_steps 100
    
    # Rename output directory to include cell type
    mv ${EXPERIMENT_DIR}/inference_results \
       ${EXPERIMENT_DIR}/inference_results_${CELL_TYPE}
    
    echo "Results for ${CELL_TYPE} saved!"
    echo ""
done

echo "All cell types complete!"

# ============================================================================
# Example 4: Generate using config file
# ============================================================================
echo "Example 4: Using config file..."

# First, edit your config file (configs/perturbseq_small.yaml) to include:
#   inference:
#     cell_type: hepg2
#     perturbations_file: datasets/20M/test_perts.txt
#     perturbations_all_file: datasets/20M/all_perts.txt

python scripts/inference_conditional.py \
    --config configs/perturbseq_small.yaml \
    --experiment_dir ${EXPERIMENT_DIR}

echo ""

# ============================================================================
# Quick Test: Generate just 10 samples for testing
# ============================================================================
echo "Quick test: Generate 10 samples per perturbation..."

python scripts/inference_conditional.py \
    --experiment_dir ${EXPERIMENT_DIR} \
    --perturbations_file ${PERTURBATIONS_FILE} \
    --perturbations_all_file ${PERTURBATIONS_ALL_FILE} \
    --cell_type hepg2 \
    --num_samples_per_pert 10 \
    --num_steps 50

echo "Test complete!"

