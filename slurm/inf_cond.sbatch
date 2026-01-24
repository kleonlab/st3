#!/bin/bash
#SBATCH --job-name=SEDD-Cond-Inf
#SBATCH --account=brics.b5cc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=128G
#SBATCH --time=200:00
#SBATCH --qos=normal
#SBATCH --output=slurm_out/sedd_conditional_inference-%A.out
#SBATCH --mail-type=END,FAIL

# SLURM batch script for SEDD Conditional inference using trained checkpoint
#
# Usage:
#   sbatch slurm/sb-inference-conditional.sbatch EXPERIMENT_DIR=experiments/your_exp [OPTIONS]
#
# Or with environment variables:
#   CONFIG=configs/perturbseq_small.yaml \
#   EXPERIMENT_DIR=experiments/5k_psed \
#   PERTURBATIONS_FILE=perturbations.txt \
#   MAPPING_DATA_PATH=/path/to/mapping.h5ad \
#   NUM_SAMPLES_PER_PERT=100 \
#   NUM_STEPS=50 \
#   TEMPERATURE=1.0 \
#   EVALUATE=true \
#   sbatch slurm/sb-inference-conditional.sbatch
#
# perturbations.txt should contain one gene/perturbation name per line:
#   KRAS
#   TP53
#   EGFR
#   non-targeting

# Change to project directory
cd /home/b5cc/sanjukta.b5cc/st3 || exit 1

# Create slurm output directory if it doesn't exist
mkdir -p slurm_out

# Load CUDA module
module load cuda/12.6 2>/dev/null || module load cuda 2>/dev/null || echo "Warning: Could not load CUDA module"

# Activate virtual environment (adjust path as needed)
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "Warning: No virtual environment found. Using system Python."
fi

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Memory optimization for large sequences
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Check CUDA availability
echo "CUDA check:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')" 2>/dev/null || echo "Could not check CUDA"

# Print job information
echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="
echo ""

echo "=========================================="
echo "Perturbation Prediction Inference - SEDD"
echo "=========================================="
echo ""

# Check required parameters
if [ -z "$EXPERIMENT_DIR" ]; then
    echo "ERROR: EXPERIMENT_DIR is required"
    echo "Usage: EXPERIMENT_DIR=path/to/experiment sbatch slurm/sb-inference-conditional.sbatch"
    echo "   or: sbatch slurm/sb-inference-conditional.sbatch EXPERIMENT_DIR=path/to/experiment"
    exit 1
fi

# Default parameters (can be overridden by environment variables)
# Only CONFIG has a default - all other paths come from YAML config unless explicitly overridden
CONFIG="${CONFIG:-/home/b5cc/sanjukta.b5cc/st3/configs/perturbseq_small.yaml}"
EXPERIMENT_DIR="${EXPERIMENT_DIR}"  # No default - must be provided
PERTURBATIONS_FILE="${PERTURBATIONS_FILE}"  # No default - use YAML config
MAPPING_DATA_PATH="${MAPPING_DATA_PATH}"  # No default - use YAML config
TEST_DATA_PATH="${TEST_DATA_PATH}"  # No default - use YAML config
NUM_SAMPLES_PER_PERT="${NUM_SAMPLES_PER_PERT}"  # No default - use YAML config
NUM_STEPS="${NUM_STEPS}"  # No default - use YAML config
TEMPERATURE="${TEMPERATURE}"  # No default - use YAML config
EVALUATE="${EVALUATE:-false}"

# Print inference configuration
echo "Config file           : ${CONFIG}"
echo "Experiment dir        : ${EXPERIMENT_DIR}"
if [ -n "$PERTURBATIONS_FILE" ]; then
    echo "Perturbations file    : ${PERTURBATIONS_FILE}"
    # Show first few lines of the file
    if [ -f "$PERTURBATIONS_FILE" ]; then
        echo "  First perturbations:"
        head -n 5 "$PERTURBATIONS_FILE" | sed 's/^/    /'
        TOTAL_LINES=$(wc -l < "$PERTURBATIONS_FILE")
        echo "  Total: ${TOTAL_LINES} perturbations"
    fi
else
    echo "Perturbations file    : (from config file or test data)"
fi
if [ -n "$MAPPING_DATA_PATH" ]; then
    echo "Mapping data path     : ${MAPPING_DATA_PATH}"
else
    echo "Mapping data path     : (from config file)"
fi
if [ -n "$TEST_DATA_PATH" ]; then
    echo "Test data path        : ${TEST_DATA_PATH}"
else
    echo "Test data path        : (from config file if needed)"
fi
echo "Samples per pert      : ${NUM_SAMPLES_PER_PERT}"
echo "Num steps             : ${NUM_STEPS}"
echo "Temperature           : ${TEMPERATURE}"
echo "Evaluate              : ${EVALUATE}"
echo "=========================================="
echo ""

# Check if config file exists
if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file not found: $CONFIG"
    exit 1
fi

# Check if experiment directory exists
if [ ! -d "$EXPERIMENT_DIR" ]; then
    echo "Error: Experiment directory not found: $EXPERIMENT_DIR"
    exit 1
fi

# Build the command
CMD="python scripts/inference_conditional.py --config $CONFIG"

# Add experiment_dir (required)
if [ -n "$EXPERIMENT_DIR" ]; then
    CMD="$CMD --experiment_dir $EXPERIMENT_DIR"
fi

# Add perturbations file if provided (overrides config)
if [ -n "$PERTURBATIONS_FILE" ]; then
    CMD="$CMD --perturbations_file $PERTURBATIONS_FILE"
fi

# Add mapping data path if provided (overrides config)
if [ -n "$MAPPING_DATA_PATH" ]; then
    CMD="$CMD --mapping_data_path $MAPPING_DATA_PATH"
fi

# Add test data path if provided (overrides config)
if [ -n "$TEST_DATA_PATH" ]; then
    CMD="$CMD --test_data_path $TEST_DATA_PATH"
fi

# Add generation parameters (only if explicitly provided, otherwise use config defaults)
if [ -n "$NUM_SAMPLES_PER_PERT" ]; then
    CMD="$CMD --num_samples_per_pert $NUM_SAMPLES_PER_PERT"
fi
if [ -n "$NUM_STEPS" ]; then
    CMD="$CMD --num_steps $NUM_STEPS"
fi
if [ -n "$TEMPERATURE" ]; then
    CMD="$CMD --temperature $TEMPERATURE"
fi

# Add evaluate flag if true
if [ "$EVALUATE" = "true" ]; then
    CMD="$CMD --evaluate"
fi

# Add any additional command line arguments passed to sbatch
CMD="$CMD $@"

# Run inference with srun
echo "Running command:"
echo "$CMD"
echo ""
srun $CMD

# Print completion information
echo ""
echo "=========================================="
echo "Perturbation prediction complete!"
echo "Check ${EXPERIMENT_DIR}/inference_results for outputs"
echo "Job completed at: $(date)"
echo "=========================================="