#!/bin/bash
#SBATCH --job-name=SEDD-PSeq-Eval
#SBATCH --account=brics.b5cc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --qos=normal
#SBATCH --output=slurm_out/sedd_pseq_eval-%A.out
#SBATCH --mail-type=END,FAIL

# SLURM batch script for perturbation prediction evaluation using eval_pseq.py
#
# Usage:
#   sbatch slurm/eval_pseq.sbatch
#
# Optional env overrides (eval_pseq.py currently uses hardcoded paths):
#   OUTPUT_DIR=... TEST_DATA_PATH=... PREDICTED_DATA_PATH=... sbatch slurm/eval_pseq.sbatch

# Change to project directory
cd /home/b5cc/sanjukta.b5cc/st3 || exit 1

# Create slurm output directory if it doesn't exist
mkdir -p slurm_out

# Load CUDA module (not required, but kept consistent)
module load cuda/12.6 2>/dev/null || module load cuda 2>/dev/null || echo "Warning: Could not load CUDA module"

# Activate virtual environment (adjust path as needed)
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "Warning: No virtual environment found. Using system Python."
fi

# Print job information
echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=========================================="
echo ""

# Optional overrides (eval_pseq.py currently uses hardcoded paths)
OUTPUT_DIR="${OUTPUT_DIR:-}"
TEST_DATA_PATH="${TEST_DATA_PATH:-}"
PREDICTED_DATA_PATH="${PREDICTED_DATA_PATH:-}"

if [ -n "$OUTPUT_DIR" ] || [ -n "$TEST_DATA_PATH" ] || [ -n "$PREDICTED_DATA_PATH" ]; then
    echo "Note: eval_pseq.py currently uses hardcoded paths."
    echo "Env overrides provided:"
    echo "  OUTPUT_DIR         : ${OUTPUT_DIR:-<unset>}"
    echo "  TEST_DATA_PATH     : ${TEST_DATA_PATH:-<unset>}"
    echo "  PREDICTED_DATA_PATH: ${PREDICTED_DATA_PATH:-<unset>}"
    echo ""
fi

# Build the command
CMD="python scripts/eval_pseq.py"
CMD="$CMD $@"

# Run evaluation with srun
echo "Running command: $CMD"
echo ""
srun $CMD

# Print completion information
echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "Job completed at: $(date)"
echo "=========================================="

