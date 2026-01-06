#!/bin/bash
#
# Submit SEDD RNA-seq training job to SLURM
# Usage: ./bash/train_submit.sh [args...]
#

# Check if sbatch is available
if ! command -v sbatch &> /dev/null; then
    echo "Error: sbatch command not found. Are you on a SLURM cluster?"
    exit 1
fi

# Submit job
echo "Submitting SEDD RNA-seq training job to SLURM..."
sbatch slurm/sb-ism.sbatch "$@"

echo ""
echo "Job submitted. Check status with: squeue -u \$USER"
echo "View output with: tail -f slurm_out/sedd_rnaseq-*.out"
