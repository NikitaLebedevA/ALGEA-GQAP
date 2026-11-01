#!/bin/bash
#SBATCH --job-name=algea
#SBATCH --partition=rocky
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=0
#SBATCH --time=48:00:00
#SBATCH --output=algea-%j.log
#SBATCH --error=algea-%j.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"

module load Python
source activate gea_gqap_v2_env

export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUM_WORKERS="$SLURM_CPUS_PER_TASK"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

config="${1:-adaptive_gea}"
if [ "$#" -gt 0 ]; then shift; fi

poetry run python scripts/run.py \
    --config-name="$config" \
    "$@" \
    "run.output_file=results/algea_${SLURM_JOB_ID}/${config}.json"
