#!/bin/bash
#SBATCH --job-name=gea_gqap_tune_algo_2
#SBATCH --error=gea_gqap_tune_algo_2-%j.err
#SBATCH --output=gea_gqap_tune_algo_2-%j.log
#SBATCH --partition=rocky
#SBATCH --time=336:00:00
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --gpus=0

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUM_WORKERS=16

module load Python
source activate gea_gqap_v2_env

ALGORITHMS=(
    pso
    hybrid_ga_pso
)

for algo in "${ALGORITHMS[@]}"; do
    echo "=== Tuning ${algo} ==="
    poetry run python scripts/tune_algorithm.py --config-name="tune_algorithm/${algo}"
done
