#!/bin/bash
#SBATCH --job-name=gea_ablation_std_p2
#SBATCH --error=gea_ablation_std_p2-%j.err
#SBATCH --output=gea_ablation_std_p2-%j.log
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
    ablation/standard/algea_3_crossover
    ablation/standard/algea_3_mutation
    ablation/standard/algea_3_gi
    ablation/standard/algea_crossover
    ablation/standard/algea_mutation
    ablation/standard/algea_rc
    ablation/standard/algea_dm
    ablation/standard/algea_gi
)

for algo in "${ALGORITHMS[@]}"; do
    echo "=== Running ${algo} ==="
    poetry run python scripts/run.py --config-name="${algo}"
done
