#!/bin/bash
#SBATCH --job-name=gea_ablation_std_p1
#SBATCH --error=gea_ablation_std_p1-%j.err
#SBATCH --output=gea_ablation_std_p1-%j.log
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
    ablation/standard/alga_crossover
    ablation/standard/alga_mutation
    ablation/standard/algea_1_crossover
    ablation/standard/algea_1_mutation
    ablation/standard/algea_1_rc
    ablation/standard/algea_2_crossover
    ablation/standard/algea_2_mutation
    ablation/standard/algea_2_dm
)

for algo in "${ALGORITHMS[@]}"; do
    echo "=== Running ${algo} ==="
    poetry run python scripts/run.py --config-name="${algo}"
done
