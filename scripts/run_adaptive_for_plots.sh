#!/usr/bin/env bash
# Re-run improved adaptive configs to collect lambda_history for plotting.
# Usage: bash scripts/run_adaptive_for_plots.sh

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIGS=(
    adaptive
    adaptive_gea_scenario_1
    adaptive_gea_scenario_2
    adaptive_gea_scenario_3
    adaptive_gea
)

for cfg in "${CONFIGS[@]}"; do
    echo "=== Running $cfg ==="
    python scripts/run.py --config-name="$cfg"
done

echo "=== Generating plots ==="
python scripts/plot_adaptive_parameters.py "$@"
