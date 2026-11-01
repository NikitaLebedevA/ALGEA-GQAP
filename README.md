# ALGEA-GQAP — Adaptive Learning-based Genetic Engineering Algorithm

ALGEA combines aggregate and extreme offspring performance with independent
ten-generation reward memory for adaptive operator application rates.

This repository also includes a Python implementation of fourteen metaheuristic families for the Generalized Quadratic
Assignment Problem (GQAP), each available in a **standard** (Holland-style) and
**improved** (GEA-style) variant — **28 Hydra configs** in total. The families cover
plain GA, full GEA, GA without scenario operators, self-adaptive GA/GEA (including adaptive
versions of the three GEA modernization scenarios), simulated annealing, particle swarm
optimization, and two GA hybrids (with PSO and with SA).

See [PROBLEM_DESCRIPTION.md](PROBLEM_DESCRIPTION.md) for the mathematical formulation
and [ALGORITHMS.md](ALGORITHMS.md) for a full description of every algorithm, its
citation, and the shared framework (operators, repair, selection) they are built on.

## Features

- **28 algorithm configs** (14 standard + 14 improved), all sharing the same permutation
  representation, problem-instance format, and run loop — see [ALGORITHMS.md](ALGORITHMS.md)
- **Two scaffolding bases**: `StandardBase` (heuristic initialization, exponential selection,
  pool replacement, no post-operator repair) and `ImprovedBase` (diversity selection, pool replacement,
  `RFRepair`, stagnation immigrants, memetic local search)
- **ALGEA adaptation**: aggregate/extreme reward and punishment, independent 10-generation
  reward memory, midpoint initialization, direct application rates (see [ALGORITHMS.md](ALGORITHMS.md)).
- **Mixins** for cross-cutting behaviour: `AdaptiveRatesMixin`, `AnnealingMixin`, `PSOMixin`
- **Reproducible runs**: `seed_all()` seeds both NumPy and numba; smoke tests pin golden
  costs for every config on dataset subsets (`tests/test_algorithms.py`)

## Installation

```bash
poetry install
```

Core dependencies: `numpy`, `scipy`, `numba`, `pandas`, `openpyxl`, `matplotlib`,
`hydra-core`, `optuna`.

## Project structure

```
src/
├── algos/
│   ├── core/              # AlgorithmBase, StandardBase, ImprovedBase, logger
│   ├── mixins/            # AdaptiveRatesMixin, AnnealingMixin, PSOMixin
│   ├── ga/                # StandardGA, ImprovedGA
│   ├── gea/               # StandardGEA, ImprovedGEA, scenario variants (×3)
│   ├── adaptive/          # adaptive GA/GEA (+ scenario variants ×3)
│   ├── sa/                # StandardSA, ImprovedSA
│   ├── pso/               # StandardParticleSwarm, ImprovedParticleSwarm
│   └── hybrid/            # GA+SA and GA+PSO hybrids
├── operators/             # crossover, mutations, repair
├── costs.py               # objective evaluation (full and delta)
├── selection.py           # DiversitySelector (improved family)
├── seeding.py             # seed_all
└── data/                  # Model, Individual, model_loader

scripts/
├── run.py                 # Hydra benchmark runner
├── tune_algorithm.py      # Optuna hyperparameter tuning
├── tune_components.py     # Optuna tuning of repair/selector (ImprovedGEA only)
├── build_results_table.py # HTML summary from results/*.json
└── conf/                  # one YAML per algorithm config (+ tune_algorithm/)

run_full_standard.sbatch   # Slurm: 14 standard configs
run_full_improved.sbatch   # Slurm: 14 improved configs
run_tune_algorithm*.sbatch # Slurm: tuning batches
run_tune_components.sbatch # Slurm: component tuning (ImprovedGEA)
```

## Usage

```python
from src.data.model_loader import load_model
from src.algos.gea import ImprovedGEA
from src.seeding import seed_all

model = load_model("c201535")
seed_all(42)

ga = ImprovedGEA(
    model,
    population_size=350,
    iterations=1000,
    crossover_rate=0.7,
    mutation_rate=0.3,
)
best = ga.run(time_limit=1000)
print(f"Best cost: {best.cost:.6f}")
```

Every algorithm follows `Algorithm(model, **params).run(time_limit=...)`. A supplied
time budget takes precedence over `iterations`; use `run()` for an iteration-limited run.
SA uses one current solution and retains the best solution separately. See
[ALGORITHMS.md](ALGORITHMS.md) for constructor parameters per class.

### Benchmark runner

```bash
python3 scripts/run.py --config-name=gea
python3 scripts/run.py --config-name=standard_gea
```

**Improved configs** (14): `ga`, `gea`, `adaptive`, `sa`, `pso`, `hybrid_ga_pso`,
`hybrid_ga_sa`, `gea_scenario_{1,2,3}`, `adaptive_gea`, `adaptive_gea_scenario_{1,2,3}`

**Standard configs** (14): `standard`, `standard_gea`, `standard_adaptive`, `standard_sa`,
`standard_pso`, `standard_hybrid_ga_pso`, `standard_hybrid_ga_sa`, `standard_gea_scenario_{1,2,3}`,
`standard_adaptive_gea`, `standard_adaptive_gea_scenario_{1,2,3}`

Runs every dataset in `scripts/conf/datasets/common.yaml` and writes statistics to
`scripts/results/<algo>.json`. Override any field on the command line, e.g.
`python3 scripts/run.py --config-name=adaptive ga.population_size=500 run.runs=10`.

### Tuning

```bash
python3 scripts/tune_algorithm.py --config-name=tune_algorithm/gea
python3 scripts/tune_algorithm.py --config-name=tune_algorithm/ga
python3 scripts/tune_algorithm.py --config-name=tune_algorithm/adaptive_gea
python3 scripts/tune_components.py   # ImprovedGEA repair/selector knobs only
```

Each `scripts/conf/tune_algorithm/<algo>.yaml` defines that algorithm's `param_space`.
Best candidates are written back to `scripts/conf/<algo>.yaml`.

### Tests

```bash
poetry run pytest tests/test_algorithms.py
```

Runs every config on `T1`/`T2` with fixed seed and checks costs against golden references.

### Slurm batch scripts

| Script | Purpose |
|---|---|
| `run_full_standard.sbatch` | All 14 standard-family configs |
| `run_full_improved.sbatch` | All 14 improved-family configs |
| `run_tune_algorithm.sbatch` … `{2..7}.sbatch` | Hyperparameter tuning batches |
| `run_tune_components.sbatch` | Repair/selector tuning for ImprovedGEA |
| `run_ablation_{improved,standard}_pack{1,2}.sbatch` | Operator-adaptation ablation, split into four 8-config Slurm jobs |

## ALGEA on Slurm

Submit from the repository root, after installing dependencies in the cluster environment:

```bash
sbatch --time=00:15:00 run_algea.sbatch adaptive_gea 'datasets.names=[T1,T2]' run.runs=2 run.time_limit=30 ga.population_size=20
sbatch run_algea.sbatch adaptive_gea
```

The full run uses 29 datasets, 30 repeats and a 1000-second search budget per repeat.
Results are isolated under `scripts/results/algea_<job_id>/`. The launcher uses the
existing cluster's `rocky` partition and `gea_gqap_v2_env` environment; adapt these
to your cluster. The installed Hydra CLI failed under Python 3.14 in local testing;
use a compatible Python environment (the project supports Python >=3.11).
