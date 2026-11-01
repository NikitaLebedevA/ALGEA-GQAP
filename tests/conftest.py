import os

# Pin BLAS/numba to one thread before any numpy/numba import so prange-based repair and
# cost evaluation stay deterministic under pytest (parallel numba RNG order varies).
for _env_var in (
    "NUMBA_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_env_var, "1")

import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
CONF_DIR = SCRIPTS / "conf"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from utils.labels import algo_label

OmegaConf.register_new_resolver("algo_label", algo_label, replace=True)

ALL_ALGORITHM_CONFIGS = [
    "standard",
    "gea",
    "standard_gea",
    "ga",
    "adaptive",
    "standard_adaptive",
    "sa",
    "standard_sa",
    "pso",
    "standard_pso",
    "hybrid_ga_pso",
    "standard_hybrid_ga_pso",
    "hybrid_ga_sa",
    "standard_hybrid_ga_sa",
    "gea_scenario_1",
    "standard_gea_scenario_1",
    "gea_scenario_2",
    "standard_gea_scenario_2",
    "gea_scenario_3",
    "standard_gea_scenario_3",
    "adaptive_gea",
    "standard_adaptive_gea",
    "adaptive_gea_scenario_1",
    "standard_adaptive_gea_scenario_1",
    "adaptive_gea_scenario_2",
    "standard_adaptive_gea_scenario_2",
    "adaptive_gea_scenario_3",
    "standard_adaptive_gea_scenario_3",
]

TEST_DATASETS = ["T1", "T2"]

SMOKE_ITERATIONS = 5
SMOKE_POPULATION_SIZE = 20
SMOKE_SEED = 42

# Golden best costs for seed=42, 5 iterations, population_size=20, single-threaded numba.
# Regenerated for direct ALGEA rates, best-ever retention, and deterministic per-row RF repair.
# Verify feasibility, full objective recomputation, and repeatability before changing these values.
EXPECTED_COSTS = {
    ("standard", "T1"): 2997987.839673718,
    ("standard", "T2"): 10900438.301304156,
    ("gea", "T1"): 2225804.305176874,
    ("gea", "T2"): 8760628.11553743,
    ("standard_gea", "T1"): 2980776.543499155,
    ("standard_gea", "T2"): 10871861.534145605,
    ("ga", "T1"): 2747085.8509826898,
    ("ga", "T2"): 10433262.736826159,
    ("adaptive", "T1"): 2776369.889556036,
    ("adaptive", "T2"): 10557084.833208203,
    ("standard_adaptive", "T1"): 3005532.458342049,
    ("standard_adaptive", "T2"): 10888100.202566212,
    ("sa", "T1"): 2957353.8085803287,
    ("sa", "T2"): 10864340.599754814,
    ("standard_sa", "T1"): 3016604.5998349353,
    ("standard_sa", "T2"): 10907543.375230681,
    ("pso", "T1"): 2815001.1989153163,
    ("pso", "T2"): 10638357.47586272,
    ("standard_pso", "T1"): 3010853.7585912915,
    ("standard_pso", "T2"): 10890570.624395194,
    ("hybrid_ga_pso", "T1"): 2704000.3042962337,
    ("hybrid_ga_pso", "T2"): 10478021.783913272,
    ("standard_hybrid_ga_pso", "T1"): 2993473.601733859,
    ("standard_hybrid_ga_pso", "T2"): 10876828.244051142,
    ("hybrid_ga_sa", "T1"): 2751339.5299632596,
    ("hybrid_ga_sa", "T2"): 10504935.530728385,
    ("standard_hybrid_ga_sa", "T1"): 2990790.2838919656,
    ("standard_hybrid_ga_sa", "T2"): 10877062.543743756,
    ("gea_scenario_1", "T1"): 2259760.661006151,
    ("gea_scenario_1", "T2"): 8106364.247562306,
    ("standard_gea_scenario_1", "T1"): 2993473.6017338596,
    ("standard_gea_scenario_1", "T2"): 10871861.534145605,
    ("gea_scenario_2", "T1"): 2632631.6491120476,
    ("gea_scenario_2", "T2"): 10295858.560524289,
    ("standard_gea_scenario_2", "T1"): 2996621.896163943,
    ("standard_gea_scenario_2", "T2"): 10854702.306285208,
    ("gea_scenario_3", "T1"): 2701750.9359217267,
    ("gea_scenario_3", "T2"): 10423707.099049384,
    ("standard_gea_scenario_3", "T1"): 2991646.1147859795,
    ("standard_gea_scenario_3", "T2"): 10885451.341142755,
    ("adaptive_gea", "T1"): 2341842.5279182117,
    ("adaptive_gea", "T2"): 8385926.992478525,
    ("standard_adaptive_gea", "T1"): 3004229.4199971813,
    ("standard_adaptive_gea", "T2"): 10862866.916501533,
    ("adaptive_gea_scenario_1", "T1"): 2331361.258278355,
    ("adaptive_gea_scenario_1", "T2"): 8921150.571255228,
    ("standard_adaptive_gea_scenario_1", "T1"): 3002868.7217457104,
    ("standard_adaptive_gea_scenario_1", "T2"): 10877062.543743756,
    ("adaptive_gea_scenario_2", "T1"): 2696835.680665497,
    ("adaptive_gea_scenario_2", "T2"): 10538743.216188177,
    ("standard_adaptive_gea_scenario_2", "T1"): 3000032.3101643585,
    ("standard_adaptive_gea_scenario_2", "T2"): 10865993.62798793,
    ("adaptive_gea_scenario_3", "T1"): 2770891.7758479994,
    ("adaptive_gea_scenario_3", "T2"): 10511589.912490275,
    ("standard_adaptive_gea_scenario_3", "T1"): 3008024.256433867,
    ("standard_adaptive_gea_scenario_3", "T2"): 10883250.223581068,
}


def load_ga_config(
    config_name: str, *, iterations: int = SMOKE_ITERATIONS, population_size: int = SMOKE_POPULATION_SIZE
) -> dict:
    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        cfg = compose(
            config_name=config_name,
            overrides=[
                f"ga.iterations={iterations}",
                f"ga.population_size={population_size}",
            ],
        )
    return OmegaConf.to_container(cfg.ga, resolve=True)
