import math

import hydra
import pytest

from src.data.model_loader import load_model
from src.costs import evaluate_permutation
from src.seeding import seed_all
from tests.conftest import (
    ALL_ALGORITHM_CONFIGS,
    EXPECTED_COSTS,
    SMOKE_ITERATIONS,
    SMOKE_SEED,
    TEST_DATASETS,
    load_ga_config,
)


def run_smoke(config_name: str, dataset: str, *, seed: int = SMOKE_SEED):
    ga_cfg = load_ga_config(config_name)
    model = load_model(dataset)
    seed_all(seed)
    ga = hydra.utils.instantiate(ga_cfg, model=model)
    best = ga.run(time_limit=None)
    return ga, best


@pytest.mark.parametrize("config_name", ALL_ALGORITHM_CONFIGS)
@pytest.mark.parametrize("dataset", TEST_DATASETS)
def test_algorithm_cost_is_fixated(config_name: str, dataset: str):
    ga, best = run_smoke(config_name, dataset)

    assert best is not None
    assert math.isfinite(best.cost)
    assert best.cost == pytest.approx(evaluate_permutation(best.permutation, ga.model).cost)
    assert all(b <= a for a, b in zip(ga.logger.cost_history, ga.logger.cost_history[1:]))
    assert ga.logger.nfe > 0
    assert len(ga.logger.cost_history) == SMOKE_ITERATIONS
    assert best.cost == pytest.approx(EXPECTED_COSTS[(config_name, dataset)])


@pytest.mark.parametrize("config_name", ALL_ALGORITHM_CONFIGS)
@pytest.mark.parametrize("run_num", [1, 2, 3])
def test_algorithm_cost_is_reproducible(config_name: str, run_num: int):
    dataset = TEST_DATASETS[run_num % len(TEST_DATASETS)]
    expected = EXPECTED_COSTS[(config_name, dataset)]

    _, first = run_smoke(config_name, dataset)
    _, second = run_smoke(config_name, dataset)

    assert first.cost == pytest.approx(expected)
    assert second.cost == pytest.approx(expected)
    assert first.cost == pytest.approx(second.cost)
