from types import SimpleNamespace

import pytest

from src.algos.core.base import AlgorithmBase
from utils.runner import experiment_seed


class Walk(AlgorithmBase):
    def __init__(self, costs, iterations=3):
        super().__init__(None, 1, iterations)
        self.costs = iter(costs)
        self.clock = 0.0
        self.logger.elapsed = lambda: self.clock

    def initialize_population(self):
        self.population = [SimpleNamespace(cost=10.0)]
        self.best_solution = self.population[0]
        self.worst_cost = 10.0

    def step(self):
        self.clock += 1.0
        self.population = [SimpleNamespace(cost=next(self.costs))]


def test_best_ever_and_first_hitting_iteration_survive_worsening_walk():
    algo = Walk([5, 8, 5])
    assert algo.run().cost == 5
    assert algo.hitting_time == 1
    assert algo.hitting_iteration == 1
    assert algo.logger.cost_history == [5, 5, 5]
    assert algo.logger.time_history == [1, 2, 3]


def test_time_budget_takes_precedence_over_iteration_count():
    algo = Walk([9, 8, 7, 6], iterations=1)
    assert algo.run(time_limit=2.5).cost == 7
    assert len(algo.logger.cost_history) == 3
    assert 2.5 <= algo.clock < 3.5


@pytest.mark.parametrize("limit", [0, -1, float("inf"), float("nan")])
def test_invalid_time_budget(limit):
    with pytest.raises(ValueError):
        Walk([]).run(time_limit=limit)


def test_seed_is_stable_in_a_fresh_interpreter():
    import os
    import subprocess
    import sys
    from pathlib import Path

    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "scripts"), PYTHONHASHSEED="random")
    actual = subprocess.check_output(
        [sys.executable, "-c", 'from utils.runner import experiment_seed; print(experiment_seed("T1", "ALGEA", 1))'],
        env=env,
        text=True,
    )
    assert int(actual) == experiment_seed("T1", "ALGEA", 1)
    assert experiment_seed("T1", "ALGEA", 1) != experiment_seed("T1", "ALGEA", 2)
