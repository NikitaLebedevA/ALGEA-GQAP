from types import SimpleNamespace

import numpy as np
import pytest

from src.algos.core.standard_base import StandardBase
from src.algos.core.improved_base import ImprovedBase
from src.algos.core.logger import GALogger


def ind(cost, value):
    return SimpleNamespace(cost=cost, permutation=np.array([value]))


def test_standard_crossover_retains_failures_and_uses_best_parent_only_for_adaptation(monkeypatch):
    import src.algos.core.standard_base as module

    cls = type("ConcreteStandard", (StandardBase,), {"step": lambda self: None})
    obj = cls.__new__(cls)
    obj.model = None
    obj.logger = GALogger()
    p1, p2 = ind(20, 1), ind(10, 2)
    obj.population = [p1, p2]
    obj._parent_selection_indices = lambda n: [0, 1]
    monkeypatch.setattr(module, "choose_crossover", lambda parents, model: ((p1.permutation, p1), (p2.permutation, p2)))
    monkeypatch.setattr(
        module, "evaluate_permutation", lambda perm, model: ind(15 if perm[0] == 1 else float("inf"), 0)
    )
    ordinary = obj._standard_crossover_pairs(2)
    assert len(ordinary) == 1
    assert ordinary[0][1] is p1
    obj._collect_adaptive_outcomes = True
    adaptive = obj._standard_crossover_pairs(2)
    assert len(adaptive) == 2
    assert all(reference is p2 for _, reference in adaptive)


@pytest.mark.parametrize("robust", [False, True])
def test_improved_reward_reference_does_not_replace_delta_evaluation_baseline(monkeypatch, robust):
    import src.algos.core.improved_base as module

    cls = type("ConcreteImproved", (ImprovedBase,), {"step": lambda self: None})
    obj = cls.__new__(cls)
    obj.model = None
    obj.logger = GALogger()
    p1, p2 = ind(20, 1), ind(10, 2)
    obj.population = [p1, p2]
    obj.best_solution = p2
    obj.selector = SimpleNamespace(roulette_wheel_selection_batch=lambda probs, n: [0, 1])
    obj.repair_batch_wrapper = lambda perms: perms
    obj._collect_adaptive_outcomes = True
    seen = []

    def evaluate(baselines, perms, model):
        seen.extend(baselines)
        return [ind(15, 1), ind(12, 2)]

    monkeypatch.setattr(module, "evaluate_permutation_delta_batch", evaluate)
    monkeypatch.setattr(module, "choose_crossover", lambda parents, model: ((p1.permutation, p1), (p2.permutation, p2)))
    monkeypatch.setattr(
        module, "crossover_robust_chromosome", lambda a, b, model: ((p1.permutation, p1), (p2.permutation, p2))
    )
    output = obj._robust_chromosome_crossover(None, 2) if robust else obj.crossover(None, 2)
    assert seen[0] is p1 and seen[1] is p2
    assert all(reference is p2 for _, reference in output)
