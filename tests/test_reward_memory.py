from types import SimpleNamespace

import pytest

from src.algos.mixins.adaptive import AdaptiveRatesMixin


def individual(cost):
    return SimpleNamespace(cost=cost)


def controller(**kwargs):
    obj = AdaptiveRatesMixin()
    obj.population_size = 100
    obj._init_adaptive_rates(
        0.7,
        0.3,
        kwargs.pop("alpha", 0.1),
        kwargs.pop("lambda_min", 0.1),
        kwargs.pop("lambda_max", 0.9),
        kwargs.pop("epsilon", 1e-12),
        **kwargs
    )
    return obj


def pairs(*costs, reference=100):
    return [(individual(cost), individual(reference)) for cost in costs]


def test_pdf_example_and_reward_expiration():
    obj = controller()
    # +0.1 and -0.1 => aggregate reward 0.5, extremes < 0.5.
    obj._update_operator("crossover", pairs(90, 110))
    assert obj.lambda_crossover == pytest.approx(0.55)
    obj._update_operator("crossover", [])
    assert obj.lambda_crossover == pytest.approx(0.595)
    obj._update_operator("crossover", [])
    assert obj.lambda_crossover == pytest.approx(0.635)
    for age in range(3, 10):
        obj._update_operator("crossover", pairs(110))
        state = obj.adaptive_diagnostics["crossover"]
        assert state["decay"] == pytest.approx(1 - age / 10)
        assert state["change"] >= 0  # punishment suppressed even at clipping bound
    before = obj.lambda_crossover
    obj._update_operator("crossover", pairs(110))
    assert obj.reward_memory["crossover"].age == 10
    assert obj.lambda_crossover == pytest.approx(before - 0.1)


def test_new_smaller_reward_replaces_memory_and_is_operator_local():
    obj = controller()
    obj._update_operator("crossover", pairs(90))
    old = obj.reward_memory["crossover"].reward
    obj._update_operator("crossover", [])
    obj._update_operator("crossover", pairs(99, 120))
    assert 0 < obj.reward_memory["crossover"].reward < old
    assert obj.reward_memory["crossover"].age == 0
    assert obj.reward_memory["mutation"].reward == 0
    assert obj.reward_memory["mutation"].age == 10


def test_aggregate_and_extreme_signals_are_combined_separately():
    obj = controller()
    # Positive objective need not be assumed: improvement of 2, deterioration 10.
    signals = obj._performance_signals(pairs(-100, 1100))
    assert signals["aggregate_reward"] == pytest.approx(1 / 6)
    assert signals["individual_reward"] == pytest.approx(2 / 3)
    assert signals["reward"] == pytest.approx(2 / 3)
    assert signals["penalty"] == pytest.approx(10 / 11)


def test_negative_and_zero_reference_costs():
    obj = controller()
    assert obj._performance_signals(pairs(-110, reference=-100))["reward"] > 0
    signals = obj._performance_signals(pairs(-1, reference=0))
    assert signals["reward"] == pytest.approx(1)


def test_infeasible_children_are_penalized_but_finite_reward_has_priority():
    obj = controller()
    obj._update_operator("crossover", pairs(float("inf")))
    assert obj.lambda_crossover == pytest.approx(0.4)
    obj._update_operator("crossover", pairs(50, float("inf")))
    assert obj.adaptive_diagnostics["crossover"]["invalid_offspring"] == 1
    assert obj.lambda_crossover > 0.4
    assert obj._performance_signals(pairs(10, reference=float("inf")))["invalid_references"] == 1


def test_no_signal_and_clipping():
    obj = controller(alpha=10)
    obj._update_operator("crossover", [])
    assert obj.lambda_crossover == 0.5
    obj._update_operator("crossover", pairs(110))
    assert obj.lambda_crossover == 0.1
    obj._update_operator("crossover", pairs(90))
    assert obj.lambda_crossover == 0.9


def test_per_operator_bounds_and_application_rates():
    obj = controller(lambda_min={"crossover": 0.2, "mutation": 0.1}, lambda_max={"crossover": 0.6, "mutation": 0.3})
    seen = []
    obj._adaptive_apply("base_crossover", "lambda_crossover", lambda n: seen.append(n) or [])
    assert seen == [40]  # no extra multiplication by base crossover 0.7
    assert obj.lambda_mutation == pytest.approx(0.2)


def test_fixed_operator_uses_configured_rate_and_filters_invalid_children():
    obj = controller(fixed_operators=["mutation"])
    seen = []

    def apply(n):
        seen.append(n)
        assert obj._collect_adaptive_outcomes
        return pairs(90, float("inf"))

    children = obj._adaptive_apply("base_mutation", "lambda_mutation", apply)
    assert seen == [30]
    assert len(children) == 1
    assert obj.lambda_mutation == 0.3
    assert not obj._collect_adaptive_outcomes


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(lambda_max=1.5),
        dict(lambda_min=-0.1),
        dict(alpha=-1),
        dict(epsilon=0),
        dict(reward_horizon=0),
        dict(reward_horizon=1.5),
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        controller(**kwargs)
