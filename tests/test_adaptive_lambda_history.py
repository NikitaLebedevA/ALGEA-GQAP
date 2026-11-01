import hydra

import pytest

from src.algos.adaptive.ga import ImprovedAdaptiveGA
from src.algos.adaptive.gea_scenario import ImprovedAdaptiveGEAScenario1
from src.data.model_loader import load_model
from src.seeding import seed_all


def test_lambda_history_recorded_during_run():
    model = load_model("T1")
    ga = ImprovedAdaptiveGA(model=model, population_size=20, iterations=5, verbose=False)
    seed_all(42)
    ga.run(time_limit=None)
    assert hasattr(ga, "lambda_history")
    assert len(ga.lambda_history) == 5
    assert "lambda_crossover" in ga.lambda_history[0]
    assert "lambda_mutation" in ga.lambda_history[0]
    assert "lambda_rc" not in ga.lambda_history[0]


def test_fixed_operator_stays_at_configured_rate_while_others_adapt():
    model = load_model("T1")
    ga = ImprovedAdaptiveGEAScenario1(
        model=model,
        population_size=20,
        iterations=5,
        alpha=1.0,
        lambda_min=0.1,
        lambda_max=0.9,
        fixed_operators=["rc"],
        verbose=False,
    )
    seed_all(42)
    ga.run(time_limit=None)

    assert all(snapshot["lambda_rc"] == ga.base_rc for snapshot in ga.lambda_history)
    assert any(
        snapshot["lambda_crossover"] != 0.5 or snapshot["lambda_mutation"] != 0.5 for snapshot in ga.lambda_history
    )


def test_unknown_fixed_operator_is_rejected():
    with pytest.raises(ValueError, match="Unknown fixed adaptive operators"):
        ImprovedAdaptiveGA(model=load_model("T1"), fixed_operators=["not_an_operator"])
