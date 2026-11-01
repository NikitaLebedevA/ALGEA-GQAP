import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Callable

from src.data.models import Individual

_OPERATOR_LAMBDAS = {
    "crossover": "lambda_crossover",
    "mutation": "lambda_mutation",
    "rc": "lambda_rc",
    "dm": "lambda_dm",
    "gi": "lambda_gi",
}


@dataclass
class RewardMemory:
    reward: float = 0.0
    age: int = 10


class AdaptiveRatesMixin:
    """ALGEA aggregate/extreme reward-punishment control with independent reward memory.

    Lambdas are application rates, not multipliers or a probability distribution.
    Scalar bounds apply to every operator; mappings must cover each active operator.
    """

    def _init_adaptive_rates(
        self,
        crossover_rate: float,
        mutation_rate: float,
        alpha: float,
        lambda_min: float | Mapping[str, float],
        lambda_max: float | Mapping[str, float],
        epsilon: float,
        fixed_operators: Iterable[str] = (),
        reward_horizon: int = 10,
    ) -> None:
        self.fixed_operators = frozenset(fixed_operators)
        unknown = self.fixed_operators.difference(_OPERATOR_LAMBDAS)
        if unknown:
            raise ValueError(f"Unknown fixed adaptive operators: {', '.join(sorted(unknown))}")
        if not math.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be finite and nonnegative")
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        if isinstance(reward_horizon, bool) or not isinstance(reward_horizon, int) or reward_horizon < 1:
            raise ValueError("reward_horizon must be a positive integer")
        self.alpha = alpha
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.epsilon = epsilon
        self.reward_horizon = reward_horizon
        self.reward_memory: dict[str, RewardMemory] = {}
        self.adaptive_diagnostics: dict[str, dict] = {}
        self.adaptation_history: list[dict] = []
        self.lambda_history: list[dict[str, float]] = []
        self._init_adaptive_scenario_rate("base_crossover", "lambda_crossover", crossover_rate)
        self._init_adaptive_scenario_rate("base_mutation", "lambda_mutation", mutation_rate)

    def _bounds(self, operator: str) -> tuple[float, float]:
        def value(setting):
            return float(setting[operator]) if isinstance(setting, Mapping) else float(setting)

        lo, hi = value(self.lambda_min), value(self.lambda_max)
        if not (math.isfinite(lo) and math.isfinite(hi) and 0 <= lo <= hi <= 1):
            raise ValueError(f"Invalid bounds for {operator}: require 0 <= lambda_min <= lambda_max <= 1")
        return lo, hi

    def _init_adaptive_scenario_rate(self, base_attr: str, lambda_attr: str, rate: float) -> None:
        if not math.isfinite(rate) or not 0 <= rate <= 1:
            raise ValueError(f"{base_attr} must be between 0 and 1")
        operator = next(name for name, attr in _OPERATOR_LAMBDAS.items() if attr == lambda_attr)
        lo, hi = self._bounds(operator)
        setattr(self, base_attr, rate)
        # Base rates are used only for fixed-operator ablations and zero-rate disabling.
        initial = rate if operator in self.fixed_operators or rate == 0 else (lo + hi) / 2
        setattr(self, lambda_attr, initial)
        self.reward_memory[operator] = RewardMemory(age=self.reward_horizon)
        self.initial_lambdas = self.snapshot_lambdas()

    def snapshot_lambdas(self) -> dict[str, float]:
        return {attr: float(getattr(self, attr)) for attr in _OPERATOR_LAMBDAS.values() if hasattr(self, attr)}

    def record_lambda_snapshot(self) -> None:
        self.lambda_history.append(self.snapshot_lambdas())
        self.adaptation_history.append({name: dict(values) for name, values in self.adaptive_diagnostics.items()})

    def _performance_signals(self, pairs: list[tuple[Individual, Individual]]) -> dict:
        positive, negative = [], []
        invalid_offspring = invalid_references = 0
        for child, baseline in pairs:
            if not math.isfinite(baseline.cost):
                invalid_references += 1
                continue
            if not math.isfinite(child.cost):
                invalid_offspring += 1
                continue
            delta = (baseline.cost - child.cost) / (abs(baseline.cost) + self.epsilon)
            if delta > 0:
                positive.append(delta)
            elif delta < 0:
                negative.append(-delta)
        # Non-finite objectives have no normalized delta in the paper. Treat an
        # infeasible child as limiting deterioration (q=b_minus=1), preserving
        # rewards from finite children through the independent extreme signal.
        reward_sum, penalty_sum = math.fsum(positive), math.fsum(negative)
        total = reward_sum + penalty_sum + self.epsilon
        r = reward_sum / total
        q = penalty_sum / total
        d_plus, d_minus = max(positive, default=0.0), max(negative, default=0.0)
        b_plus = d_plus / (1 + d_plus)
        b_minus = d_minus / (1 + d_minus)
        if invalid_offspring:
            r, q, b_minus = 0.0, 1.0, 1.0
        return {
            "offspring_count": len(pairs),
            "invalid_offspring": invalid_offspring,
            "invalid_references": invalid_references,
            "reward_sum": reward_sum,
            "penalty_sum": penalty_sum,
            "aggregate_reward": r,
            "aggregate_penalty": q,
            "individual_reward": b_plus,
            "individual_penalty": b_minus,
            "reward": max(r, b_plus),
            "penalty": max(q, b_minus),
        }

    def _update_operator(self, operator: str, pairs: list[tuple[Individual, Individual]]) -> None:
        attr = _OPERATOR_LAMBDAS[operator]
        before = getattr(self, attr)
        signals = self._performance_signals(pairs)
        memory = self.reward_memory[operator]
        if signals["reward"] > 0:
            memory.reward = signals["reward"]
            memory.age = 0
        else:
            memory.age = min(self.reward_horizon, memory.age + 1)
        decay = max(0.0, 1 - memory.age / self.reward_horizon)
        effective = decay * memory.reward
        change = self.alpha * (effective if effective > 0 else -signals["penalty"])
        lo, hi = self._bounds(operator)
        after = min(hi, max(lo, before + change))
        setattr(self, attr, after)
        self.adaptive_diagnostics[operator] = {
            **signals,
            "stored_reward": memory.reward,
            "reward_age": memory.age,
            "decay": decay,
            "effective_reward": effective,
            "lambda_before": before,
            "lambda_after": after,
            "change": after - before,
        }

    def _adaptive_apply(
        self,
        base_attr: str,
        lambda_attr: str,
        apply_fn: Callable[[int], list[tuple[Individual, Individual]]],
    ) -> list[Individual]:
        operator = next(name for name, attr in _OPERATOR_LAMBDAS.items() if attr == lambda_attr)
        rate = getattr(self, lambda_attr)
        # Tell shared producers to retain failed children and use the best-parent
        # reward reference, without changing non-adaptive GA/SA acceptance baselines.
        self._collect_adaptive_outcomes = True
        try:
            pairs = apply_fn(int(self.population_size * rate))
        finally:
            self._collect_adaptive_outcomes = False
        if operator not in self.fixed_operators and getattr(self, base_attr) > 0:
            self._update_operator(operator, pairs)
        return [child for child, _ in pairs if math.isfinite(child.cost)]

    def _adaptive_crossover_and_mutation(self, crossover_fn, mutate_fn):
        return (
            self._adaptive_apply("base_crossover", "lambda_crossover", crossover_fn),
            self._adaptive_apply("base_mutation", "lambda_mutation", mutate_fn),
        )

    def _adaptive_robust_chromosome_crossover(self, crossover_fn):
        return self._adaptive_apply("base_rc", "lambda_rc", crossover_fn)

    def _adaptive_directed_mutation(self, mutate_fn):
        return self._adaptive_apply("base_dm", "lambda_dm", mutate_fn)

    def _adaptive_gene_injection(self, inject_fn):
        return self._adaptive_apply("base_gi", "lambda_gi", inject_fn)
