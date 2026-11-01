import math

import numpy as np

from src.algos.core.base import AlgorithmBase
from src.costs import evaluate_permutation
from src.data.models import Individual
from src.heuristics.heuristic2 import heuristic2
from src.operators.crossover import choose_crossover
from src.operators.mutations import choose_mutation
from src.operators.thesis_scenario import analyze_perm, combine_q, mask_mutation
from src.repair import IdentityRepair


class StandardBase(AlgorithmBase):
    """Thesis-style standard scaffold: heuristic2 initialization, exponential parent
    selection, fixed-batch crossover/mutation, thesis scenario operators, and pool
    survivor selection. Offspring are evaluated without post-operator repair."""

    p_fixed_x: float = 0.9
    p_scenario1: float = 0.3
    p_scenario2: float = 0.3
    p_scenario3: float = 0.5
    mask_mutation_index: int = 2

    def __init__(
        self,
        model,
        population_size=350,
        iterations=1000,
        repair_class=None,
        selection_beta: float = 10.0,
        verbose: bool = False,
    ):
        super().__init__(
            model,
            population_size,
            iterations,
            repair_class=repair_class if repair_class is not None else IdentityRepair(),
            verbose=verbose,
        )
        self.selection_beta = selection_beta

    def initialize_population(self) -> None:
        with self.logger.timed("initialization"):
            population: list[Individual] = [heuristic2(self.model)]
            self.logger.record_nfe(1)

            seed_perm = population[0].permutation
            max_attempts = self.population_size * 100
            attempts = 0
            while len(population) < self.population_size and attempts < max_attempts:
                attempts += 1
                mutated_perm = choose_mutation(seed_perm, self.model)
                ind = evaluate_permutation(mutated_perm, self.model)
                self.logger.record_nfe(1)
                if math.isfinite(ind.cost):
                    population.append(ind)

            if len(population) < self.population_size:
                raise RuntimeError(
                    f"heuristic2 init could not fill population "
                    f"({len(population)}/{self.population_size}) after {max_attempts} mutations"
                )

            self.population = sorted(population, key=lambda x: x.cost)
            self.best_solution = self.population[0]
            self.worst_cost = self.population[-1].cost

    def _pool_replace(self, newcomers: list[Individual]) -> None:
        pool = self.population + newcomers
        pool.sort(key=lambda x: x.cost)
        self.population = pool[: self.population_size]

    def _parent_selection_indices(self, n: int) -> np.ndarray:
        costs = np.array([ind.cost for ind in self.population], dtype=float)
        worst = self.worst_cost
        if not math.isfinite(worst) or worst <= 0:
            worst = float(np.max(costs[np.isfinite(costs)])) if np.any(np.isfinite(costs)) else 1.0

        weights = np.zeros_like(costs)
        finite = np.isfinite(costs)
        weights[finite] = np.exp(-self.selection_beta * costs[finite] / worst)

        total = weights.sum()
        probs = weights / total if total > 0 else np.full(len(costs), 1.0 / len(costs))

        cumsum = np.cumsum(probs)
        draws = np.random.random(size=n)
        return np.minimum(np.searchsorted(cumsum, draws, side="right"), len(probs) - 1)

    def _selection_probabilities(self) -> np.ndarray:
        costs = np.array([ind.cost for ind in self.population], dtype=float)
        worst = self.worst_cost
        if not math.isfinite(worst) or worst <= 0:
            worst = float(np.max(costs[np.isfinite(costs)])) if np.any(np.isfinite(costs)) else 1.0
        weights = np.zeros_like(costs)
        finite = np.isfinite(costs)
        weights[finite] = np.exp(-self.selection_beta * costs[finite] / worst)
        total = weights.sum()
        return weights / total if total > 0 else np.full(len(costs), 1.0 / len(costs))

    def _standard_crossover_pairs(self, n: int) -> list[tuple[Individual, Individual]]:
        n = n - (n % 2)
        num_pairs = n // 2
        if not num_pairs:
            return []

        parent_idx = self._parent_selection_indices(2 * num_pairs)
        pairs: list[tuple[Individual, Individual]] = []
        for k in range(num_pairs):
            i1, i2 = parent_idx[2 * k], parent_idx[2 * k + 1]
            p1, p2 = self.population[i1], self.population[i2]
            (child1, _), (child2, _) = choose_crossover((p1, p2), self.model)
            ind1 = evaluate_permutation(child1, self.model)
            ind2 = evaluate_permutation(child2, self.model)
            self.logger.record_nfe(2)
            if math.isfinite(ind1.cost) or getattr(self, "_collect_adaptive_outcomes", False):
                pairs.append(
                    (
                        ind1,
                        (
                            min((p1, p2), key=lambda p: p.cost)
                            if getattr(self, "_collect_adaptive_outcomes", False)
                            else p1
                        ),
                    )
                )
            if math.isfinite(ind2.cost) or getattr(self, "_collect_adaptive_outcomes", False):
                pairs.append(
                    (
                        ind2,
                        (
                            min((p1, p2), key=lambda p: p.cost)
                            if getattr(self, "_collect_adaptive_outcomes", False)
                            else p2
                        ),
                    )
                )
        return pairs

    def _standard_mutate_pairs(self, n: int) -> list[tuple[Individual, Individual]]:
        if n <= 0:
            return []

        indices = np.random.randint(0, len(self.population), size=n)
        pairs: list[tuple[Individual, Individual]] = []
        for idx in indices:
            baseline = self.population[idx]
            mutated_perm = choose_mutation(baseline.permutation, self.model)
            child = evaluate_permutation(mutated_perm, self.model)
            self.logger.record_nfe(1)
            if math.isfinite(child.cost) or getattr(self, "_collect_adaptive_outcomes", False):
                pairs.append((child, baseline))
        return pairs

    def _thesis_scenario1_pairs(self, n: int) -> list[tuple[Individual, Individual]]:
        if n <= 0:
            return []

        n_pop = len(self.population)
        p_count = min(max(1, int(self.p_scenario1 * self.population_size)), n_pop)
        if p_count < 2:
            return []

        _, _, dominant_individual, _ = analyze_perm(
            self.population[:p_count],
            p_fixed_x=self.p_fixed_x,
            model=self.model,
        )
        self.logger.record_nfe(1)

        probs = self._selection_probabilities()
        pairs: list[tuple[Individual, Individual]] = []
        for _ in range(n):
            idx = int(np.searchsorted(np.cumsum(probs), np.random.random(), side="right"))
            idx = min(idx, n_pop - 1)
            partner = self.population[idx]
            (child1, _), (child2, _) = choose_crossover((dominant_individual, partner), self.model)
            reference = (
                min((dominant_individual, partner), key=lambda p: p.cost)
                if getattr(self, "_collect_adaptive_outcomes", False)
                else partner
            )
            for child_perm, baseline in ((child1, reference), (child2, reference)):
                child = evaluate_permutation(child_perm, self.model)
                self.logger.record_nfe(1)
                if math.isfinite(child.cost) or getattr(self, "_collect_adaptive_outcomes", False):
                    pairs.append((child, baseline))
        return pairs

    def _thesis_scenario2_pairs(self, n: int) -> list[tuple[Individual, Individual]]:
        if n <= 0:
            return []

        n_pop = len(self.population)
        p_count = min(max(1, int(self.p_scenario2 * self.population_size)), n_pop)
        if p_count < 1:
            return []

        _, mask_matrix, _, _ = analyze_perm(
            self.population[:p_count],
            p_fixed_x=self.p_fixed_x,
            model=self.model,
        )
        self.logger.record_nfe(1)
        mask_slice = mask_matrix[:p_count]

        pairs: list[tuple[Individual, Individual]] = []
        for _ in range(n):
            ii = int(np.random.randint(0, p_count))
            baseline = self.population[ii]
            mutated_perm = mask_mutation(
                self.mask_mutation_index,
                baseline.permutation,
                mask_slice[ii],
                self.model,
            )
            child = evaluate_permutation(mutated_perm, self.model)
            self.logger.record_nfe(1)
            if math.isfinite(child.cost) or getattr(self, "_collect_adaptive_outcomes", False):
                pairs.append((child, baseline))
        return pairs

    def _thesis_scenario3_pairs(self, n: int) -> list[tuple[Individual, Individual]]:
        if n <= 0:
            return []

        n_pop = len(self.population)
        p_count = min(max(1, int(self.p_scenario3 * self.population_size)), n_pop)
        if p_count < 1:
            return []

        _, _, dominant_individual, dominant_mask = analyze_perm(
            self.population[:p_count],
            p_fixed_x=self.p_fixed_x,
            model=self.model,
        )
        self.logger.record_nfe(1)

        tail_indices = np.arange(max(0, n_pop - p_count), n_pop)
        pairs: list[tuple[Individual, Individual]] = []
        for _ in range(n):
            jj = int(np.random.choice(tail_indices))
            baseline = self.population[jj]
            combined_perm = combine_q(
                dominant_individual.permutation,
                baseline.permutation,
                dominant_mask,
            )
            child = evaluate_permutation(combined_perm, self.model)
            self.logger.record_nfe(1)
            if math.isfinite(child.cost) or getattr(self, "_collect_adaptive_outcomes", False):
                pairs.append((child, baseline))
        return pairs

    def _scenario_batch_count(self, rate: float) -> int:
        return int(math.floor(rate * (self.p_scenario3 * self.population_size)))

    def run_batch_generational(
        self,
        crossover_rate: float,
        mutation_rate: float,
        *,
        enable_scenario1: bool = False,
        enable_scenario2: bool = False,
        enable_scenario3: bool = False,
        scenario_crossover_rate: float = 0.5,
        scenario_mutation_rate: float = 0.2,
        scenario_mutation_rate_3: float | None = None,
    ) -> None:
        n = self.population_size
        ncrossover = int(2 * round((crossover_rate * n) / 2))
        nmutation = int(math.floor(mutation_rate * n))

        candidates = [child for child, _ in self._standard_crossover_pairs(ncrossover)]
        candidates += [child for child, _ in self._standard_mutate_pairs(nmutation)]

        if enable_scenario1:
            n_scenario = self._scenario_batch_count(scenario_crossover_rate)
            candidates += [child for child, _ in self._thesis_scenario1_pairs(n_scenario)]
        if enable_scenario2:
            n_scenario = self._scenario_batch_count(scenario_mutation_rate)
            candidates += [child for child, _ in self._thesis_scenario2_pairs(n_scenario)]
        if enable_scenario3:
            rate3 = scenario_mutation_rate if scenario_mutation_rate_3 is None else scenario_mutation_rate_3
            n_scenario = self._scenario_batch_count(rate3)
            candidates += [child for child, _ in self._thesis_scenario3_pairs(n_scenario)]

        self._pool_replace(candidates)

    def run_adaptive_generational(
        self,
        *,
        rc_rate: float = 0.0,
        dm_rate: float = 0.0,
        injection_rate: float = 0.0,
    ) -> None:
        offspring = self._adaptive_apply("base_crossover", "lambda_crossover", self._standard_crossover_pairs)
        mutations = self._adaptive_apply("base_mutation", "lambda_mutation", self._standard_mutate_pairs)
        candidates = offspring + mutations

        if rc_rate > 0:
            candidates += self._adaptive_apply("base_rc", "lambda_rc", self._thesis_scenario1_pairs)
        if dm_rate > 0:
            candidates += self._adaptive_apply("base_dm", "lambda_dm", self._thesis_scenario2_pairs)
        if injection_rate > 0:
            candidates += self._adaptive_apply("base_gi", "lambda_gi", self._thesis_scenario3_pairs)

        self._pool_replace(candidates)
        if hasattr(self, "record_lambda_snapshot"):
            self.record_lambda_snapshot()

    def run_crossover_mutation_generational(self, crossover_rate: float, mutation_rate: float) -> None:
        self.run_batch_generational(crossover_rate, mutation_rate)

    def run_gea_generational(
        self,
        crossover_rate: float,
        mutation_rate: float,
        *,
        rc_rate: float = 0.0,
        dm_rate: float = 0.0,
        injection_rate: float = 0.0,
    ) -> None:
        self.run_batch_generational(
            crossover_rate,
            mutation_rate,
            enable_scenario1=rc_rate > 0,
            enable_scenario2=dm_rate > 0,
            enable_scenario3=injection_rate > 0,
            scenario_crossover_rate=rc_rate,
            scenario_mutation_rate=dm_rate,
            scenario_mutation_rate_3=injection_rate,
        )

    def run_annealing_generational(self, crossover_rate: float, mutation_rate: float) -> None:
        ncrossover = int(2 * round((crossover_rate * self.population_size) / 2))
        nmutation = int(math.floor(mutation_rate * self.population_size))

        offspring = self._anneal_accept(self._standard_crossover_pairs(ncrossover))
        mutations = self._anneal_accept(self._standard_mutate_pairs(nmutation))
        self._pool_replace(offspring + mutations)
        self._cool_temperature()
