import math

import numpy as np

from src.algos.mixins.annealing import AnnealingMixin
from src.algos.core.base import LOCAL_SEARCH_MAX_J, LOCAL_SEARCH_MAX_PASSES, AlgorithmBase
from src.algos.core.standard_base import StandardBase
from src.costs import (
    cost_function_perm,
    cost_function_perm_delta,
    evaluate_permutation,
    evaluate_permutation_delta_batch,
)
from src.operators.mutations import choose_mutation
from src.repair import RFRepair


class StandardSA(StandardBase, AnnealingMixin):
    """Classic single-solution simulated annealing on the thesis scaffold."""

    def __init__(
        self,
        model,
        population_size=1,
        iterations=1000,
        initial_temperature=50.0,
        cooling_rate=0.97,
        min_temperature=1e-3,
        repair_class=None,
        verbose=False,
    ):
        super().__init__(
            model,
            1,
            iterations,
            repair_class=repair_class,
            verbose=verbose,
        )
        self._init_annealing(initial_temperature, cooling_rate, min_temperature)

    def polish_elites(self) -> None:
        pass

    def maybe_generate_immigrants(self):
        return []

    def step(self) -> None:
        current = self.population[0]

        candidate_perm = choose_mutation(current.permutation, self.model)
        candidate = evaluate_permutation(candidate_perm, self.model)
        self.logger.record_nfe(1)

        if candidate.cost <= current.cost:
            self.population[0] = candidate
        elif math.isfinite(candidate.cost) and math.isfinite(current.cost):
            delta = candidate.cost - current.cost
            if np.random.random() < math.exp(-delta / self.temperature):
                self.population[0] = candidate

        self._cool_temperature()


class ImprovedSA(AlgorithmBase, AnnealingMixin):
    """Single-solution SA with RF repair and optional memetic polishing on small instances."""

    def __init__(
        self,
        model,
        population_size=1,
        iterations=1000,
        initial_temperature=50.0,
        cooling_rate=0.97,
        min_temperature=1e-3,
        repair_class=None,
        verbose=False,
    ):
        super().__init__(
            model,
            1,
            iterations,
            repair_class=repair_class if repair_class is not None else RFRepair(),
            verbose=verbose,
        )
        self._init_annealing(initial_temperature, cooling_rate, min_temperature)

    def maybe_generate_immigrants(self):
        return []

    def step(self) -> None:
        current = self.population[0]

        candidate_perm = choose_mutation(current.permutation, self.model)
        repaired = self.repair_batch_wrapper(np.array([candidate_perm]))
        candidates = evaluate_permutation_delta_batch([current], repaired, self.model)
        self.logger.record_nfe(1)
        candidate = candidates[0]

        if candidate.cost <= current.cost:
            self.population[0] = candidate
        elif math.isfinite(candidate.cost) and math.isfinite(current.cost):
            delta = candidate.cost - current.cost
            if np.random.random() < math.exp(-delta / self.temperature):
                self.population[0] = candidate

        self._cool_temperature()

    def polish_elites(self) -> None:
        if self.model.J > LOCAL_SEARCH_MAX_J:
            return

        ind = self.population[0]
        polished_perm = self._local_search(ind.permutation)
        if not np.array_equal(polished_perm, ind.permutation):
            self.population[0] = evaluate_permutation(polished_perm, self.model)

    def _local_search(self, perm: np.ndarray) -> np.ndarray:
        with self.logger.timed("local_search"):
            perm = perm.copy()
            best_cost, _ = cost_function_perm(perm, self.model)
            nfe = 1

            for _ in range(LOCAL_SEARCH_MAX_PASSES):
                improved = False

                for j in range(self.model.J):
                    original = perm[j]
                    best_facility = original

                    for i in range(self.model.I):
                        if i == original:
                            continue
                        trial_perm = perm.copy()
                        trial_perm[j] = i
                        if math.isfinite(best_cost):
                            cost, _ = cost_function_perm_delta(perm, trial_perm, best_cost, self.model)
                        else:
                            cost, _ = cost_function_perm(trial_perm, self.model)
                        nfe += 1
                        if cost < best_cost:
                            best_cost = cost
                            best_facility = i
                            improved = True

                    perm[j] = best_facility

                if not improved:
                    break

            self.logger.record_nfe(nfe)
            return perm
