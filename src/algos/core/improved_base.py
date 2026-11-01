import math
from typing import List, Tuple

import numpy as np

from src.algos.core.base import LOCAL_SEARCH_MAX_J, LOCAL_SEARCH_MAX_PASSES, LOCAL_SEARCH_TOP_K, AlgorithmBase
from src.costs import (
    cost_function_perm,
    cost_function_perm_delta,
    evaluate_permutation,
    evaluate_permutation_delta_batch,
)
from src.data.models import Individual
from src.operators.crossover import choose_crossover, crossover_robust_chromosome
from src.operators.mutations import choose_mutation, mutation_greedy_reassign, mutation_random
from src.repair import RFRepair
from src.selection import DiversitySelector


class ImprovedBase(AlgorithmBase):
    """GEA-family scaffolding: diversity-aware selection, RF repair, stagnation immigrants,
    memetic local search, and pool-based survivor replacement."""

    def __init__(
        self,
        model,
        population_size=350,
        iterations=1000,
        repair_class=None,
        selector: DiversitySelector | None = None,
        stagnation_limit: int = 30,
        immigrant_rate: float = 0.1,
        verbose: bool = False,
    ):
        super().__init__(
            model,
            population_size,
            iterations,
            repair_class=repair_class if repair_class is not None else RFRepair(),
            verbose=verbose,
        )
        self.selector = selector if selector is not None else DiversitySelector()

        self.stagnation_limit = stagnation_limit
        self.immigrant_rate = immigrant_rate
        self.stagnation_counter = 0
        self._last_best_cost = float("inf")

    def compute_selection_probabilities(self):
        with self.logger.timed("parent_selection"):
            return self.selector.compute_selection_probabilities(self.population)

    def select_from_pool(self, pool):
        with self.logger.timed("survivor_selection"):
            self.population = self.selector.select_from_pool(pool, self.population_size, self.progress)

    def crossover(self, probabilities: np.ndarray, n: int) -> List[Tuple[Individual, Individual]]:
        offspring = []
        valid = 0
        new_best = 0

        with self.logger.timed("crossover"):
            raw_perms = []
            baselines = []
            references = []

            num_pairs = len(range(0, n, 2))
            if num_pairs:
                parent_indices = self.selector.roulette_wheel_selection_batch(probabilities, 2 * num_pairs)

                for k in range(num_pairs):
                    i1, i2 = parent_indices[2 * k], parent_indices[2 * k + 1]

                    p1, p2 = self.population[i1], self.population[i2]
                    (child1, baseline1), (child2, baseline2) = choose_crossover((p1, p2), self.model)

                    raw_perms.extend((child1, child2))
                    baselines.extend((baseline1, baseline2))
                    best_parent = min((p1, p2), key=lambda p: p.cost)
                    references.extend((best_parent, best_parent))

            if raw_perms:
                repaired = self.repair_batch_wrapper(np.array(raw_perms))
                children = evaluate_permutation_delta_batch(baselines, repaired, self.model)
                self.logger.record_nfe(len(raw_perms))
                offspring = list(
                    zip(children, references if getattr(self, "_collect_adaptive_outcomes", False) else baselines)
                )

                for child in children:
                    if math.isfinite(child.cost):
                        valid += 1
                        if child.cost < self.best_solution.cost:
                            new_best += 1

        self.logger.record_crossover(n, valid, new_best)
        return offspring

    def mutate(self, n: int) -> List[Tuple[Individual, Individual]]:
        mutations = []
        valid = 0
        new_best = 0

        with self.logger.timed("mutation"):
            if n > 0:
                indices = np.random.randint(0, len(self.population), size=n)
                baselines = [self.population[idx] for idx in indices]

                raw_perms = np.array([choose_mutation(b.permutation, self.model) for b in baselines])
                repaired = self.repair_batch_wrapper(raw_perms)
                children = evaluate_permutation_delta_batch(baselines, repaired, self.model)
                self.logger.record_nfe(n)
                mutations = list(zip(children, baselines))

                for ind in children:
                    if math.isfinite(ind.cost):
                        valid += 1
                        if ind.cost < self.best_solution.cost:
                            new_best += 1

        self.logger.record_mutation(n, valid, new_best)
        return mutations

    def generate_immigrants(self, n: int) -> List[Individual]:
        if n <= 0:
            return []

        with self.logger.timed("immigrants"):
            perms = np.random.randint(0, self.model.I, size=(n, self.model.J), dtype=int)
            perms = self.repair_batch_wrapper(perms)
            immigrants = [evaluate_permutation(perms[i], self.model) for i in range(n)]
            self.logger.record_nfe(n)
            return immigrants

    def maybe_generate_immigrants(self) -> List[Individual]:
        if self.best_solution is not None and self.best_solution.cost < self._last_best_cost - 1e-9:
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1

        if self.best_solution is not None:
            self._last_best_cost = min(self._last_best_cost, self.best_solution.cost)

        if self.stagnation_counter < self.stagnation_limit:
            return []

        self.stagnation_counter = 0
        n = max(1, int(self.immigrant_rate * self.population_size))
        return self.generate_immigrants(n)

    def local_search(self, perm: np.ndarray) -> np.ndarray:
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

    def _robust_chromosome_crossover(self, probabilities: np.ndarray, n: int) -> List[Tuple[Individual, Individual]]:
        offspring: List[Individual] = []
        baselines_out: List[Individual] = []
        valid = 0
        new_best = 0

        n = n - (n % 2)
        with self.logger.timed("crossover"):
            num_pairs = n // 2
            if num_pairs:
                parent_indices = self.selector.roulette_wheel_selection_batch(probabilities, 2 * num_pairs)

                raw_perms = []
                baselines = []
                references = []
                for k in range(num_pairs):
                    i1, i2 = parent_indices[2 * k], parent_indices[2 * k + 1]
                    p1, p2 = self.population[i1], self.population[i2]

                    (child1, base1), (child2, base2) = crossover_robust_chromosome(p1, p2, self.model)
                    raw_perms.extend((child1, child2))
                    baselines.extend((base1, base2))
                    best_parent = min((p1, p2), key=lambda p: p.cost)
                    references.extend((best_parent, best_parent))

                repaired = self.repair_batch_wrapper(np.array(raw_perms))
                offspring = evaluate_permutation_delta_batch(baselines, repaired, self.model)
                baselines_out = references if getattr(self, "_collect_adaptive_outcomes", False) else baselines
                self.logger.record_nfe(len(raw_perms))

                for child in offspring:
                    if math.isfinite(child.cost):
                        valid += 1
                        if child.cost < self.best_solution.cost:
                            new_best += 1

        self.logger.record_crossover(n, valid, new_best)
        return list(zip(offspring, baselines_out))

    def _directed_mutation(self, n: int) -> List[Tuple[Individual, Individual]]:
        mutations: List[Individual] = []
        baselines_out: List[Individual] = []
        valid = 0
        new_best = 0

        with self.logger.timed("mutation"):
            if n > 0:
                indices = np.random.randint(0, len(self.population), size=n)
                baselines = [self.population[idx] for idx in indices]

                raw_perms = np.array([mutation_greedy_reassign(b.permutation, self.model) for b in baselines])
                repaired = self.repair_batch_wrapper(raw_perms)
                mutations = evaluate_permutation_delta_batch(baselines, repaired, self.model)
                baselines_out = baselines
                self.logger.record_nfe(n)

                for child in mutations:
                    if math.isfinite(child.cost):
                        valid += 1
                        if child.cost < self.best_solution.cost:
                            new_best += 1

        self.logger.record_mutation(n, valid, new_best)
        return list(zip(mutations, baselines_out))

    def _gene_injection(self, n: int) -> List[Tuple[Individual, Individual]]:
        injected: List[Individual] = []
        baselines_out: List[Individual] = []
        valid = 0
        new_best = 0

        with self.logger.timed("gene_injection"):
            if n > 0:
                indices = np.random.randint(0, len(self.population), size=n)
                baselines = [self.population[idx] for idx in indices]

                raw_perms = np.array([mutation_random(b.permutation, self.model) for b in baselines])
                repaired = self.repair_batch_wrapper(raw_perms)
                injected = evaluate_permutation_delta_batch(baselines, repaired, self.model)
                baselines_out = baselines
                self.logger.record_nfe(n)

                for child in injected:
                    if math.isfinite(child.cost):
                        valid += 1
                        if child.cost < self.best_solution.cost:
                            new_best += 1

        self.logger.record_mutation(n, valid, new_best)
        return list(zip(injected, baselines_out))

    def polish_elites(self) -> None:
        if self.model.J > LOCAL_SEARCH_MAX_J:
            return

        for idx in range(min(LOCAL_SEARCH_TOP_K, len(self.population))):
            ind = self.population[idx]
            polished_perm = self.local_search(ind.permutation)
            if polished_perm is not ind.permutation and not np.array_equal(polished_perm, ind.permutation):
                self.population[idx] = evaluate_permutation(polished_perm, self.model)

    def run_gea_generation(
        self,
        crossover_rate: float,
        mutation_rate: float,
        *,
        rc_rate: float = 0.0,
        dm_rate: float = 0.0,
        injection_rate: float = 0.0,
    ) -> None:
        """One improved-family GEA generation with fixed-count operator pools."""
        probs = self.compute_selection_probabilities()

        ncrossover = int(2 * round((crossover_rate * self.population_size) / 2))
        nmutation = int(math.floor(mutation_rate * self.population_size))

        offspring = [child for child, _ in self.crossover(probs, ncrossover)]
        mutations = [child for child, _ in self.mutate(nmutation)]

        extras: List[Individual] = []
        if rc_rate > 0:
            n_rc = int(math.floor(rc_rate * self.population_size))
            extras += [child for child, _ in self._robust_chromosome_crossover(probs, n_rc)]
        if dm_rate > 0:
            n_dm = int(math.floor(dm_rate * self.population_size))
            extras += [child for child, _ in self._directed_mutation(n_dm)]
        if injection_rate > 0:
            n_gi = int(math.floor(injection_rate * self.population_size))
            extras += [child for child, _ in self._gene_injection(n_gi)]

        immigrants = self.maybe_generate_immigrants()

        pool = self.population + offspring + mutations + extras + immigrants
        self.select_from_pool(pool)

    def run_crossover_mutation_generation(
        self,
        crossover_rate: float,
        mutation_rate: float,
        *,
        extra_individuals: List[Individual] | None = None,
    ) -> None:
        """One improved-family generation of standard crossover + mutation (+ optional extras)."""
        if extra_individuals:
            probs = self.compute_selection_probabilities()
            ncrossover = int(2 * round((crossover_rate * self.population_size) / 2))
            nmutation = int(math.floor(mutation_rate * self.population_size))
            offspring = [child for child, _ in self.crossover(probs, ncrossover)]
            mutations = [child for child, _ in self.mutate(nmutation)]
            immigrants = self.maybe_generate_immigrants()
            pool = self.population + offspring + mutations + extra_individuals + immigrants
            self.select_from_pool(pool)
            return

        self.run_gea_generation(crossover_rate, mutation_rate)

    def run_adaptive_generation(
        self,
        *,
        rc_rate: float = 0.0,
        dm_rate: float = 0.0,
        injection_rate: float = 0.0,
    ) -> None:
        """One improved-family adaptive generation with independent adaptive application rates."""
        probs = self.compute_selection_probabilities()
        offspring, mutations = self._adaptive_crossover_and_mutation(
            lambda n: self.crossover(probs, n),
            lambda n: self.mutate(n),
        )

        pool = self.population + offspring + mutations
        if rc_rate > 0:
            pool += self._adaptive_robust_chromosome_crossover(lambda n: self._robust_chromosome_crossover(probs, n))
        if dm_rate > 0:
            pool += self._adaptive_directed_mutation(lambda n: self._directed_mutation(n))
        if injection_rate > 0:
            pool += self._adaptive_gene_injection(lambda n: self._gene_injection(n))

        pool += self.maybe_generate_immigrants()
        self.select_from_pool(pool)
        if hasattr(self, "record_lambda_snapshot"):
            self.record_lambda_snapshot()

    def run_annealing_generation(self, crossover_rate: float, mutation_rate: float) -> None:
        """Improved GA offspring filtered by Metropolis acceptance, then pool selection."""
        probs = self.compute_selection_probabilities()

        ncrossover = int(2 * round((crossover_rate * self.population_size) / 2))
        nmutation = int(math.floor(mutation_rate * self.population_size))

        offspring = self._anneal_accept(self.crossover(probs, ncrossover))
        mutations = self._anneal_accept(self.mutate(nmutation))
        immigrants = self.maybe_generate_immigrants()

        pool = self.population + offspring + mutations + immigrants
        self.select_from_pool(pool)
        self._cool_temperature()
