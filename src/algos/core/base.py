import math

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from src.algos.core.logger import GALogger
from src.costs import evaluate_permutation
from src.data.models import Individual, Model

# Local search (memetic polishing) is only worth its O(J*I) cost-recompute overhead
# on small instances -- on the large T-datasets (J in the thousands) it would dominate runtime.
LOCAL_SEARCH_MAX_J = 50
LOCAL_SEARCH_TOP_K = 3
LOCAL_SEARCH_MAX_PASSES = 2


class AlgorithmBase(ABC):
    """Shared run loop and population lifecycle for every metaheuristic in this project."""

    def __init__(
        self,
        model: Model,
        population_size: int,
        iterations: int,
        repair_class=None,
        verbose: bool = False,
    ):
        self.model = model
        self.population_size = population_size
        self.iterations = iterations
        self.verbose = verbose

        self.repair_class = repair_class
        self.logger = GALogger()

        self.population: List[Individual] = []
        self.best_solution: Individual | None = None
        self.worst_cost: float = float("inf")
        self.progress: float = 0.0

    def repair_batch_wrapper(self, perms: np.ndarray) -> np.ndarray:
        with self.logger.timed("repair"):
            return self.repair_class.repair_batch(perms, self.model)

    def initialize_population(self) -> None:
        with self.logger.timed("initialization"):
            perms = np.random.randint(0, self.model.I, size=(self.population_size, self.model.J), dtype=int)
            perms = self.repair_batch_wrapper(perms)

            self.population = [evaluate_permutation(perms[i], self.model) for i in range(self.population_size)]
            self.logger.record_nfe(self.population_size)

            self.population.sort(key=lambda x: x.cost)
            self.best_solution = self.population[0]
            self.worst_cost = self.population[-1].cost

    def polish_elites(self) -> None:
        """Override in subclasses that apply memetic local search."""

    @abstractmethod
    def step(self) -> None:
        pass

    def _avg_diversity_for_logging(self) -> float:
        selector = getattr(self, "selector", None)
        return selector.avg_diversity if selector is not None else 0.0

    def run(self, time_limit: float | None = None):
        """Run to the wall-clock budget, or to `iterations` when no budget is given.

        A generation is atomic: a time-limited run can exceed its budget by the
        final generation. Initialization is included; model loading is not.
        """
        if time_limit is not None and (not math.isfinite(time_limit) or time_limit <= 0):
            raise ValueError("time_limit must be finite and positive")
        if time_limit is None and self.iterations < 1:
            raise ValueError("iterations must be positive without a time limit")
        self.logger.start_run()
        self.initialize_population()

        self.hitting_time: float = self.logger.elapsed()
        self.hitting_iteration = 0

        if self.verbose:
            print(f"GA started → Population: {self.population_size:,} | Iterations: {self.iterations}\n")

        it = 0
        while (self.logger.elapsed() < time_limit) if time_limit is not None else (it < self.iterations):
            it += 1
            self.progress = (
                min(1.0, self.logger.elapsed() / time_limit) if time_limit is not None else it / self.iterations
            )

            iter_start = self.logger.elapsed()
            self.step()
            self.population.sort(key=lambda x: x.cost)
            self.polish_elites()
            iter_time = self.logger.elapsed() - iter_start
            self.logger.record_iteration(iter_time)

            self.population.sort(key=lambda x: x.cost)
            prev_best = self.best_solution.cost if self.best_solution is not None else float("inf")
            current_best = self.population[0]
            self.worst_cost = max(self.worst_cost, self.population[-1].cost)

            if current_best.cost < prev_best:
                self.best_solution = current_best
                self.hitting_time = self.logger.elapsed()
                self.hitting_iteration = it

            self.logger.record_best_cost(self.best_solution.cost)

            if it % 50 == 0:
                if self.verbose:
                    self.logger.print_iteration_info(
                        it, iter_time, self.best_solution.cost, self._avg_diversity_for_logging()
                    )
                self.logger.reset_operator_counters()

            if time_limit and self.logger.elapsed() >= time_limit:
                if self.verbose:
                    print(f"Time limit reached at iteration {it}")
                break

        if self.verbose:
            self.logger.print_final_report(self.best_solution.cost)
        return self.best_solution
