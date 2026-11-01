import math

from src.algos.core.improved_base import ImprovedBase
from src.algos.mixins.pso import PSOMixin
from src.algos.core.standard_base import StandardBase


class StandardHybridGAPSO(StandardBase, PSOMixin):
    """Rank-split hybrid: discrete PSO on the best fraction, Holland GA on the rest."""

    def __init__(
        self,
        model,
        population_size=350,
        iterations=1000,
        pso_fraction=0.5,
        inertia_weight=0.4,
        cognitive_weight=0.3,
        social_weight=0.3,
        crossover_rate=0.7,
        mutation_rate=0.3,
        repair_class=None,
        verbose=False,
    ):
        super().__init__(
            model,
            population_size,
            iterations,
            repair_class=repair_class,
            verbose=verbose,
        )
        self.pso_fraction = pso_fraction
        total_weight = inertia_weight + cognitive_weight + social_weight
        self.inertia_weight = inertia_weight / total_weight
        self.cognitive_weight = cognitive_weight / total_weight
        self.social_weight = social_weight / total_weight
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self._init_pso_tracking(use_velocities=False)

    def initialize_population(self) -> None:
        super().initialize_population()
        self._sync_pso_from_population()

    def _standard_hybrid_ga_update(self, indices) -> None:
        original_population = self.population
        self.population = [self.particles[i] for i in indices]
        n = len(indices)

        ncrossover = int(2 * round((self.crossover_rate * n) / 2))
        nmutation = int(math.floor(self.mutation_rate * n))
        offspring = [child for child, _ in self._standard_crossover_pairs(ncrossover)]
        mutations = [child for child, _ in self._standard_mutate_pairs(nmutation)]

        pool = self.population + offspring + mutations
        selected = sorted(pool, key=lambda x: x.cost)[:n]
        self.population = original_population

        for slot, ind in zip(indices, selected):
            self.particles[slot] = ind
            if ind.cost < self.personal_best[slot].cost:
                self.personal_best[slot] = ind

    def step(self) -> None:
        gbest = self.best_solution
        n = self.population_size
        n_pso = int(round(self.pso_fraction * n))

        order = sorted(range(n), key=lambda i: self.particles[i].cost)
        pso_idx = order[:n_pso]
        ga_idx = order[n_pso:]

        if pso_idx:
            self._discrete_pso_update(
                pso_idx,
                gbest,
                inertia_weight=self.inertia_weight,
                cognitive_weight=self.cognitive_weight,
                social_weight=self.social_weight,
            )
        if ga_idx:
            self._standard_hybrid_ga_update(ga_idx)

        self._enforce_global_best_elite(gbest)
        self._commit_particles_to_population()


class ImprovedHybridGAPSO(ImprovedBase, PSOMixin):
    """Rank-split hybrid with GEA-style GA regeneration and stagnation immigrants."""

    def __init__(
        self,
        model,
        population_size=350,
        iterations=1000,
        pso_fraction=0.5,
        inertia_weight=0.4,
        cognitive_weight=0.3,
        social_weight=0.3,
        crossover_rate=0.7,
        mutation_rate=0.3,
        repair_class=None,
        selector=None,
        stagnation_limit=30,
        immigrant_rate=0.1,
        verbose=False,
    ):
        super().__init__(
            model,
            population_size,
            iterations,
            repair_class=repair_class,
            selector=selector,
            stagnation_limit=stagnation_limit,
            immigrant_rate=immigrant_rate,
            verbose=verbose,
        )
        self.pso_fraction = pso_fraction
        total_weight = inertia_weight + cognitive_weight + social_weight
        self.inertia_weight = inertia_weight / total_weight
        self.cognitive_weight = cognitive_weight / total_weight
        self.social_weight = social_weight / total_weight
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self._init_pso_tracking(use_velocities=False)

    def initialize_population(self) -> None:
        super().initialize_population()
        self._sync_pso_from_population()

    def step(self) -> None:
        gbest = self.best_solution
        n = self.population_size
        n_pso = int(round(self.pso_fraction * n))

        order = sorted(range(n), key=lambda i: self.particles[i].cost)
        pso_idx = order[:n_pso]
        ga_idx = order[n_pso:]

        if pso_idx:
            self._discrete_pso_update(
                pso_idx,
                gbest,
                inertia_weight=self.inertia_weight,
                cognitive_weight=self.cognitive_weight,
                social_weight=self.social_weight,
            )
        if ga_idx:
            self._hybrid_ga_update(ga_idx, self.crossover_rate, self.mutation_rate)

        self._enforce_global_best_elite(gbest)
        self._inject_pso_immigrants(self.maybe_generate_immigrants())
        self._commit_particles_to_population()
