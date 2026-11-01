from src.algos.mixins.annealing import AnnealingMixin
from src.algos.core.improved_base import ImprovedBase
from src.algos.core.standard_base import StandardBase


class StandardHybridGASA(StandardBase, AnnealingMixin):
    """Thesis-style GA with Metropolis acceptance and pool survivor selection."""

    def __init__(
        self,
        model,
        population_size=350,
        iterations=1000,
        crossover_rate=0.7,
        mutation_rate=0.3,
        initial_temperature=50.0,
        cooling_rate=0.97,
        min_temperature=1e-3,
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
        self._init_annealing(initial_temperature, cooling_rate, min_temperature)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def step(self) -> None:
        self.run_annealing_generational(self.crossover_rate, self.mutation_rate)


class ImprovedHybridGASA(ImprovedBase, AnnealingMixin):
    """GEA-style GA with Metropolis acceptance and pool-based survivor selection."""

    def __init__(
        self,
        model,
        population_size=350,
        iterations=1000,
        crossover_rate=0.7,
        mutation_rate=0.3,
        initial_temperature=50.0,
        cooling_rate=0.97,
        min_temperature=1e-3,
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
        self._init_annealing(initial_temperature, cooling_rate, min_temperature)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def step(self) -> None:
        self.run_annealing_generation(self.crossover_rate, self.mutation_rate)
