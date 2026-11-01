from src.algos.core.improved_base import ImprovedBase
from src.algos.core.standard_base import StandardBase


class StandardGA(StandardBase):
    """Holland (1992). Textbook GA on the standard scaffold: exponential parent selection,
    fixed-batch crossover/mutation, and (mu+lambda) pool survivor selection."""

    def __init__(
        self,
        model,
        population_size=350,
        iterations=1000,
        crossover_rate=0.7,
        mutation_rate=0.01,
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
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def step(self) -> None:
        self.run_crossover_mutation_generational(self.crossover_rate, self.mutation_rate)


class ImprovedGA(ImprovedBase):
    """GEA-family GA: diversity selection, RF repair, stagnation immigrants, and memetic
    local search, with only standard crossover and mutation — no RC/DM/GI scenario operators.
    """

    def __init__(
        self,
        model,
        population_size=350,
        iterations=1000,
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
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def step(self) -> None:
        self.run_crossover_mutation_generation(self.crossover_rate, self.mutation_rate)
