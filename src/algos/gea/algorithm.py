from src.algos.core.improved_base import ImprovedBase
from src.algos.core.standard_base import StandardBase


class StandardGEA(StandardBase):
    """Full GEA operator set on the thesis scaffold with pool survivor selection."""

    def __init__(
        self,
        model,
        population_size=350,
        iterations=1000,
        crossover_rate=0.7,
        mutation_rate=0.01,
        rc_rate=0.3,
        dm_rate=0.3,
        injection_rate=0.1,
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
        self.rc_rate = rc_rate
        self.dm_rate = dm_rate
        self.injection_rate = injection_rate

    def step(self) -> None:
        self.run_gea_generational(
            self.crossover_rate,
            self.mutation_rate,
            rc_rate=self.rc_rate,
            dm_rate=self.dm_rate,
            injection_rate=self.injection_rate,
        )


class ImprovedGEA(ImprovedBase):
    """Full GEA: diversity selection, RF repair, stagnation immigrants, memetic local search,
    and five fixed-count operator stages (crossover, mutation, RC, DM, GI)."""

    def __init__(
        self,
        model,
        population_size=350,
        iterations=1000,
        crossover_rate=0.7,
        mutation_rate=0.3,
        rc_rate=0.3,
        dm_rate=0.3,
        injection_rate=0.1,
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
        self.rc_rate = rc_rate
        self.dm_rate = dm_rate
        self.injection_rate = injection_rate

    def step(self) -> None:
        self.run_gea_generation(
            self.crossover_rate,
            self.mutation_rate,
            rc_rate=self.rc_rate,
            dm_rate=self.dm_rate,
            injection_rate=self.injection_rate,
        )
