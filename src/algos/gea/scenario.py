from src.algos.core.improved_base import ImprovedBase
from src.algos.core.standard_base import StandardBase


class StandardGEAScenario1(StandardBase):
    """Scenario 1 (thesis dominant crossover) on the standard scaffold."""

    def __init__(
        self,
        model,
        population_size=350,
        iterations=1000,
        crossover_rate=0.7,
        mutation_rate=0.01,
        rc_rate=0.3,
        repair_class=None,
        verbose=False,
    ):
        super().__init__(model, population_size, iterations, repair_class=repair_class, verbose=verbose)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.rc_rate = rc_rate

    def step(self) -> None:
        self.run_batch_generational(
            self.crossover_rate,
            self.mutation_rate,
            enable_scenario1=True,
            scenario_crossover_rate=self.rc_rate,
        )


class ImprovedGEAScenario1(ImprovedBase):
    """Scenario 1 (RC crossover) on the GEA-family scaffold."""

    def __init__(
        self,
        model,
        population_size=350,
        iterations=1000,
        crossover_rate=0.7,
        mutation_rate=0.3,
        rc_rate=0.3,
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

    def step(self) -> None:
        self.run_gea_generation(self.crossover_rate, self.mutation_rate, rc_rate=self.rc_rate)


class StandardGEAScenario2(StandardBase):
    """Scenario 2 (thesis mask mutation) on the standard scaffold."""

    def __init__(
        self,
        model,
        population_size=350,
        iterations=1000,
        crossover_rate=0.7,
        mutation_rate=0.01,
        dm_rate=0.3,
        repair_class=None,
        verbose=False,
    ):
        super().__init__(model, population_size, iterations, repair_class=repair_class, verbose=verbose)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.dm_rate = dm_rate

    def step(self) -> None:
        self.run_batch_generational(
            self.crossover_rate,
            self.mutation_rate,
            enable_scenario2=True,
            scenario_mutation_rate=self.dm_rate,
        )


class ImprovedGEAScenario2(ImprovedBase):
    """Scenario 2 (directed mutation) on the GEA-family scaffold."""

    def __init__(
        self,
        model,
        population_size=350,
        iterations=1000,
        crossover_rate=0.7,
        mutation_rate=0.3,
        dm_rate=0.3,
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
        self.dm_rate = dm_rate

    def step(self) -> None:
        self.run_gea_generation(self.crossover_rate, self.mutation_rate, dm_rate=self.dm_rate)


class StandardGEAScenario3(StandardBase):
    """Scenario 3 (thesis combine_q) on the standard scaffold."""

    def __init__(
        self,
        model,
        population_size=350,
        iterations=1000,
        crossover_rate=0.7,
        mutation_rate=0.01,
        injection_rate=0.1,
        repair_class=None,
        verbose=False,
    ):
        super().__init__(model, population_size, iterations, repair_class=repair_class, verbose=verbose)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.injection_rate = injection_rate

    def step(self) -> None:
        self.run_batch_generational(
            self.crossover_rate,
            self.mutation_rate,
            enable_scenario3=True,
            scenario_mutation_rate_3=self.injection_rate,
        )


class ImprovedGEAScenario3(ImprovedBase):
    """Scenario 3 (gene injection) on the GEA-family scaffold."""

    def __init__(
        self,
        model,
        population_size=350,
        iterations=1000,
        crossover_rate=0.7,
        mutation_rate=0.3,
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
        self.injection_rate = injection_rate

    def step(self) -> None:
        self.run_gea_generation(
            self.crossover_rate,
            self.mutation_rate,
            injection_rate=self.injection_rate,
        )
