from src.algos.mixins.adaptive import AdaptiveRatesMixin
from src.algos.core.improved_base import ImprovedBase
from src.algos.core.standard_base import StandardBase


class StandardAdaptiveGEAScenario1(StandardBase, AdaptiveRatesMixin):
    def __init__(
        self,
        model,
        population_size=350,
        iterations=1000,
        crossover_rate=0.7,
        mutation_rate=0.3,
        rc_rate=0.3,
        alpha=0.01,
        lambda_min=0.1,
        lambda_max=0.9,
        epsilon=1e-5,
        fixed_operators=(),
        reward_horizon=10,
        repair_class=None,
        verbose=False,
    ):
        super().__init__(model, population_size, iterations, repair_class=repair_class, verbose=verbose)
        self._init_adaptive_rates(
            crossover_rate, mutation_rate, alpha, lambda_min, lambda_max, epsilon, fixed_operators, reward_horizon
        )
        self._init_adaptive_scenario_rate("base_rc", "lambda_rc", rc_rate)

    def step(self) -> None:
        self.run_adaptive_generational(rc_rate=self.base_rc)


class ImprovedAdaptiveGEAScenario1(ImprovedBase, AdaptiveRatesMixin):
    def __init__(
        self,
        model,
        population_size=350,
        iterations=1000,
        crossover_rate=0.7,
        mutation_rate=0.3,
        rc_rate=0.3,
        alpha=0.01,
        lambda_min=0.1,
        lambda_max=0.9,
        epsilon=1e-5,
        fixed_operators=(),
        reward_horizon=10,
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
        self._init_adaptive_rates(
            crossover_rate, mutation_rate, alpha, lambda_min, lambda_max, epsilon, fixed_operators, reward_horizon
        )
        self._init_adaptive_scenario_rate("base_rc", "lambda_rc", rc_rate)

    def step(self) -> None:
        self.run_adaptive_generation(rc_rate=self.base_rc)


class StandardAdaptiveGEAScenario2(StandardBase, AdaptiveRatesMixin):
    def __init__(
        self,
        model,
        population_size=350,
        iterations=1000,
        crossover_rate=0.7,
        mutation_rate=0.3,
        dm_rate=0.3,
        alpha=0.01,
        lambda_min=0.1,
        lambda_max=0.9,
        epsilon=1e-5,
        fixed_operators=(),
        reward_horizon=10,
        repair_class=None,
        verbose=False,
    ):
        super().__init__(model, population_size, iterations, repair_class=repair_class, verbose=verbose)
        self._init_adaptive_rates(
            crossover_rate, mutation_rate, alpha, lambda_min, lambda_max, epsilon, fixed_operators, reward_horizon
        )
        self._init_adaptive_scenario_rate("base_dm", "lambda_dm", dm_rate)

    def step(self) -> None:
        self.run_adaptive_generational(dm_rate=self.base_dm)


class ImprovedAdaptiveGEAScenario2(ImprovedBase, AdaptiveRatesMixin):
    def __init__(
        self,
        model,
        population_size=350,
        iterations=1000,
        crossover_rate=0.7,
        mutation_rate=0.3,
        dm_rate=0.3,
        alpha=0.01,
        lambda_min=0.1,
        lambda_max=0.9,
        epsilon=1e-5,
        fixed_operators=(),
        reward_horizon=10,
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
        self._init_adaptive_rates(
            crossover_rate, mutation_rate, alpha, lambda_min, lambda_max, epsilon, fixed_operators, reward_horizon
        )
        self._init_adaptive_scenario_rate("base_dm", "lambda_dm", dm_rate)

    def step(self) -> None:
        self.run_adaptive_generation(dm_rate=self.base_dm)


class StandardAdaptiveGEAScenario3(StandardBase, AdaptiveRatesMixin):
    def __init__(
        self,
        model,
        population_size=350,
        iterations=1000,
        crossover_rate=0.7,
        mutation_rate=0.3,
        injection_rate=0.1,
        alpha=0.01,
        lambda_min=0.1,
        lambda_max=0.9,
        epsilon=1e-5,
        fixed_operators=(),
        reward_horizon=10,
        repair_class=None,
        verbose=False,
    ):
        super().__init__(model, population_size, iterations, repair_class=repair_class, verbose=verbose)
        self._init_adaptive_rates(
            crossover_rate, mutation_rate, alpha, lambda_min, lambda_max, epsilon, fixed_operators, reward_horizon
        )
        self._init_adaptive_scenario_rate("base_gi", "lambda_gi", injection_rate)

    def step(self) -> None:
        self.run_adaptive_generational(injection_rate=self.base_gi)


class ImprovedAdaptiveGEAScenario3(ImprovedBase, AdaptiveRatesMixin):
    def __init__(
        self,
        model,
        population_size=350,
        iterations=1000,
        crossover_rate=0.7,
        mutation_rate=0.3,
        injection_rate=0.1,
        alpha=0.01,
        lambda_min=0.1,
        lambda_max=0.9,
        epsilon=1e-5,
        fixed_operators=(),
        reward_horizon=10,
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
        self._init_adaptive_rates(
            crossover_rate, mutation_rate, alpha, lambda_min, lambda_max, epsilon, fixed_operators, reward_horizon
        )
        self._init_adaptive_scenario_rate("base_gi", "lambda_gi", injection_rate)

    def step(self) -> None:
        self.run_adaptive_generation(injection_rate=self.base_gi)
