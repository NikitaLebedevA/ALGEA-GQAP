import math
from typing import List, Tuple

import numpy as np

from src.data.models import Individual


class AnnealingMixin:
    """Metropolis acceptance and geometric cooling shared by SA and GA+SA hybrids."""

    def _init_annealing(
        self,
        initial_temperature: float,
        cooling_rate: float,
        min_temperature: float,
    ) -> None:
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.temperature = initial_temperature

    def _anneal_accept(self, pairs: List[Tuple[Individual, Individual]]) -> List[Individual]:
        accepted = []
        for child, baseline in pairs:
            if child.cost <= baseline.cost:
                accepted.append(child)
                continue

            if not (math.isfinite(child.cost) and math.isfinite(baseline.cost)):
                continue

            delta = child.cost - baseline.cost
            if np.random.random() < math.exp(-delta / self.temperature):
                accepted.append(child)

        return accepted

    def _cool_temperature(self) -> None:
        self.temperature = max(self.min_temperature, self.temperature * self.cooling_rate)
