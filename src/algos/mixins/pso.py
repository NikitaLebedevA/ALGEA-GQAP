import math

import numpy as np

from src.costs import evaluate_permutation_delta_batch
from src.operators.crossover import choose_crossover
from src.operators.mutations import choose_mutation


class PSOMixin:
    """Particle/personal-best tracking and PSO move operators."""

    def _init_pso_tracking(self, *, use_velocities: bool = False) -> None:
        self.particles = []
        self.personal_best = []
        self.velocities = [] if use_velocities else None

    def _sync_pso_from_population(self) -> None:
        self.particles = list(self.population)
        self.personal_best = list(self.population)
        if self.velocities is not None:
            self.velocities = [np.zeros(self.model.J, dtype=float) for _ in range(self.population_size)]

    def _inject_pso_immigrants(self, immigrants) -> None:
        if not immigrants:
            return

        replace_idx = np.random.choice(self.population_size, size=len(immigrants), replace=False)
        for idx, immigrant in zip(replace_idx, immigrants):
            self.particles[idx] = immigrant
            self.personal_best[idx] = immigrant
            if self.velocities is not None:
                self.velocities[idx] = np.zeros(self.model.J, dtype=float)

    def _commit_particles_to_population(self) -> None:
        self.population = list(self.particles)

    def _velocity_pso_step(
        self,
        *,
        inertia_weight: float,
        cognitive_weight: float,
        social_weight: float,
        v_max: float,
    ) -> None:
        gbest = self.best_solution
        I = self.model.I
        J = self.model.J

        new_perms = []
        for k in range(len(self.particles)):
            particle = self.particles[k]
            pbest = self.personal_best[k]
            vel = self.velocities[k]

            r1 = np.random.random(J)
            r2 = np.random.random(J)

            new_vel = (
                inertia_weight * vel
                + cognitive_weight * r1 * (pbest.permutation.astype(float) - particle.permutation.astype(float))
                + social_weight * r2 * (gbest.permutation.astype(float) - particle.permutation.astype(float))
            )
            new_vel = np.clip(new_vel, -v_max, v_max)
            self.velocities[k] = new_vel

            new_pos = np.round(particle.permutation.astype(float) + new_vel).astype(np.int64)
            new_pos = np.clip(new_pos, 0, I - 1)
            new_perms.append(new_pos)

        repaired = self.repair_batch_wrapper(np.array(new_perms))
        updated = evaluate_permutation_delta_batch(self.particles, repaired, self.model)
        self.logger.record_nfe(len(self.particles))

        for i, candidate in enumerate(updated):
            self.particles[i] = candidate
            if candidate.cost < self.personal_best[i].cost:
                self.personal_best[i] = candidate

    def _discrete_pso_update(self, indices, gbest, *, inertia_weight, cognitive_weight, social_weight) -> None:
        new_perms = []
        for i in indices:
            position = self.particles[i]
            pbest = self.personal_best[i]

            r = np.random.random()
            if r < inertia_weight:
                candidate = choose_mutation(position.permutation, self.model)
            elif r < inertia_weight + cognitive_weight:
                (child1, _), _ = choose_crossover((position, pbest), self.model)
                candidate = child1
            else:
                (child1, _), _ = choose_crossover((position, gbest), self.model)
                candidate = child1
            new_perms.append(candidate)

        repaired = self.repair_batch_wrapper(np.array(new_perms))
        baselines = [self.particles[i] for i in indices]
        updated = evaluate_permutation_delta_batch(baselines, repaired, self.model)
        self.logger.record_nfe(len(indices))

        for i, candidate in zip(indices, updated):
            self.particles[i] = candidate
            if candidate.cost < self.personal_best[i].cost:
                self.personal_best[i] = candidate

    def _hybrid_ga_update(self, indices, crossover_rate, mutation_rate) -> None:
        original_population = self.population
        self.population = [self.particles[i] for i in indices]

        probs = self.compute_selection_probabilities()
        n = len(indices)
        ncrossover = int(2 * round((crossover_rate * n) / 2))
        nmutation = int(math.floor(mutation_rate * n))

        offspring = [child for child, _ in self.crossover(probs, ncrossover)]
        mutations = [child for child, _ in self.mutate(nmutation)]

        pool = self.population + offspring + mutations
        selected = self.selector.select_from_pool(pool, n, self.progress)

        self.population = original_population

        for slot, ind in zip(indices, selected):
            self.particles[slot] = ind
            if ind.cost < self.personal_best[slot].cost:
                self.personal_best[slot] = ind

    def _enforce_global_best_elite(self, gbest) -> None:
        n = self.population_size
        worst_idx = max(range(n), key=lambda i: self.particles[i].cost)
        if gbest is not None and gbest.cost < self.particles[worst_idx].cost:
            self.particles[worst_idx] = gbest
            if gbest.cost < self.personal_best[worst_idx].cost:
                self.personal_best[worst_idx] = gbest
