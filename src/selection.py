from typing import List

import numpy as np

from src.data.models import Individual


def get_diversity(population_base: List[Individual], population_to_eval: List[Individual]) -> np.ndarray:
    """Mean Hamming distance (normalized by J) from each individual in population_to_eval
    to population_base, as a fraction of J.

    The mean over population_base only depends on per-position value frequencies in
    population_base, not on N_base*N_eval pairwise comparisons: avg_b Hamming(e, b) at
    position j is 1 - (count of b with base[b,j] == eval[e,j]) / N_base, summed over j.
    That's O((N_base + N_eval) * J) instead of the naive O(N_base * N_eval * J) -- same
    exact result, no precision/approximation tradeoff, just less redundant work.
    """
    if not population_to_eval:
        return np.array([], dtype=np.float32)

    base_perms = np.array([ind.permutation for ind in population_base], dtype=np.int32)  # (N_base, J)
    eval_perms = np.array([ind.permutation for ind in population_to_eval], dtype=np.int32)  # (N_eval, J)

    n_base, J = base_perms.shape
    num_facilities = int(max(base_perms.max(), eval_perms.max())) + 1

    # freq[j, v] = how many individuals in population_base have value v at position j
    freq = np.zeros((J, num_facilities), dtype=np.int32)
    j_grid = np.broadcast_to(np.arange(J), base_perms.shape)
    np.add.at(freq, (j_grid, base_perms), 1)

    # agreement_sum[e] = sum_j freq[j, eval_perms[e, j]] = total agreements with base, summed over j and averaged over b
    j_idx = np.arange(J)
    agreement_sum = freq[j_idx[None, :], eval_perms].sum(axis=1)

    avg_hamming = J - agreement_sum / n_base
    return (avg_hamming / J).astype(np.float32)


# Parent selection (diversity-weighted roulette wheel) and survivor selection
# (elite + diversity/cost hybrid, decaying toward pure cost over the run).
class DiversitySelector:
    def __init__(
        self,
        beta: float = 10.0,
        diversity_weight_start: float = 0.7,
        diversity_weight_end: float = 0.2,
        elite_fraction: float = 0.3,
    ):
        self.beta = beta
        self.diversity_weight_start = diversity_weight_start
        self.diversity_weight_end = diversity_weight_end
        self.elite_fraction = elite_fraction
        self.avg_diversity: float = 0.0

    def compute_selection_probabilities(self, population: List[Individual]) -> np.ndarray:
        diversity_scores = get_diversity(population_base=population, population_to_eval=population)

        probs = np.exp(self.beta * diversity_scores)
        probs = probs / probs.sum()

        self.avg_diversity = np.mean(diversity_scores)
        return probs

    def roulette_wheel_selection(self, probabilities: np.ndarray) -> int:
        cumsum = np.cumsum(probabilities)
        idx = np.searchsorted(cumsum, np.random.random(), side="right")
        return int(min(idx, len(probabilities) - 1))

    def roulette_wheel_selection_batch(self, probabilities: np.ndarray, size: int) -> np.ndarray:
        """Draw `size` parent indices at once. probabilities is constant across all of them
        within a generation, so the O(N) cumsum is computed once here instead of once per
        draw (the single-draw roulette_wheel_selection above recomputes it every call)."""
        cumsum = np.cumsum(probabilities)
        draws = np.random.random(size=size)
        # cumsum[-1] can be a hair below 1.0 due to float rounding, so a draw just under 1.0
        # can land past it -- searchsorted would then return len(probabilities), out of range.
        return np.minimum(np.searchsorted(cumsum, draws, side="right"), len(probabilities) - 1)

    def select_from_pool(self, pool: List[Individual], population_size: int, progress: float) -> List[Individual]:
        unique_pool = list(dict.fromkeys(pool))
        n = population_size

        # Sort by cost (lower is better)
        unique_pool.sort(key=lambda x: x.cost)

        # A smaller elite carve-out than half leaves more survivor slots decided by the
        # cost+diversity blend below instead of pure cost, so the population doesn't lock
        # onto a local optimum's neighborhood as fast.
        n_elite = max(1, int(self.elite_fraction * n))
        elite = unique_pool[:n_elite]
        remaining = unique_pool[n_elite:]

        diversity_array = get_diversity(population_base=elite, population_to_eval=remaining)

        # Normalize costs so lower cost -> higher score
        costs = np.array([ind.cost for ind in remaining], dtype=float)

        finite = np.isfinite(costs)
        finite_costs = costs[finite]

        if finite_costs.size > 1 and finite_costs.max() != finite_costs.min():
            cost_scores = np.zeros_like(costs)
            cost_scores[finite] = 1.0 - (finite_costs - finite_costs.min()) / (finite_costs.max() - finite_costs.min())
        else:
            cost_scores = finite.astype(float)

        # Decay diversity pressure over the run so the population can settle
        # onto the best cost found instead of being pinned apart by diversity.
        diversity_weight = (
            self.diversity_weight_start - (self.diversity_weight_start - self.diversity_weight_end) * progress
        )
        cost_weight = 1.0 - diversity_weight

        combined_scores = diversity_weight * diversity_array + cost_weight * cost_scores

        scored_remaining = list(zip(remaining, combined_scores))
        scored_remaining.sort(key=lambda x: x[1], reverse=True)

        n_diverse = n - n_elite
        diverse_selected = [ind for ind, _ in scored_remaining[:n_diverse]]

        return elite + diverse_selected
