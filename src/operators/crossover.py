from typing import Sequence, Tuple

import numpy as np
from numba import njit

from src.data.models import Individual, Model

# Each child paired with whichever parent it shares the most positions with -- the
# tightest baseline for delta-cost evaluation (see evaluate_permutation_delta).
CrossoverResult = Tuple[Tuple[np.ndarray, Individual], Tuple[np.ndarray, Individual]]


@njit(fastmath=True, cache=True)
def crossover_one_point_nb(p1: np.ndarray, p2: np.ndarray, point: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(p1)

    child1 = np.empty(n, dtype=p1.dtype)
    child2 = np.empty(n, dtype=p1.dtype)

    child1[:point] = p1[:point]
    child1[point:] = p2[point:]

    child2[:point] = p2[:point]
    child2[point:] = p1[point:]

    return child1, child2


@njit(fastmath=True, cache=True)
def crossover_two_point_nb(p1: np.ndarray, p2: np.ndarray, a: int, b: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(p1)

    child1 = np.empty(n, dtype=p1.dtype)
    child2 = np.empty(n, dtype=p1.dtype)

    child1[:a] = p2[:a]
    child1[a:b] = p1[a:b]
    child1[b:] = p2[b:]

    child2[:a] = p1[:a]
    child2[a:b] = p2[a:b]
    child2[b:] = p1[b:]

    return child1, child2


@njit(fastmath=True, cache=True)
def crossover_uniform_nb(p1: np.ndarray, p2: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    n = len(p1)

    child1 = np.empty(n, dtype=p1.dtype)
    child2 = np.empty(n, dtype=p1.dtype)
    n_match_p1 = 0

    for j in range(n):
        if mask[j]:
            child1[j] = p1[j]
            child2[j] = p2[j]
            n_match_p1 += 1
        else:
            child1[j] = p2[j]
            child2[j] = p1[j]

    return child1, child2, n_match_p1


@njit(fastmath=True, cache=True)
def crossover_greedy_nb(p1: np.ndarray, p2: np.ndarray, cij: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """Per-position, child1 takes whichever parent's assignment is cheaper for that job
    (exploitation); child2 takes the other one (the complementary, worse-of pairing) so the
    pair still spans the same diversity a random crossover would."""
    n = len(p1)

    child1 = np.empty(n, dtype=p1.dtype)
    child2 = np.empty(n, dtype=p1.dtype)
    n_match_p1 = 0

    for j in range(n):
        if cij[p1[j], j] <= cij[p2[j], j]:
            child1[j] = p1[j]
            child2[j] = p2[j]
            n_match_p1 += 1
        else:
            child1[j] = p2[j]
            child2[j] = p1[j]

    return child1, child2, n_match_p1


def crossover_one_point(parents: Sequence[Individual], model: Model) -> CrossoverResult:
    p1, p2 = parents[0], parents[1]
    n = len(p1.permutation)

    point = np.random.randint(1, n)
    child1, child2 = crossover_one_point_nb(p1.permutation, p2.permutation, point)

    # child1 = p1's prefix + p2's tail -> matches p1 outside [point:]
    # child2 = p2's prefix + p1's tail -> matches p2 outside [point:]
    return (child1, p1), (child2, p2)


def crossover_two_point(parents: Sequence[Individual], model: Model) -> CrossoverResult:
    p1, p2 = parents[0], parents[1]
    n = len(p1.permutation)

    idx = np.random.randint(1, n, size=2)
    a, b = np.sort(idx)

    child1, child2 = crossover_two_point_nb(p1.permutation, p2.permutation, a, b)

    # child1 = p2 with a p1-segment spliced into [a:b) -> matches p2 outside [a:b)
    # child2 = p1 with a p2-segment spliced into [a:b) -> matches p1 outside [a:b)
    return (child1, p2), (child2, p1)


def _pair_by_match_count(child1: np.ndarray, child2: np.ndarray, n_match_p1: int, n: int, p1: Individual, p2: Individual) -> CrossoverResult:
    # child1 matches p1 at n_match_p1 positions and p2 at the rest; child2 is the mirror
    # image (matches p2 at n_match_p1, p1 at the rest) -- pair each with its closer parent.
    if n_match_p1 >= n - n_match_p1:
        return (child1, p1), (child2, p2)
    return (child1, p2), (child2, p1)


def crossover_uniform(parents: Sequence[Individual], model: Model) -> CrossoverResult:
    p1, p2 = parents[0], parents[1]
    n = len(p1.permutation)

    mask = np.random.random(n) < 0.5
    child1, child2, n_match_p1 = crossover_uniform_nb(p1.permutation, p2.permutation, mask)

    return _pair_by_match_count(child1, child2, n_match_p1, n, p1, p2)


def crossover_greedy(parents: Sequence[Individual], model: Model) -> CrossoverResult:
    p1, p2 = parents[0], parents[1]
    n = len(p1.permutation)

    child1, child2, n_match_p1 = crossover_greedy_nb(p1.permutation, p2.permutation, model.cij)

    return _pair_by_match_count(child1, child2, n_match_p1, n, p1, p2)


@njit(fastmath=True, cache=True)
def _crossover_robust_chromosome_nb(p1, p2, cvar1, cvar2):
    """Per gene, inherit from whichever parent's assigned facility has more remaining
    capacity slack (cvar) -- the more capacity-robust gene, independent of assignment cost.
    Used by GEA and GEAScenario1 (Robust Chromosome crossover)."""
    n = len(p1)
    child1 = np.empty(n, dtype=p1.dtype)
    child2 = np.empty(n, dtype=p1.dtype)
    n_match_p1 = 0

    for j in range(n):
        if cvar1[p1[j]] >= cvar2[p2[j]]:
            child1[j] = p1[j]
            child2[j] = p2[j]
            n_match_p1 += 1
        else:
            child1[j] = p2[j]
            child2[j] = p1[j]

    return child1, child2, n_match_p1


def crossover_robust_chromosome(p1: Individual, p2: Individual, model: Model) -> CrossoverResult:
    """Robust Chromosome (RC) crossover: per gene, the child inherits from whichever parent's
    assigned facility has more remaining capacity slack. Used by GEA and GEAScenario1."""
    n = len(p1.permutation)
    child1, child2, n_match_p1 = _crossover_robust_chromosome_nb(p1.permutation, p2.permutation, p1.cvar, p2.cvar)
    return _pair_by_match_count(child1, child2, n_match_p1, n, p1, p2)


# Adding a new crossover operator is just adding it here -- choose_crossover picks
# uniformly among whatever is listed, with no separate count to remember to update.
CROSSOVER_OPERATORS = (
    crossover_one_point,
    crossover_two_point,
    crossover_uniform,
    crossover_greedy,
)


def choose_crossover(parents: Sequence[Individual], model: Model) -> CrossoverResult:
    op = CROSSOVER_OPERATORS[np.random.randint(len(CROSSOVER_OPERATORS))]
    return op(parents, model)
