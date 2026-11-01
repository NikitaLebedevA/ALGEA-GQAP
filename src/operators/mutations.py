import numpy as np
from numba import njit

from src.data.models import Model


@njit(fastmath=True, cache=True)
def mutation_swap_nb(perm: np.ndarray) -> np.ndarray:
    n = len(perm)

    i = np.random.randint(0, n - 1)
    result = perm.copy()
    result[i], result[i + 1] = result[i + 1], result[i]
    return result


@njit(fastmath=True, cache=True)
def mutation_reversion_nb(perm: np.ndarray) -> np.ndarray:
    n = len(perm)

    i, j = np.sort(np.random.choice(n, 2, replace=False))
    result = perm.copy()
    result[i : j + 1] = result[i : j + 1][::-1]
    return result


@njit(fastmath=True, cache=True)
def mutation_insertion_nb(perm: np.ndarray) -> np.ndarray:
    n = len(perm)

    idx = np.sort(np.random.choice(np.arange(1, n), 2, replace=False))
    i, j = idx[0], idx[1]

    result = np.empty(n, dtype=np.int64)
    pos = 0
    result[pos : pos + (j - i + 1)] = perm[i : j + 1]
    pos += j - i + 1
    result[pos : pos + i] = perm[:i]
    pos += i
    result[pos:] = perm[j + 1 :]

    return result


@njit(fastmath=True, cache=True)
def mutation_big_swap_nb(perm: np.ndarray) -> np.ndarray:
    n = len(perm)

    i, j = np.random.choice(n, 2, replace=False)
    result = perm.copy()
    result[i], result[j] = result[j], result[i]
    return result


@njit(fastmath=True, cache=True)
def mutation_random_nb(perm: np.ndarray, max_value: int) -> np.ndarray:
    n = len(perm)

    result = perm.copy()
    num = np.random.randint(1, min(6, n) + 1)
    indices = np.random.choice(n, num, replace=False)

    for k in range(num):
        result[indices[k]] = np.random.randint(0, max_value)

    return result


@njit(fastmath=True, cache=True)
def mutation_scramble_nb(perm: np.ndarray) -> np.ndarray:
    n = len(perm)

    k = np.random.randint(3, min(6, n) + 1)
    indices = np.random.choice(n, k, replace=False)

    result = perm.copy()
    values = result[indices].copy()
    for a in range(k - 1, 0, -1):
        b = np.random.randint(0, a + 1)
        values[a], values[b] = values[b], values[a]

    for t in range(k):
        result[indices[t]] = values[t]

    return result


@njit(fastmath=True, cache=True)
def mutation_cyclic_shift_nb(perm: np.ndarray) -> np.ndarray:
    idx = np.random.choice(len(perm), 3, replace=False)
    i, j, k = idx[0], idx[1], idx[2]

    result = perm.copy()
    result[i], result[j], result[k] = perm[k], perm[i], perm[j]
    return result


@njit(fastmath=True, cache=True)
def mutation_greedy_reassign_nb(perm: np.ndarray, aij: np.ndarray, cij: np.ndarray, bi: np.ndarray) -> np.ndarray:
    """Targets the single job with the worst current assignment cost and moves it to the
    cheapest facility with spare capacity for it -- a directed move (one step of
    local_search's per-job hill-climb) rather than the blind random moves above."""
    I = aij.shape[0]
    J = perm.shape[0]

    loads = np.zeros(I, dtype=aij.dtype)
    for j in range(J):
        loads[perm[j]] += aij[perm[j], j]

    worst_j = 0
    worst_cost = cij[perm[0], 0]
    for j in range(1, J):
        c = cij[perm[j], j]
        if c > worst_cost:
            worst_cost = c
            worst_j = j

    cur_i = perm[worst_j]
    slack = bi - loads
    slack[cur_i] += aij[cur_i, worst_j]

    best_i = cur_i
    best_cost = cij[cur_i, worst_j]
    for i in range(I):
        if i == cur_i:
            continue
        if slack[i] >= aij[i, worst_j] and cij[i, worst_j] < best_cost:
            best_cost = cij[i, worst_j]
            best_i = i

    result = perm.copy()
    result[worst_j] = best_i
    return result


@njit(fastmath=True, cache=True)
def mutation_migration_nb(perm: np.ndarray, aij: np.ndarray, bi: np.ndarray) -> np.ndarray:
    """Capacity-aware: moves the priciest job out of the most-loaded facility into the
    facility with the most spare capacity, nudging the assignment toward feasibility before
    repair runs."""
    I = aij.shape[0]
    J = perm.shape[0]

    loads = np.zeros(I, dtype=aij.dtype)
    for j in range(J):
        loads[perm[j]] += aij[perm[j], j]
    slack = bi - loads

    src_i = 0
    min_slack = slack[0]
    for i in range(1, I):
        if slack[i] < min_slack:
            min_slack = slack[i]
            src_i = i

    job = -1
    job_cost = -1.0
    for j in range(J):
        if perm[j] == src_i and aij[src_i, j] > job_cost:
            job_cost = aij[src_i, j]
            job = j

    if job == -1:
        return perm.copy()

    dst_i = 0
    max_slack = slack[0]
    for i in range(1, I):
        if i == src_i:
            continue
        if slack[i] > max_slack:
            max_slack = slack[i]
            dst_i = i

    result = perm.copy()
    result[job] = dst_i
    return result


# Every operator below takes (permutation, model) and returns a new permutation, even
# though most ignore `model` -- a uniform signature so MUTATION_OPERATORS can dispatch
# to any of them without special-casing.


def mutation_swap(permutation: np.ndarray, model: Model) -> np.ndarray:
    return mutation_swap_nb(permutation)


def mutation_reversion(permutation: np.ndarray, model: Model) -> np.ndarray:
    return mutation_reversion_nb(permutation)


def mutation_insertion(permutation: np.ndarray, model: Model) -> np.ndarray:
    return mutation_insertion_nb(permutation)


def mutation_big_swap(permutation: np.ndarray, model: Model) -> np.ndarray:
    return mutation_big_swap_nb(permutation)


def mutation_random(permutation: np.ndarray, model: Model) -> np.ndarray:
    return mutation_random_nb(permutation, int(model.I))


def mutation_scramble(permutation: np.ndarray, model: Model) -> np.ndarray:
    return mutation_scramble_nb(permutation)


def mutation_cyclic_shift(permutation: np.ndarray, model: Model) -> np.ndarray:
    return mutation_cyclic_shift_nb(permutation)


def mutation_greedy_reassign(permutation: np.ndarray, model: Model) -> np.ndarray:
    return mutation_greedy_reassign_nb(permutation, model.aij, model.cij, model.bi)


def mutation_migration(permutation: np.ndarray, model: Model) -> np.ndarray:
    return mutation_migration_nb(permutation, model.aij, model.bi)


# Adding a new operator is just adding it here -- choose_mutation picks uniformly among
# whatever is listed, with no separate count (and no np.random.randint(N) literal) to
# remember to update. (Previously this dispatch used np.random.randint(4) with 5 possible
# branches, silently making mutation_big_swap unreachable.)
MUTATION_OPERATORS = (
    mutation_swap,
    mutation_reversion,
    mutation_insertion,
    mutation_random,
    mutation_big_swap,
    mutation_scramble,
    mutation_cyclic_shift,
    mutation_greedy_reassign,
    mutation_migration,
)


def choose_mutation(permutation: np.ndarray, model: Model) -> np.ndarray:
    op = MUTATION_OPERATORS[np.random.randint(len(MUTATION_OPERATORS))]
    return op(permutation, model)
