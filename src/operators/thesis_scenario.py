"""Thesis-style scenario operators (analyze_perm, mask_mutation, combine_q)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from src.costs import evaluate_permutation
from src.data.models import Individual, Model


def combine_q(position1: np.ndarray, position2: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    pattern_bool = pattern.astype(bool)
    return np.where(pattern_bool, position1, position2)


def mask_mutation_swap(permutation: np.ndarray, mask: np.ndarray) -> np.ndarray:
    indices = np.where(~mask)[0]
    if indices.size <= 1:
        return permutation.copy()
    point = int(np.random.randint(0, indices.size - 1))
    result = permutation.copy()
    i = indices[point]
    j = indices[point + 1]
    result[[i, j]] = result[[j, i]]
    return result


def mask_mutation_big_swap(permutation: np.ndarray, mask: np.ndarray) -> np.ndarray:
    indices = np.where(~mask)[0]
    if indices.size <= 1:
        return permutation.copy()
    i, j = np.random.choice(indices, size=2, replace=False)
    result = permutation.copy()
    result[[i, j]] = result[[j, i]]
    return result


def mask_mutation_inversion(permutation: np.ndarray, mask: np.ndarray) -> np.ndarray:
    indices = np.where(~mask)[0]
    if indices.size <= 1:
        return permutation.copy()
    i, j = np.sort(np.random.choice(indices, size=2, replace=False))
    result = permutation.copy()
    result[i : j + 1] = result[i : j + 1][::-1]
    return result


def mask_mutation_displacement(permutation: np.ndarray, mask: np.ndarray) -> np.ndarray:
    indices = np.where(~mask)[0]
    if indices.size <= 2:
        return permutation.copy()
    subset = permutation[indices]
    choices = np.sort(np.random.choice(np.arange(1, subset.size), size=2, replace=False))
    i, j = choices
    temp = subset[i : j + 1]
    q1 = subset[:i]
    q2 = subset[j + 1 :]
    new_subset = np.concatenate((temp, q1, q2))
    result = permutation.copy()
    result[indices] = new_subset
    return result


def mask_mutation_perturbation(permutation: np.ndarray, mask: np.ndarray, model: Model) -> np.ndarray:
    indices = np.where(~mask)[0]
    result = permutation.copy()
    if indices.size == 0:
        return result
    idx = np.random.choice(indices)
    result[idx] = (result[idx] + 1) % model.I
    return result


def mask_mutation(index: int, permutation: np.ndarray, mask: np.ndarray, model: Model) -> np.ndarray:
    if index == 1:
        return mask_mutation_swap(permutation, mask)
    if index == 2:
        return mask_mutation_big_swap(permutation, mask)
    if index == 3:
        return mask_mutation_inversion(permutation, mask)
    if index == 4:
        return mask_mutation_displacement(permutation, mask)
    return mask_mutation_perturbation(permutation, mask, model)


def analyze_perm(
    population: Sequence[Individual],
    *,
    p_fixed_x: float,
    model: Model,
) -> tuple[np.ndarray, np.ndarray, Individual, np.ndarray]:
    n_pop = len(population)
    n_genes = population[0].permutation.size
    n_fixed = int(np.floor(p_fixed_x * n_pop))

    mask = np.zeros((n_pop, n_genes), dtype=bool)
    perms = np.stack([ind.permutation for ind in population])

    left = perms[:, :-1]
    right = perms[:, 1:]
    pair_match = (left[:, None, :] == left[None, :, :]) & (right[:, None, :] == right[None, :, :])
    pair_count = pair_match.sum(axis=1) - 1
    eligible = pair_count >= n_fixed

    for row in range(n_pop):
        col = 0
        while col < n_genes - 1:
            if eligible[row, col]:
                mask[row, col : col + 2] = True
                col += 2
            else:
                col += 1

    dominant_idx = 0
    dominant_score = -1
    for idx in range(n_pop):
        score = int(mask[idx].sum())
        if score > dominant_score:
            dominant_idx = idx
            dominant_score = score
        elif score == dominant_score and np.random.random() > 0.5:
            dominant_idx = idx
            dominant_score = score

    dominant_individual = evaluate_permutation(population[dominant_idx].permutation, model)
    return (
        dominant_individual.permutation,
        mask,
        dominant_individual,
        mask[dominant_idx],
    )
