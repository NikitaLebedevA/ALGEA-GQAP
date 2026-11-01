import numpy as np

from src.costs import evaluate_permutation
from src.data.models import Individual, Model


def heuristic2(model: Model) -> Individual:
    """Initial solution + repair, matching upstream Thesis Heuristic2.m.

    CT(i,j) = cij(i,j) + sum(DIS(i,:)) + sum(F(j,:)); greedy assign with capacity
    masking, then cascading repair while capacity slack is negative.
    """
    I, J = model.I, model.J
    X = np.zeros((I, J), dtype=int)
    count = np.zeros(I, dtype=float)

    CT = np.zeros((I, J), dtype=float)
    for i in range(I):
        for j in range(J):
            CT[i, j] = model.cij[i, j] + model.DIS[i].sum() + model.F[j].sum()

    for j in range(J):
        b = int(np.argmin(CT[:, j]))
        if count[b] <= model.bi[b]:
            X[b, j] = 1
            count[b] += model.aij[b, j]
        else:
            ct_max = float(np.max(CT))
            CT[b, j] = ct_max
            b = int(np.argmin(CT[:, j]))
            if count[b] <= model.bi[b]:
                X[b, j] = 1
                count[b] += model.aij[b, j]
            else:
                CT[b, j] = ct_max
                b = int(np.argmin(CT[:, j]))
                X[b, j] = 1
                count[b] += model.aij[b, j]

    cvar = model.bi - count
    Wij = X * model.aij
    max_repair_passes = I * J
    repair_pass = 0
    while np.any(cvar < -1e-9) and repair_pass < max_repair_passes:
        repair_pass += 1
        for i in range(I):
            while cvar[i] < -1e-9:
                assigned_jobs = np.where(X[i] == 1)[0]
                if assigned_jobs.size == 0:
                    break
                b = assigned_jobs[np.argmax(Wij[i, assigned_jobs])]
                count[i] -= model.aij[i, b]
                cvar[i] = model.bi[i] - count[i]
                X[i, b] = 0
                Wij[i, b] = 0
                d = int(np.argmin(model.aij[:, b]))
                if d == i:
                    d = int(np.argmax(cvar))
                count[d] += model.aij[d, b]
                cvar[d] = model.bi[d] - count[d]
                X[d, b] = 1
                Wij[d, b] = model.aij[d, b]

    permutation = np.argmax(X, axis=0)
    return evaluate_permutation(permutation, model)
