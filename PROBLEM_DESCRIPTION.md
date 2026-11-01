# Generalized Quadratic Assignment Problem (GQAP)

## Problem Overview

This project solves the Generalized Quadratic Assignment Problem (GQAP) with a genetic
algorithm. The objective is to assign $J$ jobs (tasks/customers) to $I$ facilities
(agents) so as to minimize total cost, subject to per-facility capacity constraints,
while accounting for the spatial interaction between facilities and between jobs.

The problem combines:
- An **assignment** decision (which facility serves which job)
- **Capacity** constraints (knapsack-like, one per facility)
- A **quadratic interaction cost** coupling facility distances with job-to-job
  interaction flows

There is no task ordering/sequencing in this formulation — it is not a routing or
TSP-like problem. Each job is assigned to exactly one facility; the cost depends on
that assignment and on how "close" interacting jobs end up via their assigned
facilities.

---

## Sets and Indices

- $I$ — number of facilities
- $J$ — number of jobs

---

## Model Parameters

### Assignment Cost ($c_{ij}$)

$c_{ij} \in \mathbb{R}^{I \times J}$ — the cost of assigning job $j$ to facility $i$.

### Resource Consumption ($a_{ij}$)

$a_{ij} \in \mathbb{R}^{I \times J}$ — the resource usage if facility $i$ serves job $j$.

### Facility Capacity ($b_i$)

$b_i \in \mathbb{R}^I$ — the capacity of facility $i$.

Capacity constraint:

$$
\sum_{j=1}^{J} a_{ij} x_{ij} \leq b_i \quad \forall i
$$

### Distance Between Facilities (DIS)

$DIS \in \mathbb{R}^{I \times I}$ — $DIS_{ik}$ is the Euclidean distance between
facilities $i$ and $k$, computed from their 2D coordinates.

### Interaction Between Jobs (F)

$F \in \mathbb{R}^{J \times J}$ — $F_{jl}$ is the Euclidean distance/flow between jobs
$j$ and $l$, computed from their 2D coordinates.

---

## Decision Variables

### Assignment Variable ($x_{ij}$)

$$
x_{ij} =
\begin{cases}
1 & \text{if job } j \text{ is assigned to facility } i \\
0 & \text{otherwise}
\end{cases}
$$

Each job is assigned to exactly one facility: $\sum_i x_{ij} = 1 \ \forall j$.

---

## Objective

Minimize total cost — assignment cost plus the quadratic interaction cost between
facility distances and job interaction flows:

$$
\min \sum_{i=1}^{I} \sum_{j=1}^{J} c_{ij} x_{ij}
\;+\;
\sum_{j=1}^{J} \sum_{l=1}^{J} DIS_{\,\text{perm}(j),\,\text{perm}(l)} \cdot F_{jl}
$$

where $\text{perm}(j)$ is the facility assigned to job $j$. The second term is what
makes this a *quadratic* assignment problem: the cost of the interaction between two
jobs depends on the distance between the facilities they are each assigned to.

A reference exact MILP formulation (with the standard QAP linearization) for one of
the benchmark instances is provided in `Exact/c201535/R2_c201535.gms`.

---

## Genetic Algorithm Representation

Each candidate solution (`Individual`, see `src/data/models.py`) contains:

- `permutation` — array of length $J$; `permutation[j]` is the **facility index**
  assigned to job $j$ (this is an assignment, not a task ordering)
- `xij` — the corresponding $(I \times J)$ binary assignment matrix, derived from
  `permutation`
- `cost` — the objective value (`inf` if the assignment violates any facility's
  capacity)
- `cvar` — remaining capacity slack per facility, $b_i - \sum_j a_{ij} x_{ij}$

Infeasible permutations (i.e. capacity-violating) produced by crossover/mutation are
fixed by the repair operator (see `src/operators/repair.py`) before being evaluated.

---

## Data Description

Facilities and jobs are placed in 2D space via coordinates `(X, Y)` for facilities and
`(XX, YY)` for jobs. Both `DIS` and `F` are computed as pairwise Euclidean distance:

$$
d = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}
$$

Dataset files (`datasets/*.m`, `debug_datasets/*.m`) follow a MATLAB-style syntax
parsed by `src/data/model_loader.py`, defining `I`, `J`, `cij`, `aij`, `bi`, `X`, `Y`,
`XX`, `YY`.

---

## Summary

> Assign jobs to facilities so as to minimize assignment cost plus the quadratic cost
> of interaction between jobs, mediated by the distance between their assigned
> facilities — subject to per-facility capacity constraints.

It combines:
- An assignment problem
- Knapsack-style capacity constraints
- A quadratic, distance-based interaction cost (the "Q" in GQAP)
