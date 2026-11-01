# Algorithms

This project solves the Generalized Quadratic Assignment Problem (GQAP, see
[PROBLEM_DESCRIPTION.md](PROBLEM_DESCRIPTION.md)) with **fourteen metaheuristic families**,
each implemented as a **standard** (Holland-style) and **improved** (GEA-style) pair —
**28 Hydra configs** in total. All variants share the same solution representation,
problem instance format, and root run loop (`AlgorithmBase`). This document describes
the shared scaffolding, the standard/improved architecture, each algorithm family, and
the tuning and experimental procedures.

---

## Architecture

### Folder layout (`src/algos/`)

| Path | Contents |
|---|---|
| `core/base.py` | `AlgorithmBase` — shared init, run loop, repair wrapper |
| `core/standard_base.py` | `StandardBase` — thesis scaffold: heuristic2 init, pool selection |
| `core/improved_base.py` | `ImprovedBase` — diversity selection, pool replacement, RC/DM/GI helpers, memetic polish |
| `core/logger.py` | `GALogger` — iteration timing, NFE, operator stats |
| `mixins/adaptive.py` | `AdaptiveRatesMixin` — reward-memory application rates |
| `mixins/annealing.py` | `AnnealingMixin` — Metropolis acceptance + cooling |
| `mixins/pso.py` | `PSOMixin` — particle tracking, velocity/discrete PSO moves |
| `ga/algorithm.py` | `StandardGA`, `ImprovedGA` |
| `gea/algorithm.py` | `StandardGEA`, `ImprovedGEA` |
| `gea/scenario.py` | `StandardGEAScenario{1,2,3}`, `ImprovedGEAScenario{1,2,3}` |
| `adaptive/ga.py` | `StandardAdaptiveGA`, `ImprovedAdaptiveGA` |
| `adaptive/gea.py` | `StandardAdaptiveGEA`, `ImprovedAdaptiveGEA` |
| `adaptive/gea_scenario.py` | `StandardAdaptiveGEAScenario{1,2,3}`, `ImprovedAdaptiveGEAScenario{1,2,3}` |
| `sa/algorithm.py` | `StandardSA`, `ImprovedSA` |
| `pso/algorithm.py` | `StandardParticleSwarm`, `ImprovedParticleSwarm` |
| `hybrid/ga_sa.py` | `StandardHybridGASA`, `ImprovedHybridGASA` |
| `hybrid/ga_pso.py` | `StandardHybridGAPSO`, `ImprovedHybridGAPSO` |

### Standard vs improved scaffolding

| Feature | Standard family (`StandardBase`) | Improved family (`ImprovedBase`) |
|---|---|---|
| Parent selection | Exponential roulette `exp(-β·cost/worst)` | Diversity-weighted roulette (`DiversitySelector`) |
| Init | `heuristic2` + mutation fill | Random permutations + `RFRepair` |
| Survivor selection | (μ+λ) pool merge, top N | Elite + diversity/cost pool selection |
| Repair on offspring | None (`IdentityRepair`) | `RFRepair` |
| Stagnation immigrants | No | Yes (`stagnation_limit`, `immigrant_rate`) |
| Memetic local search | No | Yes (top-3 elites, disabled when `J > 50`) |
| Operator batching | Fixed counts per generation | Fixed-count pools per generation |
| Scenario ops (standard GEA) | Thesis `analyze_perm` / mask / combine | RC / DM / GI operator batches |

Standard configs use tuned operator rates copied from their improved counterpart.
Improved configs are Optuna-tuned and may include
`components@ga` (repair/selector overrides) in Hydra defaults.

### Config catalog

| Improved config | Standard config | Improved class | Standard class |
|---|---|---|---|
| `ga` | `standard` | `ImprovedGA` | `StandardGA` |
| `gea` | `standard_gea` | `ImprovedGEA` | `StandardGEA` |
| `adaptive` | `standard_adaptive` | `ImprovedAdaptiveGA` | `StandardAdaptiveGA` |
| `sa` | `standard_sa` | `ImprovedSA` | `StandardSA` |
| `pso` | `standard_pso` | `ImprovedParticleSwarm` | `StandardParticleSwarm` |
| `hybrid_ga_sa` | `standard_hybrid_ga_sa` | `ImprovedHybridGASA` | `StandardHybridGASA` |
| `hybrid_ga_pso` | `standard_hybrid_ga_pso` | `ImprovedHybridGAPSO` | `StandardHybridGAPSO` |
| `gea_scenario_1` | `standard_gea_scenario_1` | `ImprovedGEAScenario1` | `StandardGEAScenario1` |
| `gea_scenario_2` | `standard_gea_scenario_2` | `ImprovedGEAScenario2` | `StandardGEAScenario2` |
| `gea_scenario_3` | `standard_gea_scenario_3` | `ImprovedGEAScenario3` | `StandardGEAScenario3` |
| `adaptive_gea` | `standard_adaptive_gea` | `ImprovedAdaptiveGEA` | `StandardAdaptiveGEA` |
| `adaptive_gea_scenario_1` | `standard_adaptive_gea_scenario_1` | `ImprovedAdaptiveGEAScenario1` | `StandardAdaptiveGEAScenario1` |
| `adaptive_gea_scenario_2` | `standard_adaptive_gea_scenario_2` | `ImprovedAdaptiveGEAScenario2` | `StandardAdaptiveGEAScenario2` |
| `adaptive_gea_scenario_3` | `standard_adaptive_gea_scenario_3` | `ImprovedAdaptiveGEAScenario3` | `StandardAdaptiveGEAScenario3` |

Results JSON stems follow `scripts/utils/labels.py` (`StandardGA` → `standard`,
`ImprovedGA` in `ga` module → `ga`, `ImprovedGEA` → `improvedgea`, etc.).

---

## Shared framework (`src/algos/core/base.py`)

### Representation

A candidate solution (`Individual`, in `src/data/models.py`) is an integer array
`permutation` of length `J` (number of jobs), where `permutation[j]` is the index of
the facility assigned to job `j`. Alongside it, every `Individual` carries:

- `cost`: the GQAP objective value (`inf` if the assignment violates any facility's
  capacity)
- `cvar`: per-facility remaining capacity slack (`b_i - load_i`), used by some
  operators as a feasibility-robustness signal

A problem instance (`Model`) holds the assignment-cost matrix `cij`, resource-usage
matrix `aij`, facility capacities `bi`, facility-distance matrix `DIS`, and
job-interaction matrix `F`.

### Cost evaluation (`src/costs.py`)

`cost_function_perm` computes a permutation's cost from scratch (assignment cost +
quadratic interaction cost), returning `inf` if any facility is over capacity.
`cost_function_perm_delta` computes the same cost incrementally from a known-feasible
baseline cost, touching only the positions that changed — O(K·J) instead of O(J²)
when K changed positions is small. Every algorithm routes offspring through the
batched version (`evaluate_permutation_delta_batch`), each child paired with whichever
parent it is "closest to" so the delta stays cheap.

### Operators

All GA-based algorithms in this project share the same operator sets, dispatched
uniformly at random by `choose_crossover` / `choose_mutation`. Operator consistency
across algorithms is a design requirement: each algorithm is only distinguished by
*how* it applies and combines these operators, not *which* operators it has access to.

**Crossover** (`src/operators/crossover.py`) — four operators:

| Operator | Mechanism |
|---|---|
| `crossover_one_point` | Splits both parents at one random point; each child is one parent's prefix + the other's suffix. |
| `crossover_two_point` | Splices a random middle segment from one parent into the other's copy, and vice versa. |
| `crossover_uniform` | Per-gene coin flip decides which parent each child inherits from at each position. |
| `crossover_greedy` | Per-gene, child 1 takes whichever parent's assignment is cheaper (`cij`); child 2 takes the complementary gene, spanning the same diversity as a random crossover. |

**Mutation** (`src/operators/mutations.py`) — nine operators, including the five
primary discrete moves (swap, big swap, insertion, reversion, random) plus four
additional operators:

| Operator | Category | Mechanism |
|---|---|---|
| `mutation_swap` | Primary | Swaps two adjacent genes. |
| `mutation_big_swap` | Primary | Swaps two genes at arbitrary (non-adjacent) positions. |
| `mutation_insertion` | Primary | Moves a random segment to a different position. |
| `mutation_reversion` | Primary | Reverses a random contiguous segment. |
| `mutation_random` | Primary | Reassigns 1–5 random genes to brand-new random facilities (fresh genetic material). |
| `mutation_scramble` | Extended | Shuffles the values within a random window of 3–5 genes. |
| `mutation_cyclic_shift` | Extended | Cyclically rotates 3 random genes. |
| `mutation_greedy_reassign` | Directed | Finds the single worst-assigned job and reassigns it to the cheapest feasible facility. |
| `mutation_migration` | Directed | Moves the priciest job out of the most-loaded facility into the facility with the most spare capacity. |

Mutation is applied **per-individual as a discrete operator**: one randomly chosen
operator is applied to the whole individual with probability `mutation_rate`.
Iterating over every gene with a small per-gene probability is not used — it raises
complexity unnecessarily and conflates operator application with gene-level probability.

### Repair (`src/repair.py`)

Every crossover and mutation can produce an infeasible permutation (a facility over
capacity). Repair iteratively evicts an overloaded facility's job and reassigns it
until feasible:

- **`GreedyRepair`**: deterministic — always picks the worst capacity violation and
  reassigns to the cheapest feasible facility. Default for the **standard family**.
- **`RFRepair`**: picks from a random subsample of candidates instead of the single
  greedy best, trading exactness for population diversity. Default for the **improved
  family**.

### Selection (`src/selection.py`, `DiversitySelector`)

Used by the **improved family** only; the standard family uses fitness-proportionate
selection implemented in `StandardBase`.

- **Parent selection**: roulette wheel weighted by each individual's mean Hamming
  distance to the rest of the population — more distinctive individuals are more
  likely to be picked as parents.
- **Survivor selection**: the cheapest `elite_fraction` of the merged pool (parents +
  offspring + mutants + immigrants, deduplicated) survives as elites; the rest is
  ranked by `diversity_weight · diversity_score + cost_weight · cost_score`, with
  `diversity_weight` decaying linearly over the run so later generations converge.

### Local search (memetic polish, improved family only)

Each generation, the top 3 individuals receive a per-job hill-climbing pass (try every
facility for each job, keep whichever lowers cost). Auto-disabled above `J = 50` jobs
where the O(J·I) cost per pass dominates runtime. `ImprovedSA` also polishes its single
solution when `J ≤ 50`.

### Run loop

`run(time_limit)` initializes the population, then repeatedly calls the subclass's
`step()`, re-sorts by cost, polishes elites, and stops once `time_limit` (wall-clock
seconds) or `iterations` is reached.

---

## 1. Genetic Algorithm — `StandardGA` / `ImprovedGA` (`src/algos/ga/algorithm.py`)

> Holland, J. H. (1992). Genetic algorithms. *Scientific American*, 267(1), 66–73.

### `StandardGA` (config: `standard`)

The textbook simple genetic algorithm on the thesis scaffold:

- **Init**: `heuristic2` seed, population filled by mutating the seed (upstream style).
- **Selection**: exponential roulette on cost (`exp(-β·cost/worst_cost)`).
- **Crossover / mutation**: fixed batch counts `ncrossover`, `nmutation` per generation;
  one crossover/mutation operator chosen uniformly at random from the project operator set.
- **Replacement**: (μ+λ) pool — parents + offspring merged, best N kept.
- No offspring repair, no diversity selection, no stagnation immigrants, no memetic polish.

This is the literal baseline the improved variants are compared against.

> **Design notes:** Uses all four crossover operators and all nine mutation operators.
> Mutation is applied as a discrete per-individual operator, not as a per-gene Bernoulli trial.

### `ImprovedGA` (config: `ga`)

Same crossover + mutation operators as `StandardGA`, but on the **improved scaffold**
(`DiversitySelector`, `RFRepair`, stagnation immigrants, memetic local search). No RC/DM/GI
scenario operators — isolates the contribution of the GEA infrastructure from the three
scenario-specific stages in `ImprovedGEA`.

Hydra config: `scripts/conf/ga.yaml` → results file `ga.json`.

---

## 2. Full GEA — `StandardGEA` / `ImprovedGEA` (`src/algos/gea/algorithm.py`)

### `ImprovedGEA` (config: `gea`)

This project's own enhanced GA. Where `StandardGA` deliberately bypasses most of the
improved framework, `ImprovedGEA` uses every improved-family component and runs **five
operator stages per generation** in fixed sequence:

| Stage | Operator | Rate | Description |
|---|---|---|---|
| 1 | **Crossover** | `crossover_rate` | `choose_crossover` — one of the 4 standard operators, per pair. |
| 2 | **Mutation** | `mutation_rate` | `choose_mutation` — one of the 9 standard operators, per individual. |
| 3 | **RC crossover** | `rc_rate` | Robust Chromosome: per gene, inherit from whichever parent has more remaining capacity slack (`cvar`). |
| 4 | **DM** | `dm_rate` | Directed Mutation: move each individual's single worst-assigned job to its cheapest feasible facility. |
| 5 | **GI** | `injection_rate` | Gene Injection: replace a handful of random genes with fresh random facility assignments. |

All five offspring pools are merged with the current population; `DiversitySelector`
picks the next generation. Additional enhancements:

- **Repair**: `RFRepair` by default.
- **Stagnation immigrants**: if best solution hasn't improved for `stagnation_limit`
  iterations, `immigrant_rate · population_size` fresh random individuals are injected.
- **Memetic local search**: enabled.

### `StandardGEA` (config: `standard_gea`)

Same five operator stages and rates as `ImprovedGEA`, but on the **Holland scaffold**:
exponential parent selection, fixed-batch operators, pool survivor selection, and thesis
scenario batches where enabled. No offspring repair, immigrants, or memetic polish.

> **Design notes:** `ImprovedGA` is the improved-family baseline with only stages 1–2 (no RC/DM/GI).
> `ImprovedGEAScenario{1,2,3}` are **single-enhancement ablations** of the full `ImprovedGEA` —
> each adds only one of stages 3, 4, or 5 on top of standard crossover + mutation.
> The RC crossover helper (`crossover_robust_chromosome`) lives in `src/operators/crossover.py`;
> RC/DM/GI driver methods live on `ImprovedBase` (`_robust_chromosome_crossover`,
> `_directed_mutation`, `_gene_injection`) so all improved subclasses inherit them.

---

## 3. Improved GA without scenarios — see §1 `ImprovedGA`

*(Documented above alongside `StandardGA`.)*

---

## 4. Adaptive GA — `StandardAdaptiveGA` / `ImprovedAdaptiveGA` (`src/algos/adaptive/ga.py`)

> **Scope note:** This algorithm is included in tuning and final experiment runs
> alongside the rest of the algorithms, but its results **may not be reported** in this
> paper. It will be formally proposed in a follow-up project with a new cost function.

Same improved-family infrastructure as `ImprovedGA` (diversity selection, `RFRepair`,
stagnation immigrants, memetic local search). Crossover and mutation counts adapt every
generation (see **Adaptive rate mechanism** below). Scenario operators (RC/DM/GI) are
not used.

### `StandardAdaptiveGA` (config: `standard_adaptive`)

Same lambda-adaptive crossover/mutation on the thesis scaffold with pool survivor selection.

### Adaptive rate mechanism

Each active operator has an independent application rate `lambda`, initialized at
`(lambda_min + lambda_max) / 2`. Lambdas do not sum to one. Offspring counts use
`int(population_size * lambda)` (crossover producers adjust to pairs), with no base-rate multiplier.
Shared implementation: `src/algos/mixins/adaptive.py`.

1. For minimization, `delta = (reference_cost - child_cost) / (abs(reference_cost) + epsilon)`.
   Crossover and RC use the best parent. The baseline used for efficient delta-cost
   evaluation remains separate and unchanged.
2. Sum positive and negative magnitudes separately into `R` and `Q`;
   `r = R / (R + Q + epsilon)`, `q = Q / (R + Q + epsilon)`.
3. Take the strongest positive and negative magnitudes `d_plus`, `d_minus`;
   transform each with `x / (1 + x)` into `b_plus`, `b_minus`.
4. Combined reward is `max(r, b_plus)`; penalty is `max(q, b_minus)`.
5. Any positive reward replaces the stored reward and resets its age to zero.
   Otherwise age increases once per generation, including empty batches.
6. Effective reward is `stored_reward * max(0, 1 - age / reward_horizon)`.
   Default horizon 10 applies factors 1.0, 0.9, ..., 0.1, then zero.
7. Add `alpha * effective_reward` while reward is active, ignoring punishment.
   Otherwise subtract `alpha * penalty`. Clip to the operator's bounds.

`lambda_min` and `lambda_max` accept a scalar or mappings keyed by active operator
names (`crossover`, `mutation`, `rc`, `dm`, `gi`). Bounds must satisfy
`0 <= min <= max <= 1`. Defaults 0.1 and 0.9 are provisional, not tuned.
A zero configured base rate disables the corresponding operator. Nonzero base rates
are otherwise used only for fixed-operator ablations; adaptive rates start at the midpoint.

The paper does not define non-finite objectives. Implementation policy: an invalid
reference is excluded and counted; an infeasible child with a finite reference gives
limiting deterioration (`q = b_minus = 1`, `r = 0`). Finite improvements in the same
batch can still supply an extreme reward. Infeasible offspring are excluded from
survivor selection. This policy is logged, rather than silently dropping failures.

`lambda_history` records post-generation rates. `adaptation_history` records per-operator
counts, aggregate/extreme signals, reward memory, age, decay, and actual clipped changes.
Rates need not oscillate: repeated rewards can hold an operator at its upper bound.

### Operator-adaptation ablation

`fixed_operators` keeps the listed operators at their configured base application
rates, without adaptation. Other operators follow the new mechanism. Fixed rates
need not equal the midpoint or lie within the bounds of adaptive operators.
Ready-to-run variants in `scripts/conf/ablation/{improved,standard}` inherit their
baseline configuration and write separate result files. Previous multiplier-based
results are not comparable without rerunning; there is no legacy mode.

| Algorithm | Adaptive operators |
|---|---|
| `ImprovedAdaptiveGA` / `StandardAdaptiveGA` | crossover, mutation |
| `ImprovedAdaptiveGEA` / `StandardAdaptiveGEA` | crossover, mutation, RC, DM, GI |
| `ImprovedAdaptiveGEAScenario1` / `StandardAdaptiveGEAScenario1` | crossover, mutation, RC |
| `ImprovedAdaptiveGEAScenario2` / `StandardAdaptiveGEAScenario2` | crossover, mutation, DM |
| `ImprovedAdaptiveGEAScenario3` / `StandardAdaptiveGEAScenario3` | crossover, mutation, GI |

Stagnation immigrants remain at fixed `immigrant_rate`.

---

## 5. Simulated Annealing — `StandardSA` / `ImprovedSA` (`src/algos/sa/algorithm.py`)

### `ImprovedSA` (config: `sa`)

Classic **single-solution** SA on `AlgorithmBase` + `AnnealingMixin`. Population size
is forced to `1`. Uses `RFRepair` by default and optional memetic polish when `J ≤ 50`.
Each iteration, one neighbor is proposed via `choose_mutation` and:

- accepted outright if better (`cost ≤ current`), or
- accepted with Metropolis probability `exp(-Δ/T)` if worse.

Temperature is cooled geometrically after each step:
`T ← max(T_min, T · cooling_rate)`. There is no crossover, no population selection,
no immigrants.

### `StandardSA` (config: `standard_sa`)

Same Metropolis annealing loop on the thesis scaffold (`heuristic2` init, no offspring repair).

> **Design notes:** A population-based multi-walker SA would share a temperature
schedule across independent chains with no selection pressure between them — this
does not correspond to the SA algorithm described in the literature. The
single-solution version is used throughout this project.

---

## 6. Particle Swarm — `StandardParticleSwarm` / `ImprovedParticleSwarm` (`src/algos/pso/algorithm.py`)

> Kennedy, J., & Eberhart, R. (1995, November). Particle swarm optimization.
> *Proceedings of ICNN'95* (Vol. 4, pp. 1942–1948).

Discrete PSO adapted for integer-encoded assignment problems. The original PSO operates
over a continuous real-valued space; here each gene `x[j]` is an integer facility
index. The standard PSO velocity equation is preserved exactly, but applied over the
integer domain via rounding and clamping:

**Velocity update** (real-valued, one entry per job):
```
v[j] ← w · v[j]
      + c1 · r1 · (pbest[j] − x[j])    # cognitive: pull toward personal best
      + c2 · r2 · (gbest[j] − x[j])    # social:    pull toward global best
```

**Position update** (integer, clamped to valid facility range):
```
x_new[j] = clip( round( x[j] + v[j] ), 0, I−1 )  →  repair  →  evaluate
```

Velocity is clamped to `[−I, I]` (I = number of facilities) to bound the maximum
step size. This extension preserves the original PSO velocity semantics — cognitive
and social pulls are proportional to the integer distance between positions —
while mapping updates back into the feasible integer domain via rounding and repair.

### `ImprovedParticleSwarm` (config: `pso`)

Uses `ImprovedBase` + `PSOMixin`: `RFRepair`, stagnation immigrants, memetic polish.

### `StandardParticleSwarm` (config: `standard_pso`)

Same velocity PSO on `StandardBase` + `PSOMixin` with `heuristic2` init and no offspring repair.

Particle identity (`self.particles` / `self.personal_best` / `self.velocities`) is
maintained separately from `self.population`'s ordering, since `AlgorithmBase.run()` re-sorts
`self.population` by cost after every step, which would corrupt positional identity.

> **Design notes:** PSO is a continuous-space algorithm by origin. Applying it directly
> to integer-encoded GQAP requires explicit justification: the velocity equation
> produces a real-valued displacement, which is discretized to the nearest integer
> facility index and then repaired for capacity feasibility. This is the standard
> extension used in the discrete/integer PSO literature for non-permutation integer
> problems.

---

## 7. Hybrid GA+PSO — `StandardHybridGAPSO` / `ImprovedHybridGAPSO` (`src/algos/hybrid/ga_pso.py`)

> Juang, C. F. (2004). A hybrid of genetic algorithm and particle swarm optimization
> for recurrent network design. *IEEE Transactions on Systems, Man, and Cybernetics,
> Part B*, 34(2), 997–1006.

Each generation, the population is ranked by cost and split in two:

- the better-ranked `pso_fraction` undergo a PSO-style discrete move (exploitation,
  pulled toward personal best or global best via `choose_crossover`, or diversified via
  `choose_mutation` for the inertia term)
- the rest are regenerated by ordinary GA crossover + mutation via `choose_crossover` /
  `choose_mutation`, selected via `select_from_pool` (exploration)

An elitist replacement step guarantees the swarm's best-ever solution survives into
the next generation, as in the original paper.

> **Design notes:** The GA component uses the same `choose_crossover` / `choose_mutation`
> operator sets as all other algorithms. `ImprovedHybridGAPSO` uses pool selection on the
> GA fraction; `StandardHybridGAPSO` uses fitness-based generational selection on that fraction.

---

## 8. Hybrid GA+SA — `StandardHybridGASA` / `ImprovedHybridGASA` (`src/algos/hybrid/ga_sa.py`)

> Chen, P. H., & Shahandashti, S. M. (2009). Hybrid of genetic algorithm and simulated
> annealing for multiple project scheduling with multiple resource constraints.
> *Automation in Construction*, 18(4), 434–443.

GA crossover + mutation (via `choose_crossover` / `choose_mutation`, same operator sets
as all other algorithms) supply every candidate move; what changes is the **acceptance
rule**. Each child must first pass a Metropolis test against its own parent baseline
before it is eligible to join the survivor pool:

- a child better than its baseline is always accepted,
- a worse child is accepted with probability `exp(-Δ/T)`.

`T` anneals down geometrically over the run, so early generations tolerate
quality-losing moves (escaping local optima) while late generations behave like a
plain elitist GA.

> **Design notes:** This hybrid retains a population-based structure (unlike standalone
> SA, §5). `ImprovedHybridGASA` filters offspring through Metropolis acceptance then
> pool selection; `StandardHybridGASA` uses the same Metropolis filter on the thesis scaffold.

---

## 9. GEA Scenario 1 — RC crossover (`src/algos/gea/scenario.py`)

`ImprovedGEAScenario1` (config: `gea_scenario_1`) / `StandardGEAScenario1` (config: `standard_gea_scenario_1`)

Modernization of `ImprovedGEA` — Scenario 1: **RC (Robust Chromosome) crossover**.

Runs three operators each generation at **fixed rates** — no adaptive lambda:

1. **Regular crossover** (`choose_crossover`, all 4 operators) at `crossover_rate`.
2. **Regular mutation** (`choose_mutation`, all 9 operators) at `mutation_rate`.
3. **RC crossover** (`rc_rate`): per gene, the child inherits from whichever parent's
   assigned facility has *more remaining capacity slack* (`cvar`) — the more
   capacity-robust gene, independent of its assignment cost.

All three sets of offspring are pooled with the current population; `DiversitySelector`
picks the next generation.

> **Design notes:** The fixed-rate version is reported in this paper. The adaptive
> counterpart (`ImprovedAdaptiveGEAScenario1`, §13) is tuned and run alongside the rest.

---

## 10. GEA Scenario 2 — Directed Mutation (`src/algos/gea/scenario.py`)

`ImprovedGEAScenario2` / `StandardGEAScenario2`

Modernization of `ImprovedGEA` — Scenario 2: **DM (Directed Mutation)**.

Runs three operators each generation at **fixed rates** — no adaptive lambda:

1. **Regular crossover** (`choose_crossover`, all 4 operators) at `crossover_rate`.
2. **Regular mutation** (`choose_mutation`, all 9 operators) at `mutation_rate`.
3. **DM** (`mutation_greedy_reassign`, `dm_rate`): moves each selected individual's
   single worst-assigned job to its cheapest feasible facility — a directed,
   fitness-improving move rather than a blind random one.

> **Design notes:** Fixed-rate version reported here; adaptive counterpart is
> `ImprovedAdaptiveGEAScenario2` (§14).

---

## 11. GEA Scenario 3 — Gene Injection (`src/algos/gea/scenario.py`)

`ImprovedGEAScenario3` / `StandardGEAScenario3`

Modernization of `ImprovedGEA` — Scenario 3: **GI (Gene Injection)**.

Runs three operators each generation at **fixed rates** — no adaptive lambda:

1. **Regular crossover** (`choose_crossover`, all 4 operators) at `crossover_rate`.
2. **Regular mutation** (`choose_mutation`, all 9 operators) at `mutation_rate`.
3. **GI** (`mutation_random`, `injection_rate`): replaces a handful of random genes
   with brand-new random facility assignments, injecting fresh genetic material rather
   than perturbing the existing assignment.

> **Design notes:** Fixed-rate version reported here; adaptive counterpart is
> `ImprovedAdaptiveGEAScenario3` (§15).

---

## 12. Adaptive GEA — `StandardAdaptiveGEA` / `ImprovedAdaptiveGEA` (`src/algos/adaptive/gea.py`)

### `ImprovedAdaptiveGEA` (config: `adaptive_gea`)

Full `ImprovedGEA` (all five operator stages) with **lambda-adaptive rates on every stage**
(crossover, mutation, RC, DM, GI) — see §4 adaptive rate mechanism.

Hydra config: `scripts/conf/adaptive_gea.yaml` → results file `adaptivegea.json`.

### `StandardAdaptiveGEA` (config: `standard_adaptive_gea`)

Same adaptive application rates on the thesis scaffold with pool survivor selection.

---

## 13. Adaptive GEA Scenario 1 (`src/algos/adaptive/gea_scenario.py`)

`ImprovedAdaptiveGEAScenario1` (config: `adaptive_gea_scenario_1`) /
`StandardAdaptiveGEAScenario1` (config: `standard_adaptive_gea_scenario_1`)

`ImprovedGEAScenario1` with adaptive crossover, mutation, and RC rates.

---

## 14. Adaptive GEA Scenario 2 (`src/algos/adaptive/gea_scenario.py`)

`ImprovedAdaptiveGEAScenario2` / `StandardAdaptiveGEAScenario2`

`ImprovedGEAScenario2` with adaptive crossover, mutation, and DM rates.

---

## 15. Adaptive GEA Scenario 3 (`src/algos/adaptive/gea_scenario.py`)

`ImprovedAdaptiveGEAScenario3` / `StandardAdaptiveGEAScenario3`

`ImprovedGEAScenario3` with adaptive crossover, mutation, and GI rates.

---

## Not implemented

`GSAIS-KMeans`, `GSAIS-DBSCAN`, and `GSAIS-NN` (Gromov, Sohrabi, & Fathollahi-Fard,
*Genetic Speciation Algorithm with Interplay among Species*, under review at
*Algorithms*) are **not implemented**: the paper is unpublished and not yet available,
so there is no reliable basis for a faithful implementation of the speciation/interplay
mechanism. This is acceptable at the current stage.

---

## Hyperparameter Tuning (`scripts/tune_algorithm.py`)

Tuning is performed using the **Taguchi method**: an orthogonal array experiment that
efficiently explores a discrete grid of candidate hyperparameter combinations with far
fewer runs than a full factorial search.

### Method

For each algorithm, **three candidate values** are defined per tunable hyperparameter.
These candidate values are provided by the research team based on domain knowledge and
preliminary exploration. The candidates are arranged into a Taguchi orthogonal array
(e.g. L9 for up to 4 factors, L27 for up to 13 factors), where each row of the array
is one parameter combination to evaluate.

Each combination is run **30 times** on a single representative test problem, and the
result considered for that combination is the **average of the 30 minimum costs** found
across those runs. The winning combination is the one with the lowest average.

After selecting the best combination, **Relative Percentage Deviation (RPD)** is
computed for each candidate combination:

```
RPD = (cost_candidate − cost_best) / cost_best × 100 %
```

where `cost_best` is the minimum cost found across all combinations. RPD plots and a
parameter-combination table are reported in the paper.

### Infrastructure (`scripts/tune_algorithm.py`)

The current implementation uses **Optuna** (Bayesian / TPE search) over continuous
parameter ranges as a working approximation until the Taguchi candidate values are
finalized by the research team. Once the three candidate values per parameter are
provided, the search will be replaced with a Taguchi orthogonal-array grid.

Each algorithm has its own tune config in `scripts/conf/tune_algorithm/<algo>.yaml`,
which specifies:
- `runs`: number of independent runs per candidate (target: 30)
- `n_candidates`: number of Optuna trials (will become number of Taguchi rows)
- `param_space`: `[min, max]` bounds per parameter (will become `[v1, v2, v3]` lists)
- `output_file`: path for tuning results JSON

To run tuning for a specific algorithm:
```bash
poetry run python scripts/tune_algorithm.py --config-name="tune_algorithm/<algo>"
# e.g.:
poetry run python scripts/tune_algorithm.py --config-name="tune_algorithm/gea"
```

### Algorithms in scope for tuning

Improved configs are Optuna-tuned; each has a matching `standard_*` config whose operator
rates are copied from the tuned improved counterpart.

| Config | Key tunable parameters |
|---|---|
| `standard` / `ga` | `crossover_rate`, `mutation_rate` |
| `standard_gea` / `gea` | `crossover_rate`, `mutation_rate`, `rc_rate`, `dm_rate`, `injection_rate`, `stagnation_limit`, `immigrant_rate` |
| `standard_adaptive` / `adaptive` | `crossover_rate`, `mutation_rate`, `alpha`, `lambda_min`, `lambda_max`, `stagnation_limit`, `immigrant_rate` |
| `standard_sa` / `sa` | `initial_temperature`, `cooling_rate`, `min_temperature` |
| `standard_pso` / `pso` | `inertia_weight`, `cognitive_weight`, `social_weight`, `stagnation_limit`, `immigrant_rate` |
| `standard_hybrid_ga_pso` / `hybrid_ga_pso` | `pso_fraction`, `inertia_weight`, `cognitive_weight`, `social_weight`, `crossover_rate`, `mutation_rate` |
| `standard_hybrid_ga_sa` / `hybrid_ga_sa` | `crossover_rate`, `mutation_rate`, `initial_temperature`, `cooling_rate` |
| `standard_gea_scenario_1` / `gea_scenario_1` | `crossover_rate`, `mutation_rate`, `rc_rate`, … |
| `standard_gea_scenario_2` / `gea_scenario_2` | `crossover_rate`, `mutation_rate`, `dm_rate`, … |
| `standard_gea_scenario_3` / `gea_scenario_3` | `crossover_rate`, `mutation_rate`, `injection_rate`, … |
| `standard_adaptive_gea` / `adaptive_gea` | all GEA rates + `alpha`, `lambda_min`, `lambda_max`, … |
| `standard_adaptive_gea_scenario_*` / `adaptive_gea_scenario_*` | scenario-specific rates + adaptive knobs |

---

## Final Experiment

After tuning, each algorithm is run on every test case to produce the paper's
main results table.

### Protocol

- **Runs per algorithm per test case**: 30 independent runs, each from a fresh random
  seed.
- **Time limit**: 1000 seconds of wall-clock time per run.
- **Statistics reported**: Min, Max, AVG, Std across the 30 runs per (algorithm, test
  case) cell. Additionally, the **number of Best** (algorithm achieves the best known
  Min on that instance) and **number of Unique Best** (algorithm is the only one
  achieving that Min) are reported at the bottom of the main table, computed from the
  Min column.

### Data stored per run

Each run stores the following for analysis and plotting:

1. **Best cost** — the minimum cost found at the end of the run (last element of the
   BestCosts list below).
2. **BestCosts list** (`cost_history` in `GALogger`) — the running best cost recorded
   at the end of every iteration, i.e. `best_cost[t]` for `t = 1, 2, ..., T`. Used to
   plot minimization curves and to compute Std interval plots (§3.3.14). Saved only in
   final experiment runs (`run.py`), not during tuning (memory-intensive for 30×n_candidates
   trials). Also stored: `nfe_history[t]` — the cumulative NFE at the same snapshot points,
   pairing each cost sample with its matching computational budget.
3. **NFE (Number of Function Evaluations)** (`nfe` in `GALogger`) — incremented once
   per individual cost evaluation: population initialization, each crossover/mutant/immigrant
   child, and each local-search probe. The final total is stored per run; the average
   across 30 runs per (algorithm, instance) pair is reported in the NFE table.

### Analysis outputs

| Output | Description |
|---|---|
| **Main results table** | Min / Max / AVG / Std × algorithm × test case; #Best and #Unique Best at the bottom. |
| **Std interval plots** | Standard deviation across 30 runs plotted as a function of iteration, for all algorithms on each test case — measures robustness and convergence stability. |
| **CPU time table** | Wall-clock time per run; plotted separately for small-scale and large-scale instances. |
| **Optimality Gap (OG) table** | `(algo_min − exact) / exact × 100 %` for small-scale instances where the exact solution is known (e.g. `c201535`). |
| **Hitting time** | Across the 30 runs, identifies the best run and records the iteration and wall-clock time at which the final minimum was first reached. Reported as a table or plot. |
| **NFE table** | Rounded average NFE for each algorithm × test case (2D table: rows = instances, columns = algorithms). Enables comparison of computational effort relative to solution quality. |

> **Implementation status:** `run.py` stores Min/Max/AVG/Std, hitting time, NFE
> (per-run and mean), and full BestCosts + NFE histories per run (the latter only in
> final-experiment runs, not during tuning). OG (Optimality Gap) is **pending** —
> requires known exact solutions for the small-scale instances.

## Run budgets and reproducibility

`run(time_limit=seconds)` uses a wall-clock budget instead of the configured iteration
limit. Without a time limit, `iterations` is the stopping criterion. Initialization
and any in-run JIT compilation count against search time; model loading does not.
A generation is atomic, so the budget may be exceeded by the last generation (or
initialization). Use consistent JIT/cache preparation for scientific timing comparisons.
The best solution is retained over the entire run, even when SA/PSO moves worsen.
Hitting time and hitting iteration record the first generation-end observation of
that final minimum (iteration zero for initialization), not a per-evaluation timestamp.

Runner seeds use SHA-256 instead of Python's process-randomized `hash`. Randomized
batch repair seeds each row from the seeded NumPy stream inside its parallel task,
so worker-local RNG state does not leak between runs. This changes historical random
trajectories of improved algorithms; their old golden costs must be regenerated.

JSON includes `run_records` with seed, configuration, Git revision/dirty flag/diff hash,
solution, wall/CPU/search times, hitting iteration and cost/time/NFE/adaptation histories.
`best_run` identifies the lowest-cost successful run, resolving ties by run number.
CPU elapsed is total worker process CPU time; it includes loading and compilation.
The source diff hash identifies tracked edits; keep the actual source alongside results.
Taguchi tuning and full paper experiments are not performed by this implementation.
