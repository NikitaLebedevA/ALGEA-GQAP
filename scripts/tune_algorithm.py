#!/usr/bin/env python3
"""Tunes an algorithm's own rate/threshold knobs (e.g. crossover_rate, mutation_rate,
stagnation_limit -- whichever ones conf/tune_algorithm/<algo>.yaml's param_space lists for
the algorithm in question) -- components (repair_class, selector, if the algorithm has
them) and the compute budget (population_size, iterations) are left as configured; see
tune_components.py for tuning components instead.

Works against whichever algorithm conf/tune_algorithm/<algo>.yaml's defaults list selects
(one file per algorithm, e.g. `--config-name=tune_algorithm/sa`); the algorithm class
itself is read from `ga._target_`, not hard-coded.

Mirrors tune_components.py's approach (Optuna TPE search against a fixed baseline) -- see
tuning.py for the shared scoring/evaluation/yaml-writing helpers.
"""

import json
from pathlib import Path

import hydra
import optuna
from omegaconf import DictConfig, OmegaConf

from utils.runner import timestamp
from utils.tuning import evaluate, relative_score, suggest_param, update_yaml_fields

SCRIPT_DIR = Path(__file__).resolve().parent

optuna.logging.set_verbosity(optuna.logging.WARNING)


@hydra.main(version_base=None, config_path="conf", config_name="tune_algorithm/gea")
def main(cfg: DictConfig) -> None:
    datasets = list(cfg.datasets.names)
    time_limit = cfg.run.time_limit
    workers = int(cfg.run.workers)

    base_ga_cfg = OmegaConf.to_container(cfg.ga, resolve=True)
    algo_class = base_ga_cfg["_target_"].rsplit(".", 1)[-1]

    fixed_params = {k: base_ga_cfg[k] for k in ("population_size", "iterations")}
    # Not every algorithm has these (e.g. StandardGA has no repair_class/selector knobs
    # to leave fixed) -- only report the ones actually present in this algorithm's config.
    components = {k: base_ga_cfg[k] for k in ("repair_class", "selector") if k in base_ga_cfg}

    runs = cfg.tune.runs
    param_space = OmegaConf.to_container(cfg.tune.param_space, resolve=True)

    baseline_params = {key: base_ga_cfg[key] for key in param_space}

    print(f"[{timestamp()}] Algorithm: {algo_class}")
    print(f"[{timestamp()}] Fixed: population/iterations={fixed_params}, components={components}")
    print(f"[{timestamp()}] Baseline: {baseline_params}")

    baseline_per_dataset = evaluate(baseline_params, base_ga_cfg, algo_class, datasets, runs, time_limit, workers)
    candidate_records = [{"params": baseline_params, "per_dataset": baseline_per_dataset, "overall_score": 1.0}]

    print(f"[{timestamp()}] Tuning algorithm params with Optuna TPE: {cfg.tune.n_candidates - 1} trials x {len(datasets)} datasets x {runs} runs")

    def objective(trial: optuna.Trial) -> float:
        params = {key: suggest_param(trial, key, low, high) for key, (low, high) in param_space.items()}
        print(f"\n[{timestamp()}] Trial {trial.number + 1}/{cfg.tune.n_candidates - 1}: {params}")

        per_dataset = evaluate(params, base_ga_cfg, algo_class, datasets, runs, time_limit, workers)
        score = relative_score(per_dataset, baseline_per_dataset, datasets, runs)
        candidate_records.append({"params": params, "per_dataset": per_dataset, "overall_score": score})
        return score

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=cfg.tune.seed))
    study.optimize(objective, n_trials=max(0, cfg.tune.n_candidates - 1))

    best = min(candidate_records, key=lambda c: c["overall_score"])

    print(f"\n[{timestamp()}] Best candidate: {best['params']} (score={best['overall_score']:.4f})")

    ga_path = SCRIPT_DIR / cfg.tune.ga_path
    if best["params"] is baseline_params:
        print(f"[{timestamp()}] Baseline remains best -- leaving {ga_path.name} unchanged")
    else:
        update_yaml_fields(ga_path, best["params"])
        print(f"[{timestamp()}] Wrote best candidate to {ga_path}")

    artifact = {
        "timestamp": timestamp(),
        "algorithm": algo_class,
        "fixed_params": fixed_params,
        "components": components,
        "datasets": datasets,
        "runs_per_dataset": runs,
        "candidates": candidate_records,
        "best": best,
    }

    out = SCRIPT_DIR / cfg.tune.output_file
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(artifact, f, indent=2)

    print(f"[{timestamp()}] Saved tuning artifact to {out}")


if __name__ == "__main__":
    main()
