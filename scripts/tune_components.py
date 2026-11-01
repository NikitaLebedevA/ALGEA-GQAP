#!/usr/bin/env python3
"""Tunes scripts/conf/components/common.yaml's repair/selection knobs (algorithm
parameters -- population_size, iterations, crossover_rate, mutation_rate, etc. -- are left
as configured in conf/tune.yaml's `ga` block, inherited from gea.yaml; see
tune_algorithm.py for tuning those instead). Only GEA is used as the evaluation
algorithm.

Search is delegated to Optuna's TPE sampler instead of blind random sampling: each trial's
result feeds back into the sampler's model of which regions of conf/tune.yaml's param_space
tend to score well, so later trials concentrate there instead of sampling uniformly forever.
See tuning.py for the shared scoring/evaluation/yaml-writing helpers.
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


@hydra.main(version_base=None, config_path="conf", config_name="tune_components")
def main(cfg: DictConfig) -> None:
    datasets = list(cfg.datasets.names)
    time_limit = cfg.run.time_limit
    workers = int(cfg.run.workers)

    base_ga_cfg = OmegaConf.to_container(cfg.ga, resolve=True)
    algo_class = base_ga_cfg["_target_"].rsplit(".", 1)[-1]
    if algo_class != "ImprovedGEA":
        raise SystemExit(
            f"tune_components.py only tunes ImprovedGEA, got {algo_class} -- run with ga=gea's defaults intact"
        )

    algorithm_params = {
        k: base_ga_cfg[k] for k in ("population_size", "iterations", "crossover_rate", "mutation_rate")
    }

    runs = cfg.tune.runs
    param_space = OmegaConf.to_container(cfg.tune.param_space, resolve=True)

    baseline_components = {key: base_ga_cfg[key.split(".", 1)[0]][key.split(".", 1)[1]] for key in param_space}

    print(f"[{timestamp()}] Algorithm params (fixed): {algorithm_params}")
    print(f"[{timestamp()}] Baseline: {baseline_components}")

    baseline_per_dataset = evaluate(baseline_components, base_ga_cfg, "GEA", datasets, runs, time_limit, workers)
    candidate_records = [{"components": baseline_components, "per_dataset": baseline_per_dataset, "overall_score": 1.0}]

    print(f"[{timestamp()}] Tuning components with Optuna TPE: {cfg.tune.n_candidates - 1} trials x {len(datasets)} datasets x {runs} runs")

    def objective(trial: optuna.Trial) -> float:
        components = {key: suggest_param(trial, key, low, high) for key, (low, high) in param_space.items()}
        print(f"\n[{timestamp()}] Trial {trial.number + 1}/{cfg.tune.n_candidates - 1}: {components}")

        per_dataset = evaluate(components, base_ga_cfg, "GEA", datasets, runs, time_limit, workers)
        score = relative_score(per_dataset, baseline_per_dataset, datasets, runs)
        candidate_records.append({"components": components, "per_dataset": per_dataset, "overall_score": score})
        return score

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=cfg.tune.seed))
    study.optimize(objective, n_trials=max(0, cfg.tune.n_candidates - 1))

    best = min(candidate_records, key=lambda c: c["overall_score"])

    print(f"\n[{timestamp()}] Best candidate: {best['components']} (score={best['overall_score']:.4f})")

    if best["components"] is baseline_components:
        print(f"[{timestamp()}] Baseline remains best -- leaving common.yaml unchanged")
    else:
        components_path = SCRIPT_DIR / cfg.tune.components_path
        update_yaml_fields(components_path, best["components"])
        print(f"[{timestamp()}] Wrote best candidate to {components_path}")

    artifact = {
        "timestamp": timestamp(),
        "algorithm": "GEA",
        "algorithm_params": algorithm_params,
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
