import os

# Must be set before numba/numpy are imported anywhere in this process -- numba reads
# NUMBA_NUM_THREADS once at its own first import and refuses to change it afterward, and
# BLAS backends (used by numpy under the hood) similarly lock in their thread count at
# first use. Worker processes (forked or spawned) inherit these from the parent's
# environment at process-creation time, so setting them here -- before the `from src...`
# imports below pull numba/numpy in -- is what makes them take effect in every worker.
# Each worker does its own numeric work single-threaded; the parallelism comes entirely
# from the process pool, so N workers don't also each spawn a thread pool across every
# core (oversubscription). run_test.sbatch sets the same four BLAS variables at the shell
# level already; setdefault here just makes ad-hoc local runs (outside sbatch) safe too,
# without overriding whatever the caller already set.
for _env_var in (
    "NUMBA_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_env_var, "1")

import hashlib
import json
import math
import subprocess
from pathlib import Path
from functools import lru_cache
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List

import numpy as np

import hydra
from omegaconf import OmegaConf

from src.data.model_loader import load_model
from src.seeding import seed_all
from utils.labels import algo_label

# Lets conf/run/common.yaml derive output_file from ga._target_ via algo_label instead of
# each config repeating its own output filename. Registered here (imported by every
# entry-point script before its @hydra.main composes a config) rather than in each script.
OmegaConf.register_new_resolver("algo_label", algo_label, replace=True)


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    # numpy handles inf/nan without raising (statistics.stdev raises AttributeError
    # on inf in Python 3.11). ddof=1 matches statistics.stdev (sample std).
    arr = np.array(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr, ddof=1)) if len(values) > 1 else 0.0,
    }


def average_lambda_trajectories(trajectories: List[List[dict]]) -> List[dict]:
    """Element-wise mean of λ trajectories (truncate to shortest run)."""
    valid = [t for t in trajectories if t]
    if not valid:
        return []
    n_iters = min(len(t) for t in valid)
    keys = valid[0][0].keys()
    averaged: List[dict] = []
    for i in range(n_iters):
        snap: dict[str, float] = {}
        for key in keys:
            vals = [float(t[i][key]) for t in valid if key in t[i]]
            if vals:
                snap[key] = float(np.mean(vals))
        averaged.append(snap)
    return averaged


def experiment_seed(dataset_name: str, algo_label: str, run_num: int) -> int:
    payload = json.dumps([dataset_name, algo_label, run_num], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


@lru_cache(maxsize=1)
def source_version() -> dict:
    root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
        diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=root)
        return {"revision": revision, "dirty": bool(status), "diff_sha256": hashlib.sha256(diff).hexdigest()}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None, "diff_sha256": None}


def run_single_experiment(
    dataset_name: str,
    algo_label: str,
    run_num: int,
    ga_cfg: dict,
    time_limit: float,
):
    seed = experiment_seed(dataset_name, algo_label, run_num)
    seed_all(seed)

    start = time.perf_counter()
    cpu_start = time.process_time()
    metadata = {"seed": seed, "config": ga_cfg, "source": source_version()}
    try:
        model = load_model(dataset_name)
        ga = hydra.utils.instantiate(ga_cfg, model=model)
        best = ga.run(time_limit=time_limit)
        elapsed = time.perf_counter() - start
        hitting_time = getattr(ga, "hitting_time", None)
        nfe = getattr(ga.logger, "nfe", 0)
        cost_history = getattr(ga.logger, "cost_history", [])
        lambda_history = getattr(ga, "lambda_history", None)

        if not math.isfinite(best.cost):
            return {
                **metadata,
                "dataset": dataset_name,
                "run": run_num,
                "cost": None,
                "elapsed": elapsed,
                "hitting_time": None,
                "nfe": nfe,
                "cost_history": [],
                "lambda_history": [],
                "error": f"infeasible: cost={best.cost}",
            }

        return {
            **metadata,
            "dataset": dataset_name,
            "run": run_num,
            "cost": best.cost,
            "elapsed": elapsed,
            "hitting_time": hitting_time,
            "nfe": nfe,
            "cost_history": cost_history,
            "lambda_history": lambda_history or [],
            "adaptation_history": getattr(ga, "adaptation_history", []),
            "initial_lambdas": getattr(ga, "initial_lambdas", {}),
            "hitting_iteration": getattr(ga, "hitting_iteration", 0),
            "time_history": ga.logger.time_history,
            "nfe_history": ga.logger.nfe_history,
            "search_elapsed": ga.logger.elapsed(),
            "cpu_elapsed": time.process_time() - cpu_start,
            "best_permutation": best.permutation.tolist(),
            "error": None,
        }

    except Exception as e:
        return {
            **metadata,
            "dataset": dataset_name,
            "run": run_num,
            "cost": None,
            "elapsed": time.perf_counter() - start,
            "hitting_time": None,
            "nfe": 0,
            "cost_history": [],
            "lambda_history": [],
            "error": str(e),
        }


def run_all_experiments(
    datasets: List[str],
    algo_label: str,
    ga_cfg: dict,
    time_limit: float,
    runs: int,
    workers: int,
    track_history: bool = False,
) -> List[dict]:
    """Runs every (dataset, run) combination across a process pool and returns one stats
    dict per dataset, in the same shape run_dataset_tests used to produce. Each run is an
    independent GA call (fresh model load, own seed), so this parallelizes flatly across
    every (dataset, run) pair rather than nesting a pool per dataset.
    """
    tasks = [(ds, r) for ds in datasets for r in range(1, runs + 1)]
    results_by_dataset: Dict[str, List[float]] = {ds: [] for ds in datasets}
    runtimes_by_dataset: Dict[str, List[float]] = {ds: [] for ds in datasets}
    hitting_times_by_dataset: Dict[str, List[float]] = {ds: [] for ds in datasets}
    nfes_by_dataset: Dict[str, List[int]] = {ds: [] for ds in datasets}
    histories_by_dataset: Dict[str, List[list]] = {ds: [] for ds in datasets}
    lambda_histories_by_dataset: Dict[str, List[list]] = {ds: [] for ds in datasets}
    records_by_dataset: Dict[str, List[dict]] = {ds: [] for ds in datasets}
    errors_by_dataset: Dict[str, int] = {ds: 0 for ds in datasets}
    completed_by_dataset: Dict[str, int] = {ds: 0 for ds in datasets}

    print(f"   Running {len(tasks)} experiments ({len(datasets)} datasets x {runs} runs) across {workers} workers...")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_single_experiment, ds, algo_label, r, ga_cfg, time_limit): (ds, r) for ds, r in tasks
        }

        for future in as_completed(futures):
            ds, _ = futures[future]
            res = future.result()

            records_by_dataset[ds].append(res)
            runtimes_by_dataset[ds].append(res["elapsed"])
            if res["error"]:
                errors_by_dataset[ds] += 1
            else:
                results_by_dataset[ds].append(res["cost"])
                if res["hitting_time"] is not None:
                    hitting_times_by_dataset[ds].append(res["hitting_time"])
                nfes_by_dataset[ds].append(res["nfe"])
                if track_history and res["cost_history"]:
                    histories_by_dataset[ds].append(res["cost_history"])
                if res["lambda_history"]:
                    lambda_histories_by_dataset[ds].append(res["lambda_history"])
            completed_by_dataset[ds] += 1

            # Print one aggregated line per dataset, once all its runs are in, instead of
            # one line per individual run -- with many datasets/runs/workers running at
            # once, per-run lines from concurrent futures interleave into unreadable noise.
            if completed_by_dataset[ds] == runs:
                stats = calculate_statistics(results_by_dataset[ds])
                total_runtime = sum(runtimes_by_dataset[ds])
                avg_hit = statistics.mean(hitting_times_by_dataset[ds]) if hitting_times_by_dataset[ds] else None
                avg_nfe = round(statistics.mean(nfes_by_dataset[ds])) if nfes_by_dataset[ds] else None
                hit_note = f" ; avg hitting time = {avg_hit:.1f}s" if avg_hit is not None else ""
                nfe_note = f" ; avg NFE = {avg_nfe:,}" if avg_nfe is not None else ""
                error_note = f" ({errors_by_dataset[ds]} errors)" if errors_by_dataset[ds] else ""
                print(
                    f"   {ds} run → avg cost = {stats['mean']:.2f} ; std = {stats['std']:.2f} ; "
                    f"total runtime = {total_runtime:.2f}s{hit_note}{nfe_note}{error_note}"
                )

    # Completion order varies across workers. Keep positional per-run arrays in
    # run-number order so downstream paired comparisons do not pair random runs.
    best_runs = {}
    for ds in datasets:
        records = sorted(records_by_dataset[ds], key=lambda r: r["run"])
        records_by_dataset[ds] = records
        successful = [r for r in records if r["error"] is None]
        results_by_dataset[ds] = [r["cost"] for r in successful]
        nfes_by_dataset[ds] = [r["nfe"] for r in successful]
        histories_by_dataset[ds] = [r["cost_history"] for r in successful if r["cost_history"]]
        lambda_histories_by_dataset[ds] = [r["lambda_history"] for r in successful if r["lambda_history"]]
        best = min(successful, key=lambda r: (r["cost"], r["run"]), default=None)
        best_runs[ds] = best["run"] if best else None

    return [
        {
            "dataset": ds,
            "run_records": {algo_label: records_by_dataset[ds]},
            "best_run": {algo_label: best_runs[ds]},
            "results": {
                algo_label: {
                    **calculate_statistics(results_by_dataset[ds]),
                    "per_run": results_by_dataset[ds],
                }
            },
            "runtime": {
                algo_label: {**calculate_statistics(runtimes_by_dataset[ds]), "total": sum(runtimes_by_dataset[ds])}
            },
            "hitting_time": {
                algo_label: {
                    "mean": (
                        float(statistics.mean(hitting_times_by_dataset[ds])) if hitting_times_by_dataset[ds] else None
                    ),
                    "min": float(min(hitting_times_by_dataset[ds])) if hitting_times_by_dataset[ds] else None,
                    "max": float(max(hitting_times_by_dataset[ds])) if hitting_times_by_dataset[ds] else None,
                    "std": (
                        float(statistics.stdev(hitting_times_by_dataset[ds]))
                        if len(hitting_times_by_dataset[ds]) > 1
                        else 0.0
                    ),
                }
            },
            "nfe": {
                algo_label: {
                    "mean": round(statistics.mean(nfes_by_dataset[ds])) if nfes_by_dataset[ds] else None,
                    "std": round(statistics.stdev(nfes_by_dataset[ds])) if len(nfes_by_dataset[ds]) > 1 else 0,
                    "per_run": nfes_by_dataset[ds],
                }
            },
            **({"cost_history": {algo_label: histories_by_dataset[ds]}} if track_history else {}),
            **(
                {
                    "lambda_history": {
                        algo_label: {
                            "per_run": lambda_histories_by_dataset[ds],
                            "mean": average_lambda_trajectories(lambda_histories_by_dataset[ds]),
                        }
                    }
                }
                if lambda_histories_by_dataset[ds]
                else {}
            ),
            "errors": errors_by_dataset[ds],
        }
        for ds in datasets
    ]
