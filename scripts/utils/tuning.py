"""Shared helpers for the tuning scripts (tune_components.py tunes repair/selection knobs,
tune_algorithm.py tunes crossover/mutation/stagnation knobs). Both follow the same shape:
evaluate a fixed baseline once, let Optuna's TPE sampler search a param_space against that
fixed reference, and write the winner back into a yaml file if it beats the baseline.
"""

import copy
import math
import re
import statistics
from pathlib import Path
from typing import Dict, List

import optuna

from utils.runner import run_all_experiments


def effective_mean(dataset_entry: dict, runs: int) -> float:
    """calculate_statistics reports mean=0 when a dataset had zero successful runs (e.g.
    every run errored), which would look like a perfect score -- treat that as worst-case
    instead so a candidate can't "win" a dataset by failing on it."""
    if dataset_entry["errors"] >= runs:
        return math.inf
    return dataset_entry["mean"]


def relative_score(per_dataset: dict, baseline_per_dataset: dict, datasets: List[str], runs: int) -> float:
    """Mean, across datasets, of a candidate's cost relative to a fixed baseline's cost on
    that dataset. A fixed reference (rather than "best seen so far among trials") keeps the
    objective stable for Optuna -- "best so far" would retroactively change past trials'
    scores every time a new best shows up, which confuses the sampler. Relative (not
    absolute) scoring also keeps datasets with much larger costs from dominating the
    average."""
    ratios = []
    for ds in datasets:
        baseline_mean = effective_mean(baseline_per_dataset[ds], runs)
        if not math.isfinite(baseline_mean):
            continue
        mean = effective_mean(per_dataset[ds], runs)
        ratios.append(mean / (baseline_mean + 1e-9) if math.isfinite(mean) else math.inf)
    return float(statistics.mean(ratios)) if ratios else math.inf


def apply_overrides(base_ga_cfg: dict, overrides: Dict[str, object]) -> dict:
    """Sets each `overrides` entry onto a copy of base_ga_cfg. A dotted key
    ("section.field", e.g. "repair_class.subsample_size") reaches into a nested sub-dict; a
    plain key (e.g. "crossover_rate") sets a top-level field directly -- so this works for
    both tune_components.py's nested repair_class/selector knobs and tune_algorithm.py's
    flat algorithm knobs."""
    ga_cfg = copy.deepcopy(base_ga_cfg)
    for key, value in overrides.items():
        if "." in key:
            section, field = key.split(".", 1)
            ga_cfg[section][field] = value
        else:
            ga_cfg[key] = value
    return ga_cfg


def evaluate(
    overrides: Dict[str, object],
    base_ga_cfg: dict,
    algo_label: str,
    datasets: List[str],
    runs: int,
    time_limit: float,
    workers: int,
) -> dict:
    """Runs every dataset for `runs` runs with `overrides` applied on top of base_ga_cfg, and
    returns {dataset: {mean, median, min, max, std, runtime, errors}}."""
    ga_cfg = apply_overrides(base_ga_cfg, overrides)
    per_dataset_results = run_all_experiments(datasets, algo_label, ga_cfg, time_limit, runs, workers)
    return {
        res["dataset"]: {**res["results"][algo_label], "runtime": res["runtime"][algo_label], "errors": res["errors"]}
        for res in per_dataset_results
    }


def suggest_param(trial: optuna.Trial, key: str, low, high):
    """suggest_int when both bounds are plain ints (e.g. stagnation_limit's [10, 60]),
    suggest_float otherwise -- lets conf/tune*.yaml's param_space stay a plain [low, high]
    pair per parameter instead of needing to spell out each one's type."""
    if isinstance(low, int) and isinstance(high, int):
        return trial.suggest_int(key, low, high)
    return trial.suggest_float(key, float(low), float(high))


def update_yaml_fields(path: Path, fields: Dict[str, object]) -> None:
    """Regex-replaces each `<field>: ...` line in place, preserving the rest of the file
    (comments, ordering, untouched fields) instead of a full yaml dump, which would lose
    hand-written comments. Dotted keys ("section.field") use only the part after the last
    "." as the line to match, since field names are unique within one of these config files
    regardless of nesting."""
    text = path.read_text()
    for dotted_key, value in fields.items():
        field = dotted_key.rsplit(".", 1)[-1]
        pattern = re.compile(rf"^(\s*{re.escape(field)}:\s*).*$", re.MULTILINE)
        text, n = pattern.subn(lambda m: f"{m.group(1)}{value}", text, count=1)
        if n == 0:
            raise ValueError(f"field {field!r} not found in {path}")
    path.write_text(text)
