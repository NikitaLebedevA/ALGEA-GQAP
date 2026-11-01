#!/usr/bin/env python3
"""
Paired Wilcoxon signed-rank tests on per-run benchmark costs.

Matches the reference workflow:
  - One sheet / dataset with columns = algorithms, 30 rows = independent runs
  - Focal algorithm ("ours") compared to each competitor on every dataset
  - scipy.stats.wilcoxon(x, y, alternative="two-sided") on paired runs
  - diff = focal - competitor; median(diff) < 0 → win (+) when p < alpha

Requires per-run costs in results JSON (``results.*.per_run``).
Re-run benchmarks after updating runner.py if JSON files lack this field.

Usage:
    python scripts/wilcoxon_tests.py
    python scripts/wilcoxon_tests.py --family improved --focal adaptive_gea
    python scripts/run_all_statistical_tests.py
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from build_results_table import (  # noqa: E402
    IMPROVED_ALGO_ORDER,
    STANDARD_ALGO_ORDER,
    _get_stats,
    load_results,
)
from statistical_tests import (  # noqa: E402
    PAPER_ALGO_LABELS,
    PAPER_DATASET_ORDER,
    _display_label,
)

Outcome = str  # '+', '=', '-'

# Default focal algorithm per family (paper names: ALGEA / Standard Adaptive GEA).
DEFAULT_FOCAL: dict[str, str] = {
    "improved": "adaptive_gea",
    "standard": "standard_adaptive_gea",
}


@dataclass(frozen=True)
class PairwiseRecord:
    wins: int
    ties: int
    losses: int

    def formatted(self) -> str:
        return f"{self.wins}+/{self.ties}=/"+f"{self.losses}-"


def _per_run_costs(record: dict) -> list[float] | None:
    stats = _get_stats(record, "results") or {}
    per_run = stats.get("per_run")
    if not per_run:
        return None
    values = [float(v) for v in per_run if v is not None and np.isfinite(v)]
    return values or None


def load_per_run_matrix(
    data: dict[str, dict[str, dict]],
    algorithms: list[str],
    datasets: list[str],
) -> tuple[list[str], dict[str, dict[str, list[float]]]]:
    """Return (used_datasets, {dataset: {algo_stem: [costs]}})."""
    used: list[str] = []
    matrix: dict[str, dict[str, list[float]]] = {}

    for dataset in datasets:
        row: dict[str, list[float]] = {}
        complete = True
        for algo in algorithms:
            if algo not in data or dataset not in data[algo]:
                complete = False
                break
            costs = _per_run_costs(data[algo][dataset])
            if not costs:
                complete = False
                break
            row[algo] = costs
        if complete:
            used.append(dataset)
            matrix[dataset] = row

    return used, matrix


def paired_wilcoxon_compare(
    focal_runs: list[float],
    competitor_runs: list[float],
    *,
    alpha: float,
) -> tuple[Outcome, float]:
    """
    Paired Wilcoxon test: focal (x) vs competitor (y).

    Returns (sign, p_value) where sign is +/=/- from focal's perspective.
    Win (+): p < alpha and median(focal - competitor) < 0 (focal lower cost).
    """
    x = pd.Series(focal_runs, dtype=float)
    y = pd.Series(competitor_runs, dtype=float)
    valid = ~(x.isna() | y.isna())
    x = x[valid].to_numpy(dtype=float)
    y = y[valid].to_numpy(dtype=float)

    if len(x) == 0:
        return "=", float("nan")

    diff = x - y
    if np.allclose(diff, 0):
        return "=", 1.0

    try:
        _, p_value = wilcoxon(x, y, alternative="two-sided")
    except ValueError:
        return "=", 1.0

    p_value = float(p_value)
    if p_value >= alpha:
        return "=", p_value
    if np.median(diff) < 0:
        return "+", p_value
    return "-", p_value


def build_wilcoxon_per_dataset(
    matrix: dict[str, dict[str, list[float]]],
    datasets: list[str],
    focal_stem: str,
    focal_label: str,
    competitors: list[str],
    competitor_labels: list[str],
    *,
    alpha: float,
) -> pd.DataFrame:
    """One row per dataset — same layout as the reference Wilcoxon Excel output."""
    rows: list[dict[str, str | float]] = []
    for dataset in datasets:
        row: dict[str, str | float] = {"Function": dataset}
        focal_runs = matrix[dataset][focal_stem]
        for stem, label in zip(competitors, competitor_labels, strict=True):
            sign, p_value = paired_wilcoxon_compare(
                focal_runs,
                matrix[dataset][stem],
                alpha=alpha,
            )
            row[f"{label}_p"] = p_value
            row[f"{label}_result"] = sign
        rows.append(row)
    return pd.DataFrame(rows)


def build_summary_counts(
    wilcoxon_df: pd.DataFrame,
    competitors: list[str],
    competitor_labels: list[str],
) -> pd.DataFrame:
    """Aggregate +/=/- counts across datasets for each competitor."""
    rows: list[dict[str, str | int]] = []
    for label in competitor_labels:
        col = f"{label}_result"
        if col not in wilcoxon_df.columns:
            continue
        signs = wilcoxon_df[col].astype(str)
        rows.append(
            {
                "Competitor": label,
                "Wins (+)": int((signs == "+").sum()),
                "Ties (=)": int((signs == "=").sum()),
                "Losses (-)": int((signs == "-").sum()),
                "Record": PairwiseRecord(
                    int((signs == "+").sum()),
                    int((signs == "=").sum()),
                    int((signs == "-").sum()),
                ).formatted(),
            }
        )
    return pd.DataFrame(rows)


def build_per_run_workbook(
    matrix: dict[str, dict[str, list[float]]],
    datasets: list[str],
    stems: list[str],
    labels: list[str],
) -> dict[str, pd.DataFrame]:
    """Input-format workbook: one sheet per dataset, columns = algorithms."""
    sheets: dict[str, pd.DataFrame] = {}
    for dataset in datasets:
        columns: dict[str, list[float]] = {}
        max_len = 0
        for stem, label in zip(stems, labels, strict=True):
            runs = matrix[dataset][stem]
            columns[label] = runs
            max_len = max(max_len, len(runs))
        # Pad shorter columns (should not happen) for a rectangular frame.
        for label, runs in columns.items():
            if len(runs) < max_len:
                columns[label] = runs + [float("nan")] * (max_len - len(runs))
        sheets[dataset[:31]] = pd.DataFrame(columns)
    return sheets


def write_excel_report(
    output_path: Path,
    *,
    family: str,
    focal_label: str,
    alpha: float,
    wilcoxon_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    per_run_sheets: dict[str, pd.DataFrame],
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        wilcoxon_df.to_excel(writer, sheet_name="Wilcoxon", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        for sheet_name, frame in per_run_sheets.items():
            frame.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        pd.DataFrame(
            [
                {"Field": "Family", "Value": family},
                {"Field": "Focal algorithm", "Value": focal_label},
                {"Field": "Alpha", "Value": alpha},
                {"Field": "Test", "Value": "Paired Wilcoxon signed-rank (scipy.stats.wilcoxon)"},
                {"Field": "Win (+)", "Value": "p < alpha and median(focal - competitor) < 0"},
                {"Field": "Tie (=)", "Value": "p >= alpha, or all differences zero"},
                {"Field": "Loss (-)", "Value": "p < alpha and median(focal - competitor) > 0"},
                {"Field": "Input layout", "Value": "Per-dataset sheets: 30 rows x algorithm columns"},
            ]
        ).to_excel(writer, sheet_name="Info", index=False)
    return output_path


def run_analysis(
    *,
    data: dict[str, dict[str, dict]],
    family: str,
    datasets: list[str],
    alpha: float,
    output_dir: Path,
    focal_stem: str | None,
) -> None:
    if family == "improved":
        algorithms = [a for a in IMPROVED_ALGO_ORDER if a in data]
    elif family == "standard":
        algorithms = [a for a in STANDARD_ALGO_ORDER if a in data]
    else:
        raise ValueError(f"Unknown family: {family}")

    focal = focal_stem or DEFAULT_FOCAL[family]
    if focal not in algorithms:
        print(f"  {family}: skipped — focal {focal!r} not in available results.")
        return

    labels = {_display_label(a, family): a for a in algorithms}
    stem_labels = {a: _display_label(a, family) for a in algorithms}
    focal_label = stem_labels[focal]
    competitors = [a for a in algorithms if a != focal]
    competitor_labels = [stem_labels[a] for a in competitors]

    used_datasets, matrix = load_per_run_matrix(data, algorithms, datasets)
    if not used_datasets:
        print(
            f"  {family}: skipped — no per-run costs in JSON. "
            f"Re-run benchmarks (runner now saves results.*.per_run)."
        )
        return

    if len(used_datasets) < len(datasets):
        print(f"  {family}: using {len(used_datasets)}/{len(datasets)} datasets with complete per-run data")

    wilcoxon_df = build_wilcoxon_per_dataset(
        matrix,
        used_datasets,
        focal,
        focal_label,
        competitors,
        competitor_labels,
        alpha=alpha,
    )
    summary_df = build_summary_counts(wilcoxon_df, competitors, competitor_labels)
    per_run_sheets = build_per_run_workbook(
        matrix,
        used_datasets,
        algorithms,
        [stem_labels[a] for a in algorithms],
    )

    family_dir = output_dir / family
    family_dir.mkdir(parents=True, exist_ok=True)

    wilcoxon_df.to_csv(family_dir / "wilcoxon_per_dataset.csv", index=False)
    summary_df.to_csv(family_dir / "wilcoxon_summary.csv", index=False)

    xlsx = write_excel_report(
        family_dir / f"Wilcoxon_Test_{family}.xlsx",
        family=family,
        focal_label=focal_label,
        alpha=alpha,
        wilcoxon_df=wilcoxon_df,
        summary_df=summary_df,
        per_run_sheets=per_run_sheets,
    )

    # Standalone per-run input workbook (reference input format).
    per_run_path = family_dir / f"Per_Run_Results_{family}.xlsx"
    with pd.ExcelWriter(per_run_path, engine="openpyxl") as writer:
        for sheet_name, frame in per_run_sheets.items():
            frame.to_excel(writer, sheet_name=sheet_name[:31], index=False)

    print(f"  {family}: focal={focal_label}, {len(used_datasets)} datasets → {xlsx}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=SCRIPT_DIR / "results")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "results" / "statistical_tests")
    parser.add_argument("--family", choices=["improved", "standard", "both"], default="both")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--focal",
        type=str,
        default=None,
        help="Focal algorithm JSON stem (default: adaptive_gea / standard_adaptive_gea)",
    )
    parser.add_argument("--dataset-order", choices=["paper", "config"], default="paper")
    args = parser.parse_args()

    datasets = PAPER_DATASET_ORDER
    if args.dataset_order == "config":
        from build_results_table import load_dataset_order

        datasets = load_dataset_order()

    print(f"Loading results from {args.results_dir} ...")
    data = load_results(args.results_dir)
    families = ["improved", "standard"] if args.family == "both" else [args.family]
    print(f"Running paired Wilcoxon (alpha = {args.alpha}) on {', '.join(families)} ...")

    for family in families:
        run_analysis(
            data=data,
            family=family,
            datasets=datasets,
            alpha=args.alpha,
            output_dir=args.output_dir,
            focal_stem=args.focal,
        )

    print(f"Done. Reports in {args.output_dir}")


if __name__ == "__main__":
    main()
