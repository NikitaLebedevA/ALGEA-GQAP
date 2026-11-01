#!/usr/bin/env python3
"""
Friedman test and Nemenyi post-hoc analysis on benchmark results.

For each criterion (mean, std, best, worst, hitting_time, nfe), algorithms are
ranked per dataset (lower value = better rank). Friedman tests whether average
ranks differ significantly; Nemenyi identifies which pairs differ at alpha.

Usage:
    python scripts/statistical_tests.py
    python scripts/statistical_tests.py --family improved --alpha 0.05
    python scripts/statistical_tests.py --criterion mean std best
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, studentized_range

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from build_results_table import (  # noqa: E402
    ALGO_DISPLAY,
    IMPROVED_ALGO_ORDER,
    STANDARD_ALGO_ORDER,
    _get_stats,
    load_results,
)

# Paper table order (c-instances, then T1–T15).
PAPER_DATASET_ORDER: list[str] = [
    "c201535",
    "c201555",
    "c201575",
    "c300695",
    "c300775",
    "c300855",
    "c302035",
    "c302055",
    "c302075",
    "c302095",
    "c351535",
    "c351555",
    "c351575",
    "c351595",
    *[f"T{i}" for i in range(1, 16)],
]

# Paper labels for improved-family algorithms (Experiment 2).
PAPER_ALGO_LABELS: dict[str, str] = {
    "ga": "GA",
    "gea_scenario_1": "GEA_1",
    "gea_scenario_2": "GEA_2",
    "gea_scenario_3": "GEA_3",
    "gea": "GEA",
    "improvedsa": "SA",
    "improvedparticleswarm": "PSO",
    "improvedhybridgasa": "GA-SA",
    "improvedhybridgapso": "GA-PSO",
    "adaptive": "ALGA",
    "adaptive_gea_scenario_1": "ALGEA_1",
    "adaptive_gea_scenario_2": "ALGEA_2",
    "adaptive_gea_scenario_3": "ALGEA_3",
    "adaptive_gea": "ALGEA",
}

CRITERIA: dict[str, tuple[str, str, bool]] = {
    # key: (block, field, lower_is_better)
    "mean": ("results", "mean", True),
    "std": ("results", "std", True),
    "best": ("results", "min", True),
    "worst": ("results", "max", True),
    "hitting_time": ("hitting_time", "mean", True),
    "nfe": ("nfe", "mean", True),
}

# Excel sheet names (capitalised, matching paper-style labels).
CRITERION_SHEET_NAMES: dict[str, str] = {
    "mean": "Mean",
    "std": "Std",
    "best": "Best",
    "worst": "Worst",
    "hitting_time": "Hitting_Time",
    "nfe": "NFE",
}


@dataclass(frozen=True)
class FriedmanNemenyiResult:
    criterion: str
    datasets: list[str]
    algorithms: list[str]
    labels: list[str]
    values: np.ndarray  # shape (n_datasets, n_algorithms)
    ranks: np.ndarray  # shape (n_datasets, n_algorithms)
    avg_ranks: np.ndarray  # shape (n_algorithms,)
    friedman_chi2: float
    friedman_p: float
    nemenyi_cd: float
    nemenyi_alpha: float
    significant_pairs: list[tuple[str, str, float]]  # (algo_a, algo_b, rank_diff)


def _display_label(stem: str, family: str) -> str:
    if family == "improved" and stem in PAPER_ALGO_LABELS:
        return PAPER_ALGO_LABELS[stem]
    return ALGO_DISPLAY.get(stem, stem)


def _extract_value(record: dict, block: str, field: str) -> float | None:
    stats = _get_stats(record, block) or {}
    value = stats.get(field)
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return None
    return float(value)


def build_performance_matrix(
    data: dict[str, dict[str, dict]],
    algorithms: list[str],
    datasets: list[str],
    block: str,
    field: str,
) -> tuple[list[str], np.ndarray]:
    """Return (used_datasets, matrix) with shape (n_datasets, n_algorithms)."""
    used_datasets: list[str] = []
    rows: list[list[float]] = []

    for dataset in datasets:
        row: list[float] = []
        complete = True
        for algo in algorithms:
            if algo not in data or dataset not in data[algo]:
                complete = False
                break
            value = _extract_value(data[algo][dataset], block, field)
            if value is None:
                complete = False
                break
            row.append(value)
        if complete:
            used_datasets.append(dataset)
            rows.append(row)

    if not rows:
        return [], np.empty((0, len(algorithms)))

    return used_datasets, np.asarray(rows, dtype=float)


def rank_per_dataset(values: np.ndarray, *, lower_is_better: bool) -> np.ndarray:
    """Average ranks per dataset row; rank 1 = best."""
    scores = values if lower_is_better else -values
    return np.apply_along_axis(lambda row: rankdata(row, method="average"), 1, scores)


def nemenyi_critical_difference(k: int, n: int, alpha: float) -> float:
    """Critical difference for Nemenyi post-hoc (Demsar 2006, infinite df)."""
    if k < 2 or n < 1:
        return float("inf")
    q_alpha = float(studentized_range.ppf(1.0 - alpha, k, np.inf))
    return q_alpha * math.sqrt(k * (k + 1) / (6.0 * n))


def friedman_nemenyi(
    *,
    criterion: str,
    block: str,
    field: str,
    lower_is_better: bool,
    data: dict[str, dict[str, dict]],
    algorithms: list[str],
    datasets: list[str],
    labels: list[str],
    alpha: float,
) -> FriedmanNemenyiResult | None:
    used_datasets, values = build_performance_matrix(data, algorithms, datasets, block, field)
    if values.size == 0:
        return None

    n_datasets, n_algos = values.shape
    if n_datasets < 2 or n_algos < 2:
        return None

    ranks = rank_per_dataset(values, lower_is_better=lower_is_better)
    avg_ranks = ranks.mean(axis=0)

    samples = [values[:, j] for j in range(n_algos)]
    chi2, p_value = friedmanchisquare(*samples)

    cd = nemenyi_critical_difference(n_algos, n_datasets, alpha)

    significant_pairs: list[tuple[str, str, float]] = []
    for i in range(n_algos):
        for j in range(i + 1, n_algos):
            diff = abs(avg_ranks[i] - avg_ranks[j])
            if diff > cd:
                significant_pairs.append((algorithms[i], algorithms[j], diff))

    return FriedmanNemenyiResult(
        criterion=criterion,
        datasets=used_datasets,
        algorithms=algorithms,
        labels=labels,
        values=values,
        ranks=ranks,
        avg_ranks=avg_ranks,
        friedman_chi2=float(chi2),
        friedman_p=float(p_value),
        nemenyi_cd=cd,
        nemenyi_alpha=alpha,
        significant_pairs=significant_pairs,
    )


def _rank_table_rows(result: FriedmanNemenyiResult) -> list[tuple[str, float, int]]:
    order = np.argsort(result.avg_ranks)
    rows: list[tuple[str, float, int]] = []
    for pos, idx in enumerate(order, start=1):
        rows.append((result.labels[idx], float(result.avg_ranks[idx]), pos))
    return rows


def write_csv_reports(output_dir: Path, results: list[FriedmanNemenyiResult]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    ranks_path = output_dir / "friedman_average_ranks.csv"
    with ranks_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["criterion", "rank", "algorithm", "avg_rank"])
        for result in results:
            for label, avg_rank, rank_pos in _rank_table_rows(result):
                writer.writerow([result.criterion, rank_pos, label, f"{avg_rank:.4f}"])

    summary_path = output_dir / "friedman_nemenyi_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["criterion", "n_datasets", "n_algorithms", "friedman_chi2", "friedman_p", "nemenyi_cd", "n_significant_pairs"]
        )
        for result in results:
            writer.writerow(
                [
                    result.criterion,
                    len(result.datasets),
                    len(result.algorithms),
                    f"{result.friedman_chi2:.6f}",
                    f"{result.friedman_p:.6e}",
                    f"{result.nemenyi_cd:.6f}",
                    len(result.significant_pairs),
                ]
            )

    pairs_path = output_dir / "nemenyi_significant_pairs.csv"
    with pairs_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["criterion", "algorithm_a", "algorithm_b", "rank_difference"])
        for result in results:
            label_by_stem = dict(zip(result.algorithms, result.labels, strict=True))
            for a, b, diff in result.significant_pairs:
                writer.writerow([result.criterion, label_by_stem[a], label_by_stem[b], f"{diff:.4f}"])


def write_html_report(output_dir: Path, results: list[FriedmanNemenyiResult], *, family: str, alpha: float) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"statistical_tests_{family}.html"

    sections = []
    for result in results:
        rank_rows = "".join(
            f"<tr><td>{pos}</td><td>{label}</td><td>{avg_rank:.4f}</td></tr>"
            for label, avg_rank, pos in _rank_table_rows(result)
        )

        sig_rows = ""
        label_by_stem = dict(zip(result.algorithms, result.labels, strict=True))
        if result.significant_pairs:
            sig_rows = "".join(
                f"<tr><td>{label_by_stem[a]}</td><td>{label_by_stem[b]}</td><td>{diff:.4f}</td></tr>"
                for a, b, diff in result.significant_pairs
            )
        else:
            sig_rows = '<tr><td colspan="3">No pairs exceed the critical difference.</td></tr>'

        sig_note = "significant" if result.friedman_p < alpha else "not significant"
        sections.append(
            f"""
<section>
  <h2>{result.criterion}</h2>
  <p>
    Datasets: {len(result.datasets)} &nbsp;|&nbsp;
    Friedman χ² = {result.friedman_chi2:.4f}, p = {result.friedman_p:.4e} ({sig_note} at α = {alpha}) &nbsp;|&nbsp;
    Nemenyi CD = {result.nemenyi_cd:.4f}
  </p>
  <h3>Average ranks (1 = best)</h3>
  <table>
    <thead><tr><th>Rank</th><th>Algorithm</th><th>Avg rank</th></tr></thead>
    <tbody>{rank_rows}</tbody>
  </table>
  <h3>Nemenyi significant pairs (|Δ rank| &gt; CD)</h3>
  <table>
    <thead><tr><th>Algorithm A</th><th>Algorithm B</th><th>|Δ rank|</th></tr></thead>
    <tbody>{sig_rows}</tbody>
  </table>
</section>
"""
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Statistical tests ({family})</title>
<style>
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; margin: 24px; color: #1a1a2e; }}
  h1 {{ font-size: 1.3em; }}
  h2 {{ margin-top: 28px; border-bottom: 1px solid #c8d0e0; padding-bottom: 4px; }}
  h3 {{ font-size: 1em; margin-top: 16px; }}
  p {{ color: #4a5568; }}
  table {{ border-collapse: collapse; margin: 8px 0 16px; }}
  th, td {{ border: 1px solid #c8d0e0; padding: 6px 12px; text-align: right; }}
  th:first-child, td:first-child {{ text-align: center; }}
  th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
  th {{ background: #f0f2f7; }}
</style>
</head>
<body>
<h1>Friedman &amp; Nemenyi tests ({family} family)</h1>
<p>Per-dataset ranks use the aggregated benchmark value for each criterion (lower = better).
Friedman tests differences in average ranks; Nemenyi post-hoc uses α = {alpha}.</p>
{"".join(sections)}
</body>
</html>
"""
    out.write_text(html)
    return out


def write_excel_report(
    output_path: Path,
    results: list[FriedmanNemenyiResult],
    *,
    family: str,
    alpha: float,
) -> Path:
    """Write Friedman/Nemenyi results to a multi-sheet Excel workbook."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_rows = []
        rank_frames: list[pd.DataFrame] = []
        pair_rows: list[dict[str, str | float]] = []

        for result in results:
            sheet = CRITERION_SHEET_NAMES.get(result.criterion, result.criterion)

            data_df = pd.DataFrame(result.values, index=result.datasets, columns=result.labels)
            data_df.index.name = "Dataset"
            data_df.to_excel(writer, sheet_name=sheet[:31])

            rank_rows = _rank_table_rows(result)
            ranks_df = pd.DataFrame(
                {"Algorithm": [label for label, _, _ in rank_rows], "Average Rank": [r for _, r, _ in rank_rows]},
            )
            ranks_df.to_excel(writer, sheet_name=f"{sheet[:28]}_Ranks", index=False)

            rank_frames.append(
                pd.DataFrame(
                    {
                        "Algorithm": result.labels,
                        result.criterion: result.avg_ranks,
                    }
                )
            )

            summary_rows.append(
                {
                    "Criterion": sheet,
                    "N datasets": len(result.datasets),
                    "N algorithms": len(result.algorithms),
                    "Friedman statistic": result.friedman_chi2,
                    "P-value": result.friedman_p,
                    "Significant (p < alpha)": result.friedman_p < alpha,
                    "Nemenyi CD": result.nemenyi_cd,
                    "N significant pairs": len(result.significant_pairs),
                    "Alpha": alpha,
                }
            )

            label_by_stem = dict(zip(result.algorithms, result.labels, strict=True))
            for a, b, diff in result.significant_pairs:
                pair_rows.append(
                    {
                        "Criterion": sheet,
                        "Algorithm A": label_by_stem[a],
                        "Algorithm B": label_by_stem[b],
                        "Rank difference": diff,
                    }
                )

        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Friedman_Summary", index=False)

        if rank_frames:
            avg_ranks_wide = rank_frames[0]
            for frame in rank_frames[1:]:
                avg_ranks_wide = avg_ranks_wide.merge(frame, on="Algorithm", how="outer")
            avg_ranks_wide = avg_ranks_wide.sort_values(
                by=results[0].criterion,
                ascending=True,
                na_position="last",
            )
            avg_ranks_wide.to_excel(writer, sheet_name="Average_Ranks", index=False)

        pairs_df = pd.DataFrame(pair_rows)
        if pairs_df.empty:
            pairs_df = pd.DataFrame(columns=["Criterion", "Algorithm A", "Algorithm B", "Rank difference"])
        pairs_df.to_excel(writer, sheet_name="Nemenyi_Pairs", index=False)

        info_df = pd.DataFrame(
            [
                {"Field": "Family", "Value": family},
                {"Field": "Alpha", "Value": alpha},
                {"Field": "Ranking", "Value": "Lower value = better (rank 1 = best)"},
                {"Field": "Friedman test", "Value": "scipy.stats.friedmanchisquare"},
                {"Field": "Post-hoc", "Value": "Nemenyi critical difference"},
            ]
        )
        info_df.to_excel(writer, sheet_name="Info", index=False)

    return output_path


def run_analysis(
    *,
    data: dict[str, dict[str, dict]],
    family: str,
    criteria: list[str],
    datasets: list[str],
    alpha: float,
    output_dir: Path,
) -> list[FriedmanNemenyiResult]:
    if family == "improved":
        algorithms = [a for a in IMPROVED_ALGO_ORDER if a in data]
    elif family == "standard":
        algorithms = [a for a in STANDARD_ALGO_ORDER if a in data]
    else:
        raise ValueError(f"Unknown family: {family}")

    labels = [_display_label(a, family) for a in algorithms]
    results: list[FriedmanNemenyiResult] = []

    for criterion in criteria:
        if criterion not in CRITERIA:
            raise ValueError(f"Unknown criterion {criterion!r}; choose from {list(CRITERIA)}")
        block, field, lower_is_better = CRITERIA[criterion]
        result = friedman_nemenyi(
            criterion=criterion,
            block=block,
            field=field,
            lower_is_better=lower_is_better,
            data=data,
            algorithms=algorithms,
            datasets=datasets,
            labels=labels,
            alpha=alpha,
        )
        if result is not None:
            results.append(result)
        else:
            print(f"  Warning: skipped {criterion} (insufficient data)")

    family_dir = output_dir / family
    write_csv_reports(family_dir, results)
    html_path = write_html_report(output_dir, results, family=family, alpha=alpha)
    xlsx_path = write_excel_report(
        family_dir / f"Friedman_Statistical_Test_{family}.xlsx",
        results,
        family=family,
        alpha=alpha,
    )
    print(f"  {family}: {len(results)} criteria → {html_path}, {xlsx_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=SCRIPT_DIR / "results",
        help="Directory with per-algorithm JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "results" / "statistical_tests",
        help="Directory for CSV/HTML reports",
    )
    parser.add_argument(
        "--family",
        choices=["improved", "standard", "both"],
        default="both",
        help="Algorithm family to analyse",
    )
    parser.add_argument(
        "--criterion",
        nargs="+",
        default=list(CRITERIA),
        choices=list(CRITERIA),
        help="Criteria to test (default: all)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for Nemenyi critical difference",
    )
    parser.add_argument(
        "--dataset-order",
        choices=["paper", "config"],
        default="paper",
        help="Dataset ordering (paper = c-instances then T1–T15)",
    )
    args = parser.parse_args()

    datasets = PAPER_DATASET_ORDER if args.dataset_order == "paper" else None
    if datasets is None:
        from build_results_table import load_dataset_order

        datasets = load_dataset_order()

    print(f"Loading results from {args.results_dir} ...")
    data = load_results(args.results_dir)
    if not data:
        print("No result JSON files found.")
        sys.exit(1)

    families = ["improved", "standard"] if args.family == "both" else [args.family]
    print(f"Running Friedman + Nemenyi (α = {args.alpha}) on {', '.join(families)} ...")

    for family in families:
        run_analysis(
            data=data,
            family=family,
            criteria=args.criterion,
            datasets=datasets,
            alpha=args.alpha,
            output_dir=args.output_dir,
        )

    print(f"Done. Reports in {args.output_dir}")


if __name__ == "__main__":
    main()
