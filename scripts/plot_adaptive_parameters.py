#!/usr/bin/env python3
"""
Plot adaptive λ trajectories for ALGA / ALGEA variants.

Layout (A4, 10 subplots per page except the last):
  - 2 columns = 2 benchmark instances (paper table order)
  - 5 rows    = ALGA, ALGEA_1, ALGEA_2, ALGEA_3, ALGEA (top → bottom)

29 instances → 14 full pages + 1 page with a single instance (5 subplots).

Requires lambda_history in adaptive JSON results (re-run adaptive configs after
logging update).

Usage:
    python scripts/plot_adaptive_parameters.py
    python scripts/plot_adaptive_parameters.py --main-text T1 T2 c201535 c300695
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from build_results_table import _get_stats, load_results  # noqa: E402
from statistical_tests import PAPER_DATASET_ORDER  # noqa: E402

# Row order (top → bottom).
ADAPTIVE_ALGOS: list[tuple[str, str]] = [
    ("adaptive", "ALGA"),
    ("adaptive_gea_scenario_1", "ALGEA_1"),
    ("adaptive_gea_scenario_2", "ALGEA_2"),
    ("adaptive_gea_scenario_3", "ALGEA_3"),
    ("adaptive_gea", "ALGEA"),
]

LAMBDA_PLOT_STYLE: dict[str, tuple[str, str]] = {
    "lambda_crossover": ("λ_cross", "#1f77b4"),
    "lambda_mutation": ("λ_mut", "#ff7f0e"),
    "lambda_rc": ("λ_RC", "#2ca02c"),
    "lambda_dm": ("λ_DM", "#d62728"),
    "lambda_gi": ("λ_GI", "#9467bd"),
}

A4_SIZE = (8.27, 11.69)  # inches
ROWS, COLS = 5, 2


def _lambda_trajectory(record: dict) -> list[dict] | None:
    block = record.get("lambda_history", {})
    if not block:
        return None
    for payload in block.values():
        mean_traj = payload.get("mean")
        if mean_traj:
            return mean_traj
        per_run = payload.get("per_run") or []
        if per_run:
            return per_run[0]
    return None


def load_lambda_data(
    data: dict[str, dict[str, dict]],
    datasets: list[str],
) -> tuple[list[str], dict[str, dict[str, list[dict]]]]:
    """Return (used_datasets, {dataset: {algo_stem: trajectory}})."""
    used: list[str] = []
    out: dict[str, dict[str, list[dict]]] = {}

    for dataset in datasets:
        row: dict[str, list[dict]] = {}
        complete = True
        for stem, _ in ADAPTIVE_ALGOS:
            if stem not in data or dataset not in data[stem]:
                complete = False
                break
            traj = _lambda_trajectory(data[stem][dataset])
            if not traj:
                complete = False
                break
            row[stem] = traj
        if complete:
            used.append(dataset)
            out[dataset] = row

    return used, out


def _plot_trajectory(ax: plt.Axes, trajectory: list[dict], *, title: str) -> None:
    if not trajectory:
        ax.set_visible(False)
        return

    iterations = np.arange(1, len(trajectory) + 1)
    keys = trajectory[0].keys()
    for key in keys:
        if key not in LAMBDA_PLOT_STYLE:
            continue
        label, color = LAMBDA_PLOT_STYLE[key]
        values = [float(step[key]) for step in trajectory]
        ax.plot(iterations, values, label=label, color=color, linewidth=1.2)

    ax.set_title(title, fontsize=9)
    ax.set_xlim(1, len(trajectory))
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.tick_params(labelsize=7)
    if len(keys) <= 3:
        ax.legend(fontsize=6, loc="best", framealpha=0.8)


def _page_dataset_pairs(datasets: list[str]) -> list[list[str]]:
    """Chunk datasets into pages of 1–2 instances."""
    pages: list[list[str]] = []
    i = 0
    while i < len(datasets):
        if i + 1 < len(datasets):
            pages.append([datasets[i], datasets[i + 1]])
            i += 2
        else:
            pages.append([datasets[i]])
            i += 1
    return pages


def render_page(
    page_datasets: list[str],
    lambda_data: dict[str, dict[str, list[dict]]],
    *,
    page_num: int,
    total_pages: int,
) -> plt.Figure:
    n_cols = min(COLS, len(page_datasets))
    fig, axes = plt.subplots(ROWS, COLS, figsize=A4_SIZE, squeeze=False)
    fig.suptitle(
        f"Adaptive parameters — page {page_num}/{total_pages}",
        fontsize=11,
        y=0.995,
    )

    for row_idx, (stem, algo_label) in enumerate(ADAPTIVE_ALGOS):
        for col_idx in range(COLS):
            ax = axes[row_idx, col_idx]
            if col_idx >= n_cols:
                ax.set_visible(False)
                continue
            dataset = page_datasets[col_idx]
            trajectory = lambda_data[dataset][stem]
            _plot_trajectory(
                ax,
                trajectory,
                title=f"{algo_label} — {dataset}",
            )
            if row_idx == ROWS - 1:
                ax.set_xlabel("Iteration", fontsize=7)
            if col_idx == 0:
                ax.set_ylabel("λ", fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig


def write_pages(
    datasets: list[str],
    lambda_data: dict[str, dict[str, list[dict]]],
    output_dir: Path,
    *,
    prefix: str,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pages = _page_dataset_pairs(datasets)
    paths: list[Path] = []
    for page_idx, page_datasets in enumerate(pages, start=1):
        fig = render_page(page_datasets, lambda_data, page_num=page_idx, total_pages=len(pages))
        path = output_dir / f"{prefix}_page_{page_idx:02d}.pdf"
        fig.savefig(path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def write_manifest(
    path: Path,
    *,
    all_datasets: list[str],
    small_instances: list[str],
    large_instances: list[str],
) -> None:
    lines = [
        "# Adaptive λ plot manifest",
        "",
        "## All supplementary pages",
        f"Directory: {path.parent / 'supplementary'}",
        f"Instances ({len(all_datasets)}): " + ", ".join(all_datasets),
        "",
        "## Suggested main-text candidates",
        f"Small instances (T*): {', '.join(small_instances)}",
        f"Large instances (c*): {', '.join(large_instances)}",
        "",
        "Pick 2 small + 2 large for the main text (20 subplots = 2 A4 pages):",
        "  python scripts/plot_adaptive_parameters.py --main-text <d1> <d2> <d3> <d4>",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=SCRIPT_DIR / "results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "results" / "adaptive_plots",
    )
    parser.add_argument(
        "--main-text",
        nargs="+",
        default=None,
        metavar="DATASET",
        help="Four datasets for main-text figures (2 pages, 20 subplots)",
    )
    args = parser.parse_args()

    print(f"Loading adaptive results from {args.results_dir} ...")
    data = load_results(args.results_dir)
    used, lambda_data = load_lambda_data(data, PAPER_DATASET_ORDER)
    if not used:
        print(
            "No lambda_history found in adaptive JSON files.\n"
            "Re-run the 5 improved adaptive configs:\n"
            "  adaptive, adaptive_gea_scenario_1/2/3, adaptive_gea"
        )
        sys.exit(1)

    if len(used) < len(PAPER_DATASET_ORDER):
        print(f"Warning: only {len(used)}/{len(PAPER_DATASET_ORDER)} datasets have λ history")

    supp_dir = args.output_dir / "supplementary"
    paths = write_pages(used, lambda_data, supp_dir, prefix="adaptive_lambda")
    print(f"Wrote {len(paths)} supplementary pages to {supp_dir}")

    small = [d for d in used if d.startswith("T")]
    large = [d for d in used if d.startswith("c")]
    write_manifest(args.output_dir / "README.txt", all_datasets=used, small_instances=small, large_instances=large)

    if args.main_text:
        if len(args.main_text) != 4:
            print("Error: --main-text requires exactly 4 datasets (2 small + 2 large suggested).")
            sys.exit(1)
        missing = [d for d in args.main_text if d not in lambda_data]
        if missing:
            print(f"Error: no λ data for: {', '.join(missing)}")
            sys.exit(1)
        main_dir = args.output_dir / "main_text"
        main_paths = write_pages(args.main_text, lambda_data, main_dir, prefix="main_text")
        print(f"Wrote {len(main_paths)} main-text pages to {main_dir}")
    else:
        print("Tip: pass --main-text T1 T2 c201535 c300695 for main-text pages (2 A4 sheets).")

    print(f"Done. Output in {args.output_dir}")


if __name__ == "__main__":
    main()
