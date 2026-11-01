import numpy as np
import pandas as pd

from scripts.wilcoxon_tests import (
    PairwiseRecord,
    build_summary_counts,
    build_wilcoxon_per_dataset,
    load_per_run_matrix,
    paired_wilcoxon_compare,
)


def _make_record(per_run):
    return {"results": {"X": {"mean": float(np.mean(per_run)), "per_run": per_run}}}


def test_paired_wilcoxon_focal_wins():
    focal = [1.0, 2.0, 3.0, 4.0, 5.0] * 6
    competitor = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0] * 5
    sign, p = paired_wilcoxon_compare(focal, competitor, alpha=0.05)
    assert p < 0.05
    assert sign == "+"


def test_paired_wilcoxon_tie_on_identical():
    runs = [5.0, 5.1, 4.9, 5.0, 5.2, 4.8] * 5
    sign, p = paired_wilcoxon_compare(runs, runs, alpha=0.05)
    assert sign == "="
    assert p == 1.0


def test_build_wilcoxon_per_dataset_layout():
    data = {
        "focal": {
            "d1": _make_record([1, 2, 3, 4, 5] * 6),
        },
        "other": {
            "d1": _make_record([10, 11, 12, 13, 14, 15] * 5),
        },
    }
    used, matrix = load_per_run_matrix(data, ["focal", "other"], ["d1"])
    df = build_wilcoxon_per_dataset(
        matrix,
        used,
        "focal",
        "OURS",
        ["other"],
        ["OTHER"],
        alpha=0.05,
    )
    assert list(df.columns) == ["Function", "OTHER_p", "OTHER_result"]
    assert df.loc[0, "Function"] == "d1"
    assert df.loc[0, "OTHER_result"] == "+"


def test_summary_counts():
    df = pd.DataFrame(
        {
            "Function": ["d1", "d2", "d3"],
            "GA_result": ["+", "=", "-"],
        }
    )
    summary = build_summary_counts(df, ["ga"], ["GA"])
    assert summary.loc[0, "Record"] == PairwiseRecord(1, 1, 1).formatted()
