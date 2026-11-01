import numpy as np

from scripts.statistical_tests import (
    friedman_nemenyi,
    nemenyi_critical_difference,
    rank_per_dataset,
)


def test_rank_per_dataset_lower_is_better():
    values = np.array([[10.0, 20.0, 30.0], [5.0, 15.0, 25.0]])
    ranks = rank_per_dataset(values, lower_is_better=True)
    assert ranks[0, 0] == 1.0
    assert ranks[0, 2] == 3.0
    assert ranks[1, 0] == 1.0


def test_nemenyi_cd_increases_with_k():
    cd_small = nemenyi_critical_difference(5, 10, 0.05)
    cd_large = nemenyi_critical_difference(10, 10, 0.05)
    assert cd_large > cd_small


def test_friedman_detects_clear_difference():
    algorithms = ["a", "b", "c"]
    datasets = ["d1", "d2", "d3"]
    data = {
        "a": {
            "d1": {"results": {"X": {"mean": 1.0}}},
            "d2": {"results": {"X": {"mean": 1.0}}},
            "d3": {"results": {"X": {"mean": 1.0}}},
        },
        "b": {
            "d1": {"results": {"X": {"mean": 2.0}}},
            "d2": {"results": {"X": {"mean": 2.0}}},
            "d3": {"results": {"X": {"mean": 2.0}}},
        },
        "c": {
            "d1": {"results": {"X": {"mean": 3.0}}},
            "d2": {"results": {"X": {"mean": 3.0}}},
            "d3": {"results": {"X": {"mean": 3.0}}},
        },
    }
    result = friedman_nemenyi(
        criterion="mean",
        block="results",
        field="mean",
        lower_is_better=True,
        data=data,
        algorithms=algorithms,
        datasets=datasets,
        labels=algorithms,
        alpha=0.05,
    )
    assert result is not None
    assert result.friedman_p < 0.05
    assert result.avg_ranks[0] < result.avg_ranks[1] < result.avg_ranks[2]
