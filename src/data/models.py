from dataclasses import dataclass

import numpy as np


# A GQAP problem instance.
@dataclass(frozen=True)
class Model:
    I: int
    J: int

    cij: np.ndarray  # (I, J)
    aij: np.ndarray  # (I, J)
    bi: np.ndarray  # (I,)
    DIS: np.ndarray  # (I, I)
    F: np.ndarray  # (J, J)

    def __post_init__(self) -> None:
        expected_shapes = {
            "cij": (self.I, self.J),
            "aij": (self.I, self.J),
            "bi": (self.I,),
            "DIS": (self.I, self.I),
            "F": (self.J, self.J),
        }

        actual_arrays = {
            "cij": self.cij,
            "aij": self.aij,
            "bi": self.bi,
            "DIS": self.DIS,
            "F": self.F,
        }

        for name, expected_shape in expected_shapes.items():
            if actual_arrays[name].shape != expected_shape:
                raise ValueError(f"{name} shape mismatch: expected {expected_shape}, got {actual_arrays[name].shape}")


# A GA candidate solution.
@dataclass
class Individual:
    permutation: np.ndarray  # (J,)
    cost: float
    cvar: np.ndarray  # (I,)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Individual):
            return False
        # Two individuals are equal if they have the same permutation
        return np.array_equal(self.permutation, other.permutation)

    def __hash__(self) -> int:
        # .tobytes() avoids boxing every element into a Python int the way tuple() would;
        # normalize dtype first so equal-valued arrays of different dtypes (which
        # np.array_equal in __eq__ treats as equal) still hash the same.
        return hash(self.permutation.astype(np.int64, copy=False).tobytes())

    def __repr__(self) -> str:
        return f"Individual(cost={self.cost:.4f})"
