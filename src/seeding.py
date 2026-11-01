import numpy as np
from numba import njit


@njit(cache=True)
def _seed_numba_rng(seed: int) -> None:
    np.random.seed(seed)


def seed_all(seed: int) -> None:
    """Seed both NumPy's global RNG and numba's (separate) RNG used inside @njit functions.

    numba-jitted code (mutations, repair) keeps its own RNG state that np.random.seed()
    does not touch, so plain np.random.seed() alone does not make runs reproducible.
    """
    np.random.seed(seed)
    _seed_numba_rng(seed)
