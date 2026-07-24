from __future__ import annotations
from typing import Optional

import numpy as np


def coerce_rng(random_state: Optional[int] = None, rng: Optional[object] = None) -> np.random.Generator:
    """Return a `numpy.random.Generator` coerced from common inputs.

    Accepts: None, int, `np.random.Generator`, `np.random.RandomState`.
    `rng` takes precedence over `random_state` when both are given, so a
    shared Generator can be threaded through a pipeline while individual
    calls still accept a plain seed as a fallback.
    """
    if rng is not None:
        if isinstance(rng, np.random.Generator):
            return rng
        if isinstance(rng, np.random.RandomState):
            return np.random.default_rng(rng.randint(0, 2 ** 32 - 1))
        if isinstance(rng, (int, np.integer)):
            return np.random.default_rng(int(rng))

    if random_state is None:
        return np.random.default_rng()
    return np.random.default_rng(int(random_state))


__all__ = ["coerce_rng"]
