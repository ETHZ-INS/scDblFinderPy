from __future__ import annotations
from typing import Optional

import numpy as np

from .r_mt19937 import MT19937
# Avoid importing sample_int_r at module import time to prevent circular imports;
# import it lazily inside methods that need it.


class CentralRNG:
    """Central RNG that owns an R-faithful MT19937 and a derived numpy Generator.

    The MT instance is used for R-ported routines (sample_int parity via
    `randint32`/`unif_rand`). A single derived `np.random.Generator` is
    deterministically seeded from the MT state and used for NumPy-style
    draws (`poisson`, `integers`, `permutation`). This keeps all randomness
    rooted in the same reproducible sequence.
    """
    def __init__(self, seed: Optional[int] = None):
        if isinstance(seed, MT19937):
            self.mt = seed
        else:
            if seed is None:
                seed = 5489
            # allow list/iterable seeds through MT19937 constructor
            self.mt = MT19937(seed)

        # derive a numpy Generator seed deterministically without advancing the
        # live MT state when we were given an already-seeded MT instance.
        if isinstance(seed, MT19937):
            hi, lo = self.mt.peek_ints(2)
        else:
            hi = int(self.mt.randint32()) & 0xFFFFFFFF
            lo = int(self.mt.randint32()) & 0xFFFFFFFF
        seed64 = (hi << 32) | lo
        # numpy accepts Python int seeds; default_rng will digest it
        self._gen = np.random.default_rng(int(seed64 & ((1 << 63) - 1)))

    # MT-like interface (used by sample_int_r)
    def randint32(self) -> int:
        return int(self.mt.randint32())

    def unif_rand(self) -> float:
        return float(self.mt.unif_rand())

    def random(self) -> float:
        return float(self.mt.random())

    # NumPy-style interface (delegates to derived Generator)
    def integers(self, *args, **kwargs):
        return self._gen.integers(*args, **kwargs)

    def permutation(self, *args, **kwargs):
        return self._gen.permutation(*args, **kwargs)

    def poisson(self, *args, **kwargs):
        return self._gen.poisson(*args, **kwargs)

    def sample_int(self, n: int, size: int, replace: bool = False):
        # delegate lazily to avoid circular import at module import time
        from .r_sample_emulation import sample_int_r
        return sample_int_r(n, size, replace=replace, rng=self)


def coerce_rng(random_state: Optional[int] = None, rng: Optional[object] = None) -> CentralRNG:
    """Return a CentralRNG instance coerced from common inputs.

    Accepts: None, int, CentralRNG, np.random.Generator, np.random.RandomState.
    """
    if rng is not None:
        if isinstance(rng, CentralRNG):
            return rng
        # If a numpy Generator is supplied, extract a seed deterministically
        if isinstance(rng, np.random.Generator):
            try:
                s = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
            except Exception:
                s = 5489
            return CentralRNG(s)
        # RandomState: take a 32-bit int
        if isinstance(rng, np.random.mtrand.RandomState):
            try:
                s = int(rng.randint(0, 2**32 - 1))
            except Exception:
                s = 5489
            return CentralRNG(s)
        # If raw MT19937-like object provided
        if hasattr(rng, 'randint32') and hasattr(rng, 'unif_rand'):
            return CentralRNG(rng)

    # Fallback: use provided random_state integer
    if random_state is None:
        return CentralRNG(None)
    return CentralRNG(int(random_state))


__all__ = ["CentralRNG", "coerce_rng"]
