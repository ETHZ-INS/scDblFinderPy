"""Pure-Python MT19937 implementation (reference implementation port).

This provides `MT19937` with `seed(seed)` and `random()` returning floats
in [0,1). It's intended to match R's Mersenne-Twister internals where
possible (state layout, initialization and `unif_rand()` fixup).
"""
from __future__ import annotations
from typing import List


class MT19937:
    def __init__(self, seed: int = 5489):
        self.N = 624
        self.M = 397
        self.MATRIX_A = 0x9908b0df
        self.UPPER_MASK = 0x80000000
        self.LOWER_MASK = 0x7fffffff
        self.mt: List[int] = [0] * self.N
        self.mti = self.N + 1
        # allow seed to be an int or an iterable of ints (for init_by_array)
        if isinstance(seed, (list, tuple)):
            self.seed_by_array(seed)
        else:
            self.seed(seed)

    def MT_sgenrand(self, seed: int):
        """R's MT_sgenrand initialization: sets mt[] using two LCG steps per element."""
        seed = int(seed) & 0xffffffff
        for i in range(self.N):
            self.mt[i] = seed & 0xffff0000
            seed = (69069 * seed + 1) & 0xffffffff
            self.mt[i] |= (seed & 0xffff0000) >> 16
            seed = (69069 * seed + 1) & 0xffffffff
        self.mti = self.N

    def seed_by_array(self, init_key):
        # Port of mt19937 init_by_array from the reference implementation.
        key_length = len(init_key)
        # initialize with a fixed value
        self.seed(19650218)
        i = 1
        j = 0
        k = self.N if self.N > key_length else key_length
        for _ in range(k):
            prev = self.mt[i - 1]
            self.mt[i] = (self.mt[i] ^ ((prev ^ (prev >> 30)) * 1664525)) + int(init_key[j]) + j
            self.mt[i] &= 0xFFFFFFFF
            i += 1
            j += 1
            if i >= self.N:
                self.mt[0] = self.mt[self.N - 1]
                i = 1
            if j >= key_length:
                j = 0
        for _ in range(self.N - 1):
            prev = self.mt[i - 1]
            self.mt[i] = (self.mt[i] ^ ((prev ^ (prev >> 30)) * 1566083941)) - i
            self.mt[i] &= 0xFFFFFFFF
            i += 1
            if i >= self.N:
                self.mt[0] = self.mt[self.N - 1]
                i = 1
        self.mt[0] = 0x80000000

    def do_setseed(self, seed_vec):
        """Install either an R `.Random.seed` vector or treat `seed_vec` as
        an init_key/array for `seed_by_array()`.
        """
        if isinstance(seed_vec, (list, tuple)) and len(seed_vec) >= 2 + self.N:
            self.set_state_from_random_seed(seed_vec)
        else:
            self.seed_by_array([int(x) & 0xFFFFFFFF for x in seed_vec])

    def seed(self, s: int):
        # Use R's MT_sgenrand for compatibility
        self.MT_sgenrand(int(s))

    def extract_number(self) -> int:
        if self.mti >= self.N:
            self.twist()

        y = self.mt[self.mti]
        y ^= (y >> 11)
        y ^= (y << 7) & 0x9d2c5680
        y ^= (y << 15) & 0xefc60000
        y ^= (y >> 18)

        self.mti += 1
        return y & 0xFFFFFFFF

    def twist(self):
        for i in range(self.N):
            y = (self.mt[i] & self.UPPER_MASK) | (self.mt[(i + 1) % self.N] & self.LOWER_MASK)
            self.mt[i] = self.mt[(i + self.M) % self.N] ^ (y >> 1)
            if y & 0x1:
                self.mt[i] ^= self.MATRIX_A
        self.mti = 0

    def randint32(self) -> int:
        return self.extract_number()

    def randint53(self) -> int:
        a = self.extract_number() >> 5
        b = self.extract_number() >> 6
        return ((a << 26) | b) & ((1 << 53) - 1)

    def random(self) -> float:
        # genrand_real2 + R's fixup to avoid exact 0 or 1
        y = int(self.randint32()) & 0xFFFFFFFF
        x = y / 4294967296.0
        i2_32m1 = 1.0 / 4294967295.0
        if x <= 0.0:
            return 0.5 * i2_32m1
        if (1.0 - x) <= 0.0:
            return 1.0 - 0.5 * i2_32m1
        return x

    def unif_rand(self) -> float:
        return self.random()

    def ru(self) -> float:
        # (floor(U*unif_rand()) + unif_rand())/U
        U = 33554432.0
        u1 = self.unif_rand()
        u2 = self.unif_rand()
        return (float(int(u1 * U)) + u2) / U

    def set_state_from_random_seed(self, seed_vec):
        if len(seed_vec) < 2 + self.N:
            raise ValueError("seed vector too short for MT19937 state")
        mti = int(seed_vec[1])
        mt_vals = [int(x) & 0xFFFFFFFF for x in seed_vec[2:2 + self.N]]
        if len(mt_vals) != self.N:
            raise ValueError("unexpected MT state length in seed vector")
        self.mt = mt_vals[:]
        # apply FixupSeeds-like behavior
        self.mti = int(mti)
        if self.mti <= 0:
            self.mti = self.N
        if all((int(x) & 0xFFFFFFFF) == 0 for x in self.mt):
            # fallback default as in R
            self.MT_sgenrand(4357)

    def peek_ints(self, count: int):
        saved_mt = self.mt[:]
        saved_mti = self.mti
        vals = []
        try:
            for _ in range(count):
                vals.append(self.randint32())
        finally:
            self.mt = saved_mt
            self.mti = saved_mti
        return vals


__all__ = ["MT19937"]
