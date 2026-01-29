"""Encoding-specific function for binary mutation."""

import numpy as np
from numpy.typing import NDArray

gen_arr = NDArray[np.uint8]


def bit_flip_mutation(
    children: gen_arr, mutation_probability: float, rng: np.random.Generator
) -> gen_arr:
    """Apply bit-flip mutation to children population.

    Randomly flips bits (0→1 or 1→0) in the children array based on mutation
    probability. Each gene has an independent chance of being mutated.

    Args:
        children: Binary array to mutate (shape: [pop_size, genome_length]).
        mutation_probability: Probability of mutation occurring for a single gene.
        rng: NumPy random generator for reproducible mutation.

    Returns:
        Mutated children array (modified in-place).
    """
    mask = rng.random(size=children.shape) < mutation_probability
    children[mask] ^= 1
    return children
