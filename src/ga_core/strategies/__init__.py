"""Genetic algorithm strategies module.

Provides encoding-agnostic interfaces for selection, reproduction,
fitness evaluation, and elitism strategies. Supports multiple encodings
(currently binary, extensible to permutation, real-valued, etc.).

Public API:
    Protocols (for type checking):
        - SelectionFn
        - ReproductionExecutor
        - FitnessFn
        - PairingFn
        - ElitismFn

    Factory functions (encoding-agnostic):
        - create_selection_fn
        - create_reproduction_executor
        - create_fitness_fn
        - create_pairing_fn
        - create_elitism_fn

    Direct strategy functions (encoding-agnostic):
        - roulette_selection
        - tournament_selection
        - linear_rank_selection
        - random_pairing
        - sequential_pairing
        - select_elites
"""

from src.ga_core.strategies.elitism import select_elites
from src.ga_core.strategies.factory import (
    create_elitism_fn,
    create_fitness_fn,
    create_pairing_fn,
    create_reproduction_executor,
    create_selection_fn,
)
from src.ga_core.strategies.parent_pairing import (
    random_pairing,
    sequential_pairing,
)
from src.ga_core.strategies.protocols import (
    ElitismFn,
    FitnessFn,
    PairingFn,
    ReproductionExecutor,
    SelectionFn,
)
from src.ga_core.strategies.selection_methods import (
    linear_rank_selection,
    roulette_selection,
    tournament_selection,
)

__all__ = [
    # Protocols
    "SelectionFn",
    "ReproductionExecutor",
    "FitnessFn",
    "PairingFn",
    "ElitismFn",
    # Factory functions
    "create_selection_fn",
    "create_reproduction_executor",
    "create_fitness_fn",
    "create_pairing_fn",
    "create_elitism_fn",
    # Direct strategy functions
    "roulette_selection",
    "tournament_selection",
    "linear_rank_selection",
    "random_pairing",
    "sequential_pairing",
    "select_elites",
]
