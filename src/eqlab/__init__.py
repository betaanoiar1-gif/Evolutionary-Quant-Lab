"""Evolutionary Quant Strategy Laboratory."""

from .dna import StrategyDNA
from .selection import SelectionCandidate, SelectionConfig, SelectionEngine, SelectionReport

__version__ = "0.1.0"

__all__ = [
    "StrategyDNA",
    "SelectionCandidate",
    "SelectionConfig",
    "SelectionEngine",
    "SelectionReport",
]
