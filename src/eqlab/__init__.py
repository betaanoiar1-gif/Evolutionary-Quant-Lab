"""Evolutionary Quant Strategy Laboratory."""

from .dna import StrategyDNA
from .pipeline import Laboratory, ResearchResult
from .selection import SelectionCandidate, SelectionConfig, SelectionEngine, SelectionReport

__version__ = "0.1.0"

__all__ = [
    "StrategyDNA",
    "Laboratory",
    "ResearchResult",
    "SelectionCandidate",
    "SelectionConfig",
    "SelectionEngine",
    "SelectionReport",
]
