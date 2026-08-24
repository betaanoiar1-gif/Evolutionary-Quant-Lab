from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class OverfitAssessment:
    performance_gap: float
    complexity_penalty: float
    score: float
    flagged: bool

class OverfittingDetector:
    """Conservative leakage/fragility detector based only on development data."""
    def assess(self, train_return: float, validation_return: float, complexity: float) -> OverfitAssessment:
        gap=abs(train_return-validation_return)
        complexity_penalty=max(0.0,(complexity-2.0)*0.02)
        score=min(1.0,gap+complexity_penalty)
        return OverfitAssessment(gap,complexity_penalty,score,score>=0.50)
