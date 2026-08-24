from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class OverfitAssessment:
    performance_gap: float
    complexity_penalty: float
    multiple_testing_penalty: float
    score: float
    flagged: bool


class OverfittingDetector:
    """Development-only overfitting detector.

    It never consumes OOS observations. Multiple-testing pressure increases with
    the number of hypotheses searched, making large evolutionary searches pay a
    statistical penalty even when the best observed return is high.
    """

    def assess(
        self,
        train_return: float,
        validation_return: float,
        complexity: float,
        trials: int = 1,
    ) -> OverfitAssessment:
        if trials < 1:
            raise ValueError("trials must be positive")
        gap = abs(train_return - validation_return)
        complexity_penalty = max(0.0, (complexity - 2.0) * 0.02)
        multiple_testing_penalty = min(0.50, 0.02 * math.log10(max(trials, 1)))
        score = min(1.0, gap + complexity_penalty + multiple_testing_penalty)
        return OverfitAssessment(
            gap, complexity_penalty, multiple_testing_penalty, score, score >= 0.50
        )
