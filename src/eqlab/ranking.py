from __future__ import annotations

from dataclasses import dataclass

from .evolution import Candidate
from .validation import RobustnessReport


@dataclass(frozen=True, slots=True)
class RankedStrategy:
    candidate: Candidate
    robustness: RobustnessReport
    final_score: float


class RankingEngine:
    """Rank on robustness and risk-adjusted evidence, not return alone."""

    def rank(self, items) -> list[RankedStrategy]:
        out: list[RankedStrategy] = []
        for candidate, robustness in items:
            final_score = (
                candidate.score
                + 2.0 * robustness.walk_forward_mean
                - robustness.walk_forward_std
                + robustness.parameter_stability
                + robustness.regime_consistency
                + robustness.monte_carlo_p05
                - 2.0 * robustness.overfit_penalty
                - robustness.complexity_penalty
            )
            if not robustness.passed:
                final_score -= 10.0
            out.append(RankedStrategy(candidate, robustness, final_score))
        return sorted(out, key=lambda x: x.final_score, reverse=True)
