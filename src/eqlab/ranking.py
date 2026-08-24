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
    def rank(self, items):
        out=[]
        for c,r in items:
            score=c.score + 2*r.walk_forward_mean + r.parameter_stability + r.monte_carlo_p05 - 2*r.overfit_penalty
            if not r.passed: score -= 10
            out.append(RankedStrategy(c,r,score))
        return sorted(out,key=lambda x:x.final_score,reverse=True)
