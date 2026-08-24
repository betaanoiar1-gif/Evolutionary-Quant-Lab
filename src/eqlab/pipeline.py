from __future__ import annotations

from dataclasses import dataclass
from .backtest import BacktestEngine, PerformanceReport
from .data import DataEngine
from .dna import StrategyDNA
from .evolution import Candidate, StrategyGenerator, score
from .fast import FastBacktestEngine
from .ranking import RankingEngine
from .validation import OOSGate, RobustnessEngine, RobustnessReport

@dataclass(frozen=True, slots=True)
class ResearchResult:
    candidates: tuple[Candidate, ...]
    robust: tuple[tuple[Candidate, RobustnessReport], ...]
    final: tuple[Candidate, PerformanceReport] | None

class Laboratory:
    """End-to-end research pipeline. OOS is accessed only by final_evaluate."""
    def __init__(self, seed=42, backtester=None):
        self.data=DataEngine(); self.fast=FastBacktestEngine(); self.bt=backtester or BacktestEngine(); self.robust=RobustnessEngine(self.bt,seed); self.gen=StrategyGenerator(seed); self.rank=RankingEngine(); self.oos=OOSGate(self.bt)
    def screen(self, train, population, min_return=-.25):
        survivors=[]
        for d in population:
            r=self.fast.run(train,d)
            if r.trades>=5 and r.total_return>=min_return and r.max_drawdown<.8: survivors.append(d)
        return survivors
    def full_rank(self, train, population):
        out=[]
        for d in population:
            r=self.bt.run(train,d); out.append(Candidate(d,r,score(r)))
        return sorted(out,key=lambda x:x.score,reverse=True)
    def robustness(self, train, validation, candidates, top_n=20):
        return [(c,self.robust.evaluate(train,validation,c.dna)) for c in candidates[:top_n]]
    def final_evaluate(self, oos, dna):
        return self.oos.evaluate(oos,dna)
