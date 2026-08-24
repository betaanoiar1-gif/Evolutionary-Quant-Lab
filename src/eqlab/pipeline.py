from __future__ import annotations

from dataclasses import dataclass

from .archive import StrategyArchive
from .backtest import BacktestEngine, PerformanceReport
from .data import DataEngine
from .evolution import Candidate, SearchEngine, StrategyGenerator, score
from .fast import FastBacktestEngine
from .ranking import RankedStrategy, RankingEngine
from .validation import OOSGate, RobustnessEngine


@dataclass(frozen=True, slots=True)
class ResearchResult:
    candidates: tuple[Candidate, ...]
    robust: tuple[RankedStrategy, ...]
    final: tuple[Candidate, PerformanceReport] | None


class Laboratory:
    """Development pipeline with a one-way final OOS evaluation boundary."""

    def __init__(self, seed: int = 42, backtester: BacktestEngine | None = None,
                 archive: StrategyArchive | None = None) -> None:
        self.data = DataEngine()
        self.fast = FastBacktestEngine()
        self.bt = backtester or BacktestEngine()
        self.robust = RobustnessEngine(self.bt, seed)
        self.gen = StrategyGenerator(seed)
        self.search = SearchEngine(self.bt, seed)
        self.rank = RankingEngine()
        self.archive = archive or StrategyArchive()
        self.oos = OOSGate(self.bt)

    def screen(self, train, population, min_return: float = -.25, max_drawdown: float = .8):
        survivors = []
        for dna in population:
            report = self.fast.run(train, dna)
            if report.trades >= 5 and report.total_return >= min_return and report.max_drawdown < max_drawdown:
                survivors.append(dna)
        return survivors

    def full_rank(self, train, population) -> list[Candidate]:
        candidates = []
        for dna in population:
            report = self.bt.run(train, dna)
            candidates.append(Candidate(dna, report, score(report)))
        return sorted(candidates, key=lambda item: item.score, reverse=True)

    def evolve(self, train, population, generations: int = 5, elite: int = 10):
        evolved = self.search.evolve(train, population, generations, elite)
        self.archive.save(evolved)
        return evolved

    def robustness(self, train, validation, candidates, top_n: int = 20):
        return [(candidate, self.robust.evaluate(train, validation, candidate.dna)) for candidate in candidates[:top_n]]

    def robust_rank(self, train, validation, candidates, top_n: int = 20) -> list[RankedStrategy]:
        ranked = self.rank.rank(self.robustness(train, validation, candidates, top_n))
        self.archive.save([item.candidate for item in ranked if item.robustness.passed])
        return ranked

    def final_evaluate(self, oos, dna) -> PerformanceReport:
        return self.oos.evaluate(oos, dna)
