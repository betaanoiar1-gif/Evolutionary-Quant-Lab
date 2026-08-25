from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .backtest import BacktestEngine, PerformanceReport
from .dna import StrategyDNA


@dataclass(frozen=True, slots=True)
class SelectionConfig:
    min_train_trades: int = 10
    min_validation_trades: int = 5
    min_profit_factor: float = 1.0
    max_drawdown: float = 0.35
    min_validation_return: float = -0.05
    max_complexity: float = 8.0
    volume_reliable: bool = True
    reject_volume_strategies_when_unreliable: bool = True
    elite_count: int = 10
    pareto_only: bool = False


@dataclass(frozen=True, slots=True)
class SelectionCandidate:
    dna: StrategyDNA
    train: PerformanceReport
    validation: PerformanceReport
    score: float
    robust: bool
    pareto: bool
    rejection_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SelectionReport:
    candidates: tuple[SelectionCandidate, ...]
    selected: tuple[SelectionCandidate, ...]

    @property
    def robust_count(self) -> int:
        return sum(c.robust for c in self.candidates)

    @property
    def pareto_count(self) -> int:
        return sum(c.pareto for c in self.candidates)


class SelectionEngine:
    """Development/validation selector. It never accepts OOS data.

    Train is used for hypothesis quality; validation is used for robustness.
    OOS must remain outside this API and is reserved for the final gate.
    """

    def __init__(self, backtester: BacktestEngine | None = None, config: SelectionConfig | None = None) -> None:
        self.bt = backtester or BacktestEngine()
        self.config = config or SelectionConfig()

    @staticmethod
    def _finite(value: float, fallback: float = 0.0) -> float:
        return float(value) if isfinite(value) else fallback

    def _score(self, train: PerformanceReport, validation: PerformanceReport) -> float:
        train_pf = min(self._finite(train.profit_factor), 5.0)
        val_pf = min(self._finite(validation.profit_factor), 5.0)
        stability = 1.0 - min(abs(train.total_return - validation.total_return), 1.0)
        return (
            2.0 * validation.total_return
            + 1.0 * train.total_return
            + 0.35 * self._finite(validation.sharpe)
            + 0.20 * self._finite(train.sharpe)
            + 0.20 * val_pf
            + 0.10 * train_pf
            + 0.50 * stability
            + 0.20 * validation.win_rate
            + 0.10 * self._finite(validation.expectancy) * 10.0
            - 3.0 * validation.max_drawdown
            - 1.0 * train.max_drawdown
            - 0.002 * validation.complexity
        )

    def _reasons(self, dna: StrategyDNA, train: PerformanceReport, validation: PerformanceReport) -> list[str]:
        c = self.config
        reasons: list[str] = []
        if train.trades < c.min_train_trades:
            reasons.append("insufficient_train_trades")
        if validation.trades < c.min_validation_trades:
            reasons.append("insufficient_validation_trades")
        if validation.profit_factor < c.min_profit_factor:
            reasons.append("validation_profit_factor_below_threshold")
        if validation.max_drawdown > c.max_drawdown:
            reasons.append("validation_drawdown_above_threshold")
        if validation.total_return < c.min_validation_return:
            reasons.append("validation_return_below_threshold")
        if dna.complexity > c.max_complexity:
            reasons.append("complexity_above_threshold")
        if (not c.volume_reliable and c.reject_volume_strategies_when_unreliable and dna.use_volume_filter):
            reasons.append("volume_unreliable")
        return reasons

    @staticmethod
    def _dominates(a: SelectionCandidate, b: SelectionCandidate) -> bool:
        av = (a.validation.total_return, a.validation.sharpe, a.validation.profit_factor, -a.validation.max_drawdown)
        bv = (b.validation.total_return, b.validation.sharpe, b.validation.profit_factor, -b.validation.max_drawdown)
        return all(x >= y for x, y in zip(av, bv)) and any(x > y for x, y in zip(av, bv))

    def _pareto_front(self, candidates: list[SelectionCandidate]) -> set[str]:
        front: set[str] = set()
        for candidate in candidates:
            if not any(self._dominates(other, candidate) for other in candidates if other is not candidate):
                front.add(candidate.dna.fingerprint())
        return front

    def evaluate(self, train_df, validation_df, population) -> SelectionReport:
        candidates: list[SelectionCandidate] = []
        for dna in population:
            train = self.bt.run(train_df, dna)
            validation = self.bt.run(validation_df, dna)
            reasons = self._reasons(dna, train, validation)
            candidates.append(SelectionCandidate(
                dna=dna,
                train=train,
                validation=validation,
                score=self._score(train, validation),
                robust=not reasons,
                pareto=False,
                rejection_reasons=tuple(reasons),
            ))

        eligible = [c for c in candidates if c.robust]
        front = self._pareto_front(eligible)
        candidates = [
            SelectionCandidate(c.dna, c.train, c.validation, c.score, c.robust,
                               c.dna.fingerprint() in front, c.rejection_reasons)
            for c in candidates
        ]
        ranked = [c for c in candidates if c.robust and (not self.config.pareto_only or c.pareto)]
        ranked.sort(key=lambda c: (c.pareto, c.score), reverse=True)
        selected = tuple(ranked[: max(0, self.config.elite_count)])
        return SelectionReport(tuple(candidates), selected)
