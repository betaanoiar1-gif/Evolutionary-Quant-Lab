from __future__ import annotations

from dataclasses import dataclass

from .backtest import BacktestEngine


@dataclass(frozen=True, slots=True)
class Regime:
    trend: str
    volatility: str

    @property
    def key(self) -> str:
        return f"{self.trend}:{self.volatility}"


class RegimeDetector:
    def detect(self, df) -> Regime:
        ret = df.close.pct_change().dropna()
        if len(ret) < 20:
            raise ValueError("At least 20 returns are required")
        vol = float(ret.std())
        baseline = float(ret.rolling(100, min_periods=20).std().median())
        trend = "bull" if df.close.iloc[-1] > df.close.iloc[0] else "bear"
        if baseline and vol > baseline * 1.25:
            volatility = "high"
        elif baseline and vol < baseline * .75:
            volatility = "low"
        else:
            volatility = "normal"
        return Regime(trend, volatility)


class MetaStrategy:
    """Regime-conditioned selector trained only on development data.

    ``fit`` must be called on train/validation data before deployment. ``choose``
    never re-scores strategies using the same observations it is asked to trade.
    """

    def __init__(self, strategies, backtester: BacktestEngine | None = None) -> None:
        self.strategies = list(strategies)
        self.bt = backtester or BacktestEngine()
        self.mapping: dict[str, object] = {}

    def fit(self, df) -> None:
        if not self.strategies:
            raise ValueError("No strategies available")
        detector = RegimeDetector()
        regime = detector.detect(df)
        ranked = []
        for dna in self.strategies:
            report = self.bt.run(df, dna)
            score = report.sharpe + report.sortino + report.total_return - report.max_drawdown * 3
            ranked.append((score, dna))
        self.mapping[regime.key] = max(ranked, key=lambda x: x[0])[1]

    def choose(self, current_regime: Regime):
        if not self.mapping:
            raise RuntimeError("MetaStrategy must be fitted before choose")
        return self.mapping.get(current_regime.key, next(iter(self.mapping.values())))
