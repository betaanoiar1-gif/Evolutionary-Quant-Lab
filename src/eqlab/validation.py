from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .backtest import BacktestEngine
from .dna import StrategyDNA


@dataclass(frozen=True, slots=True)
class RobustnessReport:
    walk_forward_mean: float
    walk_forward_std: float
    monte_carlo_p05: float
    monte_carlo_p50: float
    monte_carlo_p95: float
    parameter_stability: float
    regime_count: int
    regime_consistency: float
    overfit_penalty: float
    complexity_penalty: float
    passed: bool


class RobustnessEngine:
    def __init__(self, backtester: BacktestEngine | None = None, seed: int = 42) -> None:
        self.bt = backtester or BacktestEngine()
        self.seed = seed

    def walk_forward(self, df: pd.DataFrame, dna: StrategyDNA, folds: int = 5) -> list[float]:
        if folds < 2:
            raise ValueError("folds must be >= 2")
        n = len(df)
        values: list[float] = []
        train_len = n // (folds + 1)
        test_len = train_len
        for i in range(folds):
            train_end = train_len * (i + 1)
            test_end = min(train_end + test_len, n)
            if train_end < 30 or test_end - train_end < 10:
                continue
            result = self.bt.run(df.iloc[:train_end], dna)
            test = self.bt.run(df.iloc[train_end:test_end], dna)
            if result.trades >= 3 and test.trades >= 1:
                values.append(test.total_return)
        return values

    def monte_carlo(self, trade_returns: tuple[float, ...] | list[float], paths: int = 2000) -> np.ndarray:
        returns = np.asarray(trade_returns, dtype=float)
        if len(returns) < 3:
            return np.array([0.0, 0.0, 0.0])
        rng = np.random.default_rng(self.seed)
        samples = rng.choice(returns, size=(paths, len(returns)), replace=True)
        wealth = np.prod(1.0 + samples, axis=1) - 1.0
        return np.quantile(wealth, [0.05, 0.50, 0.95])

    def parameter_stability(self, df: pd.DataFrame, dna: StrategyDNA) -> float:
        base = self.bt.run(df, dna).total_return
        variants: list[float] = []
        for delta in (-3, -1, 1, 3):
            fast = max(2, dna.fast + delta)
            slow = max(fast + 1, dna.slow + delta)
            try:
                variant = replace(dna, fast=fast, slow=slow)
                variants.append(self.bt.run(df, variant).total_return)
            except ValueError:
                continue
        if not variants:
            return 0.0
        scale = max(abs(base), 0.01)
        dispersion = float(np.std([base, *variants])) / scale
        return float(max(0.0, 1.0 - min(dispersion, 1.0)))

    def regime_analysis(self, df: pd.DataFrame, dna: StrategyDNA, windows: int = 3) -> tuple[int, float]:
        if len(df) < windows * 20:
            return 0, 0.0
        returns = df.close.pct_change().dropna()
        vol = returns.rolling(min(50, max(10, len(returns) // 10))).std()
        trend = df.close.pct_change(min(50, max(10, len(df) // 10)))
        regime = (vol > vol.median()).astype(int) + (trend > 0).astype(int) * 2
        values = []
        for code in sorted(regime.dropna().unique()):
            mask = regime == code
            idx = np.flatnonzero(mask.to_numpy())
            if len(idx) < 20:
                continue
            values.append(float(self.bt.run(df.iloc[idx], dna).total_return))
        if not values:
            return 0, 0.0
        consistency = sum(v > 0 for v in values) / len(values)
        return len(values), float(consistency)

    def evaluate(self, train: pd.DataFrame, validation: pd.DataFrame, dna: StrategyDNA) -> RobustnessReport:
        train_report = self.bt.run(train, dna)
        validation_report = self.bt.run(validation, dna)
        wf = self.walk_forward(train, dna)
        mc = self.monte_carlo(train_report.trade_returns)
        stability = self.parameter_stability(validation, dna)
        regime_count, regime_consistency = self.regime_analysis(validation, dna)
        wf_mean = float(np.mean(wf)) if wf else -1.0
        wf_std = float(np.std(wf)) if wf else 1.0
        gap = train_report.total_return - validation_report.total_return
        overfit_penalty = max(0.0, gap - 0.05)
        complexity_penalty = max(0.0, dna.complexity - 3.0) * 0.02
        passed = (
            train_report.trades >= 10
            and validation_report.trades >= 5
            and wf_mean > -0.05
            and validation_report.max_drawdown < 0.50
            and stability >= 0.50
            and regime_consistency >= 0.50
            and overfit_penalty < 0.25
        )
        return RobustnessReport(
            wf_mean, wf_std, float(mc[0]), float(mc[1]), float(mc[2]), stability,
            regime_count, regime_consistency, overfit_penalty, complexity_penalty, passed,
        )


class OOSGate:
    """Final evaluation gate. It has no optimization, ranking, or mutation API."""

    def __init__(self, backtester: BacktestEngine | None = None) -> None:
        self.bt = backtester or BacktestEngine()

    def evaluate(self, oos: pd.DataFrame, dna: StrategyDNA):
        return self.bt.run(oos, dna)
