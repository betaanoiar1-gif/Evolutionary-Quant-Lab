import numpy as np
import pandas as pd

from eqlab.backtest import BacktestEngine, CostModel
from eqlab.dna import StrategyDNA
from eqlab.fast import FastBacktestEngine
from eqlab.meta import MetaStrategy, RegimeDetector
from eqlab.reporting import ReportWriter


def market(n=500):
    rng = np.random.default_rng(9)
    close = 100 * np.cumprod(1 + rng.normal(.0002, .008, n))
    ts = pd.date_range("2021-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"timestamp": ts, "open": close, "high": close * 1.005,
                         "low": close * .995, "close": close, "volume": 100})


def test_fast_and_full_exist():
    df = market(); d = StrategyDNA(10, 50)
    fast = FastBacktestEngine().run(df, d); full = BacktestEngine().run(df, d)
    assert fast.trades >= 0 and np.isfinite(full.final)


def test_funding_changes_equity_when_positions_exist():
    df = market(); d = StrategyDNA(10, 50)
    a = BacktestEngine(CostModel(funding_rate=0)).run(df, d)
    b = BacktestEngine(CostModel(funding_rate=.0001)).run(df, d)
    assert b.funding != 0 or a.trades == 0


def test_regime_and_report_writer(tmp_path):
    df = market(); d = StrategyDNA(10, 50); report = BacktestEngine().run(df, d)
    regime = RegimeDetector().detect(df)
    assert regime.trend in {"bull", "bear"}
    p = ReportWriter().write(tmp_path / "report.json", report)
    assert p.exists()


def test_meta_strategy_requires_development_fit():
    df = market(); strategies = [StrategyDNA(10, 50), StrategyDNA(20, 100)]
    meta = MetaStrategy(strategies)
    meta.fit(df)
    regime = RegimeDetector().detect(df)
    assert meta.choose(regime) in strategies
