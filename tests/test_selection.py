import numpy as np
import pandas as pd

from eqlab.backtest import BacktestEngine
from eqlab.dna import StrategyDNA
from eqlab.selection import SelectionConfig, SelectionEngine


def market(n=700):
    rng = np.random.default_rng(21)
    ret = rng.normal(.0004, .008, n)
    close = 100 * np.cumprod(1 + ret)
    ts = pd.date_range("2021-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts,
        "open": close,
        "high": close * 1.005,
        "low": close * .995,
        "close": close,
        "volume": rng.uniform(10, 100, n),
    })


def test_selection_is_deterministic_and_oos_free():
    df = market()
    train = df.iloc[:450].reset_index(drop=True)
    validation = df.iloc[450:].reset_index(drop=True)
    population = [StrategyDNA(10, 50), StrategyDNA(20, 100)]
    engine = SelectionEngine(config=SelectionConfig(min_train_trades=0, min_validation_trades=0))
    a = engine.evaluate(train, validation, population)
    b = engine.evaluate(train, validation, population)
    assert [c.dna.fingerprint() for c in a.selected] == [c.dna.fingerprint() for c in b.selected]
    assert all("oos" not in r for c in a.candidates for r in c.rejection_reasons)


def test_volume_filter_rejected_when_volume_unreliable():
    dna = StrategyDNA(10, 50, use_volume_filter=True)
    engine = SelectionEngine(config=SelectionConfig(
        min_train_trades=0, min_validation_trades=0, volume_reliable=False,
    ))
    report = engine.evaluate(market(300), market(300), [dna])
    assert "volume_unreliable" in report.candidates[0].rejection_reasons
    assert not report.candidates[0].robust


def test_pareto_front_marks_non_dominated_candidates():
    df = market()
    train = df.iloc[:400].reset_index(drop=True)
    validation = df.iloc[400:].reset_index(drop=True)
    population = [StrategyDNA(10, 50), StrategyDNA(15, 80), StrategyDNA(25, 120)]
    report = SelectionEngine(config=SelectionConfig(
        min_train_trades=0, min_validation_trades=0, elite_count=3,
    )).evaluate(train, validation, population)
    assert report.pareto_count >= 1
    assert report.selected


def test_selected_are_robust_only():
    df = market()
    engine = SelectionEngine(config=SelectionConfig(
        min_train_trades=10_000, min_validation_trades=0, elite_count=5,
    ))
    report = engine.evaluate(df.iloc[:500], df.iloc[500:], [StrategyDNA(10, 50)])
    assert report.selected == ()
