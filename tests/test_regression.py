import numpy as np
import pandas as pd
import pytest

from eqlab.data import DataEngine
from eqlab.dna import StrategyDNA
from eqlab.evolution import EvolutionEngine, StrategyGenerator
from eqlab.overfitting import OverfittingDetector


def market(n=300):
    rng = np.random.default_rng(123)
    close = 100 * np.cumprod(1 + rng.normal(.0001, .006, n))
    ts = pd.date_range("2022-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"timestamp": ts, "open": close, "high": close * 1.003,
                         "low": close * .997, "close": close, "volume": 100.0})


def test_invalid_ohlc_is_rejected():
    df = market(); df.loc[10, "high"] = df.loc[10, "close"] - 1
    with pytest.raises(ValueError, match="Invalid high"):
        DataEngine().normalize(df)


def test_infinite_values_are_rejected():
    df = market(); df.loc[10, "close"] = np.inf
    with pytest.raises(ValueError, match="Infinite"):
        DataEngine().normalize(df)


def test_dna_round_trip_and_fingerprint_are_stable():
    dna = StrategyDNA(10, 50, use_adx_filter=True, use_volume_filter=True)
    restored = StrategyDNA.from_dict(dna.to_dict())
    assert restored == dna
    assert restored.fingerprint() == dna.fingerprint()


def test_evolution_is_seed_reproducible():
    a = StrategyGenerator(77).generate(20)
    b = StrategyGenerator(77).generate(20)
    assert [x.fingerprint() for x in a] == [x.fingerprint() for x in b]


def test_mutation_preserves_valid_genome():
    engine = EvolutionEngine(88); dna = StrategyDNA(10, 50)
    for _ in range(100):
        child = engine.mutate(dna)
        assert child.fast < child.slow
        dna = child


def test_overfitting_penalty_grows_with_trials():
    detector = OverfittingDetector()
    small = detector.assess(.20, .18, 2.0, trials=10)
    large = detector.assess(.20, .18, 2.0, trials=10000)
    assert large.multiple_testing_penalty > small.multiple_testing_penalty
