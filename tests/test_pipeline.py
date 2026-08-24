import numpy as np
import pandas as pd

from eqlab.data import DataEngine
from eqlab.pipeline import Laboratory


def market(n=450):
    rng = np.random.default_rng(321)
    close = 100 * np.cumprod(1 + rng.normal(.00015, .007, n))
    ts = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"timestamp": ts, "open": close, "high": close * 1.004,
                         "low": close * .996, "close": close, "volume": rng.uniform(100, 1000, n)})


def test_pipeline_preserves_oos_boundary(tmp_path):
    data = DataEngine().temporal_split(market())
    lab = Laboratory(seed=11)
    population = lab.gen.generate(12)
    survivors = lab.screen(data.train, population)
    evolved = lab.evolve(data.train, survivors or population, generations=1, elite=3)
    ranked = lab.robust_rank(data.train, data.validation, evolved, top_n=min(5, len(evolved)))
    assert ranked
    assert all(hasattr(item, "robustness") for item in ranked)
    # OOS is only touched by this explicit final gate.
    report = lab.final_evaluate(data.oos, ranked[0].candidate.dna)
    assert report.initial == 10_000
