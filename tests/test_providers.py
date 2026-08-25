import numpy as np
import pandas as pd
import pytest

from eqlab.providers import DataQualityEngine, canonicalize_ohlcv


def sample() -> pd.DataFrame:
    ts = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts,
        "open": [100, 101, 102, 103],
        "high": [101, 102, 103, 104],
        "low": [99, 100, 101, 102],
        "close": [100.5, 101.5, 102.5, 103.5],
        "volume": [10, 11, 12, 13],
    })


def test_canonicalize_yahoo_style_columns() -> None:
    df = sample().set_index("timestamp")
    df.columns = pd.MultiIndex.from_tuples([(c, "BTC-USD") for c in df.columns])
    df.index.name = "Datetime"
    out = canonicalize_ohlcv(df.reset_index())
    assert list(out.columns) == ["timestamp", "open", "high", "low", "close", "volume"]


def test_quality_report_detects_gaps_and_zero_volume() -> None:
    df = sample()
    df.loc[2, "volume"] = 0
    df.loc[3, "timestamp"] += pd.Timedelta(hours=2)
    report = DataQualityEngine().audit(df, expected_interval="1h")
    assert report.zero_volume_rows == 1
    assert report.temporal_gaps == 1
    assert report.max_gap == pd.Timedelta(hours=3)
    assert report.passed is True
    assert report.volume_reliable is False


def test_quality_report_rejects_nan_and_infinite_quality() -> None:
    df = sample()
    df.loc[1, "close"] = np.inf
    report = DataQualityEngine().audit(df)
    assert report.infinite_values == 1
    assert report.passed is False


def test_canonicalization_requires_all_ohlcv_fields() -> None:
    with pytest.raises(ValueError, match="Missing required"):
        canonicalize_ohlcv(pd.DataFrame({"Open": [1]}))
