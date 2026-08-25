"""Market-data provider contracts and canonicalization helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd

from eqlab.data import REQUIRED


@dataclass(frozen=True, slots=True)
class DataQualityReport:
    rows: int
    duplicate_timestamps: int
    nan_values: int
    infinite_values: int
    zero_volume_rows: int
    negative_volume_rows: int
    temporal_gaps: int
    max_gap: pd.Timedelta | None

    @property
    def volume_reliable(self) -> bool:
        return self.rows > 0 and self.zero_volume_rows == 0 and self.negative_volume_rows == 0

    @property
    def passed(self) -> bool:
        return (
            self.rows > 0
            and self.duplicate_timestamps == 0
            and self.nan_values == 0
            and self.infinite_values == 0
            and self.negative_volume_rows == 0
        )


class MarketDataProvider(ABC):
    """Provider boundary; research code must not depend on a vendor SDK."""

    name: str

    @abstractmethod
    def fetch_ohlcv(self, symbol: str, timeframe: str, **kwargs: Any) -> pd.DataFrame:
        """Return canonical OHLCV columns."""


class DataQualityEngine:
    """Inspect data without silently repairing or deleting observations."""

    def audit(self, df: pd.DataFrame, expected_interval: str | None = None) -> DataQualityReport:
        if not set(REQUIRED).issubset(df.columns):
            raise ValueError(f"Missing required columns: {set(REQUIRED) - set(df.columns)}")

        numeric = df[list(REQUIRED[1:])]
        timestamps = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
        deltas = timestamps.sort_values().diff().dropna()
        expected = pd.Timedelta(expected_interval) if expected_interval else None
        gaps = int((deltas > expected).sum()) if expected is not None else 0
        max_gap = deltas.max() if len(deltas) else None

        import numpy as np

        values = numeric.to_numpy(dtype=float)
        return DataQualityReport(
            rows=len(df),
            duplicate_timestamps=int(timestamps.duplicated().sum()),
            nan_values=int(numeric.isna().sum().sum()),
            infinite_values=int((~np.isfinite(values)).sum()),
            zero_volume_rows=int(df["volume"].eq(0).sum()),
            negative_volume_rows=int(df["volume"].lt(0).sum()),
            temporal_gaps=gaps,
            max_gap=max_gap,
        )


def canonicalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common provider column naming to the laboratory schema."""
    aliases = {
        "Datetime": "timestamp",
        "Date": "timestamp",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [str(col[0]) for col in out.columns]
    out = out.rename(columns=aliases)
    missing = [c for c in REQUIRED if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns after canonicalization: {missing}")
    return out.loc[:, REQUIRED].copy()
