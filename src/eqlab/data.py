from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ("timestamp", "open", "high", "low", "close", "volume")


@dataclass(frozen=True, slots=True)
class DataSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    oos: pd.DataFrame

    def __post_init__(self) -> None:
        if self.train.empty or self.validation.empty or self.oos.empty:
            raise ValueError("All temporal splits must be non-empty")
        if self.train.timestamp.max() >= self.validation.timestamp.min():
            raise ValueError("Train and validation overlap")
        if self.validation.timestamp.max() >= self.oos.timestamp.min():
            raise ValueError("Validation and OOS overlap")


class DataEngine:
    def load(self, path: str | Path) -> pd.DataFrame:
        path = Path(path)
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        elif path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        else:
            raise ValueError("Only CSV and Parquet are supported")
        return self.normalize(df)

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        out = df.loc[:, REQUIRED].copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="raise")
        for c in REQUIRED[1:]:
            out[c] = pd.to_numeric(out[c], errors="raise")
        if out["timestamp"].duplicated().any():
            raise ValueError("Duplicate timestamps detected")
        out = out.sort_values("timestamp").reset_index(drop=True)
        self.validate(out)
        return out

    def validate(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError("Dataset is empty")
        if not df["timestamp"].is_monotonic_increasing:
            raise ValueError("Timestamps must be increasing")
        if df["timestamp"].duplicated().any():
            raise ValueError("Duplicate timestamps detected")

        numeric = list(REQUIRED[1:])
        values = df[numeric]

        # Check finiteness before OHLC relationship checks so NaN/Inf inputs
        # receive deterministic, diagnostic errors instead of secondary OHLC
        # validation failures.
        if values.isna().any().any():
            raise ValueError("NaN values detected")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError("Infinite market values detected")
        if not values.map(pd.api.types.is_number).all().all():
            raise ValueError("Non-numeric market values detected")

        if not (df["high"] >= df[["open", "close", "low"]].max(axis=1)).all():
            raise ValueError("Invalid high prices")
        if not (df["low"] <= df[["open", "close", "high"]].min(axis=1)).all():
            raise ValueError("Invalid low prices")
        if (df[["open", "high", "low", "close"]] <= 0).any().any():
            raise ValueError("Prices must be positive")
        if (df["volume"] < 0).any():
            raise ValueError("Negative volume detected")

    def validate_frequency(self, df: pd.DataFrame, expected: str, tolerance: int = 0) -> None:
        if len(df) < 3:
            return
        expected_delta = pd.Timedelta(expected)
        deltas = df.timestamp.diff().dropna()
        bad = (deltas - expected_delta).abs() > pd.Timedelta(seconds=tolerance)
        if bad.any():
            raise ValueError(f"Unexpected candle interval detected: expected {expected}")

    def fingerprint(self, df: pd.DataFrame) -> str:
        normalized = self.normalize(df)
        payload = pd.util.hash_pandas_object(normalized, index=True).values.tobytes()
        return hashlib.sha256(payload).hexdigest()

    def temporal_split(self, df: pd.DataFrame, train: float = .6, validation: float = .2) -> DataSplit:
        if not 0 < train < 1 or not 0 < validation < 1 or train + validation >= 1:
            raise ValueError("Invalid split ratios")
        clean = self.normalize(df)
        n = len(clean)
        a, b = int(n * train), int(n * (train + validation))
        if min(a, b - a, n - b) < 2:
            raise ValueError("Each split needs at least two rows")
        return DataSplit(clean.iloc[:a].copy(), clean.iloc[a:b].copy(), clean.iloc[b:].copy())
