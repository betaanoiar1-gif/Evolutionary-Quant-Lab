from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REQUIRED = ("timestamp", "open", "high", "low", "close", "volume")


@dataclass(frozen=True, slots=True)
class DataSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    oos: pd.DataFrame


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
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        out = df.loc[:, REQUIRED].copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="raise")
        for c in REQUIRED[1:]:
            out[c] = pd.to_numeric(out[c], errors="raise")
        out = out.sort_values("timestamp").reset_index(drop=True)
        self.validate(out)
        return out

    def validate(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError("Dataset is empty")
        if df["timestamp"].duplicated().any():
            raise ValueError("Duplicate timestamps detected")
        if not df["timestamp"].is_monotonic_increasing:
            raise ValueError("Timestamps must be increasing")
        if df[list(REQUIRED[1:])].isna().any().any():
            raise ValueError("NaN values detected")
        if not (df["high"] >= df[["open", "close", "low"]].max(axis=1)).all():
            raise ValueError("Invalid high prices")
        if not (df["low"] <= df[["open", "close", "high"]].min(axis=1)).all():
            raise ValueError("Invalid low prices")
        if (df["volume"] < 0).any():
            raise ValueError("Negative volume detected")

    def fingerprint(self, df: pd.DataFrame) -> str:
        payload = pd.util.hash_pandas_object(df, index=True).values.tobytes()
        return hashlib.sha256(payload).hexdigest()

    def temporal_split(self, df: pd.DataFrame, train: float = .6, validation: float = .2) -> DataSplit:
        if not 0 < train < 1 or not 0 < validation < 1 or train + validation >= 1:
            raise ValueError("Invalid split ratios")
        n = len(df)
        a, b = int(n * train), int(n * (train + validation))
        if min(a, b - a, n - b) < 2:
            raise ValueError("Each split needs at least two rows")
        return DataSplit(df.iloc[:a].copy(), df.iloc[a:b].copy(), df.iloc[b:].copy())
