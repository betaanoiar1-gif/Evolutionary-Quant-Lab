from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def wma(s: pd.Series, n: int) -> pd.Series:
    weights = np.arange(1, n + 1, dtype=float)
    return s.rolling(n, min_periods=n).apply(lambda x: float(np.dot(x, weights) / weights.sum()), raw=True)


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    prev = df.close.shift(1)
    tr = pd.concat([df.high - df.low, (df.high - prev).abs(), (df.low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0)
    down = -d.clip(upper=0)
    avg_up = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_down = down.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series]:
    line = ema(s, fast) - ema(s, slow)
    return line, ema(line, signal)


def bollinger(s: pd.Series, n: int = 20, k: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(s, n)
    std = s.rolling(n, min_periods=n).std()
    return mid - k * std, mid, mid + k * std


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up = df.high.diff()
    down = -df.low.diff()
    plus = up.where((up > down) & (up > 0), 0.0)
    minus = down.where((down > up) & (down > 0), 0.0)
    a = atr(df, n).replace(0, np.nan)
    pdi = 100 * plus.rolling(n, min_periods=n).mean() / a
    mdi = 100 * minus.rolling(n, min_periods=n).mean() / a
    return (100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)).rolling(n, min_periods=n).mean()


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df.close.diff()).fillna(0)
    return (direction * df.volume).cumsum()


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    name: str
    calculator: Callable
    category: str


class FeatureRegistry:
    def __init__(self) -> None:
        self._items: dict[str, FeatureSpec] = {}

    def register(self, spec: FeatureSpec) -> None:
        if spec.name in self._items:
            raise ValueError(f"Feature already registered: {spec.name}")
        self._items[spec.name] = spec

    def get(self, name: str) -> FeatureSpec:
        try:
            return self._items[name]
        except KeyError as exc:
            raise KeyError(f"Unknown feature: {name}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))


def default_registry() -> FeatureRegistry:
    registry = FeatureRegistry()
    for name, fn, category in (
        ("sma", sma, "trend"), ("ema", ema, "trend"), ("wma", wma, "trend"),
        ("atr", atr, "volatility"), ("rsi", rsi, "momentum"), ("adx", adx, "trend"),
        ("macd", macd, "momentum"), ("bollinger", bollinger, "volatility"),
        ("obv", obv, "volume"),
    ):
        registry.register(FeatureSpec(name, fn, category))
    return registry


def build_features(df: pd.DataFrame, dna) -> pd.DataFrame:
    x = df.copy()
    x["fast"] = ema(x.close, dna.fast)
    x["slow"] = ema(x.close, dna.slow)
    x["rsi"] = rsi(x.close, dna.rsi_period)
    x["atr"] = atr(x, dna.rsi_period)
    x["adx"] = adx(x, dna.rsi_period)
    x["ret"] = x.close.pct_change()
    x["volume_ma"] = sma(x.volume, max(2, dna.rsi_period))
    x["obv"] = obv(x)
    x["macd"], x["macd_signal"] = macd(x.close)
    x["bb_low"], x["bb_mid"], x["bb_high"] = bollinger(x.close)
    return x
