from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series: return s.rolling(n).mean()
def ema(s: pd.Series, n: int) -> pd.Series: return s.ewm(span=n, adjust=False).mean()
def atr(df: pd.DataFrame, n: int) -> pd.Series:
    prev = df.close.shift(1)
    tr = pd.concat([df.high-df.low, (df.high-prev).abs(), (df.low-prev).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()
def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff(); up = d.clip(lower=0); down = -d.clip(upper=0)
    rs = up.ewm(alpha=1/n, adjust=False).mean() / down.ewm(alpha=1/n, adjust=False).mean().replace(0, np.nan)
    return 100 - 100/(1+rs)
def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up = df.high.diff(); dn = -df.low.diff()
    plus = up.where((up > dn) & (up > 0), 0.0); minus = dn.where((dn > up) & (dn > 0), 0.0)
    a = atr(df, n).replace(0, np.nan)
    pdi = 100*plus.rolling(n).mean()/a; mdi = 100*minus.rolling(n).mean()/a
    return (100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)).rolling(n).mean()

def build_features(df: pd.DataFrame, dna) -> pd.DataFrame:
    x = df.copy()
    x["fast"] = ema(x.close, dna.fast)
    x["slow"] = ema(x.close, dna.slow)
    x["rsi"] = rsi(x.close, dna.rsi_period)
    x["atr"] = atr(x, dna.rsi_period)
    x["adx"] = adx(x, dna.rsi_period)
    x["ret"] = x.close.pct_change()
    return x
