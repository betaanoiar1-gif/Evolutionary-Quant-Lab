from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from .dna import StrategyDNA
from .features import build_features

@dataclass(frozen=True, slots=True)
class FastReport:
    total_return: float
    max_drawdown: float
    trades: int

class FastBacktestEngine:
    """Cheap deterministic screening engine. It deliberately excludes detailed execution costs."""
    def run(self, df: pd.DataFrame, dna: StrategyDNA, initial: float = 10_000) -> FastReport:
        x=build_features(df,dna).dropna()
        if len(x)<2: return FastReport(0.0,0.0,0)
        direction=np.where(x.fast>x.slow,1,np.where(x.fast<x.slow,-1,0))
        signal=pd.Series(direction,index=x.index).shift(1).fillna(0).to_numpy()
        ret=x.close.pct_change().fillna(0).to_numpy()
        equity=initial*np.cumprod(1+signal*ret)
        peak=np.maximum.accumulate(equity)
        dd=np.max((peak-equity)/np.where(peak==0,1,peak))
        trades=int(np.count_nonzero(np.diff(signal)!=0))
        return FastReport(float(equity[-1]/initial-1),float(dd),trades)
