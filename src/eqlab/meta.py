from __future__ import annotations

from dataclasses import dataclass
from .dna import StrategyDNA
from .backtest import BacktestEngine

@dataclass(frozen=True, slots=True)
class Regime:
    trend: str
    volatility: str

class RegimeDetector:
    def detect(self, df):
        ret=df.close.pct_change().dropna(); vol=float(ret.std())
        trend="bull" if df.close.iloc[-1]>df.close.iloc[0] else "bear"
        return Regime(trend,"high" if vol>float(ret.rolling(min(100,len(ret))).mean().abs().iloc[-1] if len(ret) else 0) else "normal")

class MetaStrategy:
    def __init__(self, strategies, backtester=None): self.strategies=list(strategies); self.bt=backtester or BacktestEngine()
    def choose(self, df):
        if not self.strategies: raise ValueError("No strategies available")
        regime=RegimeDetector().detect(df); ranked=[]
        for d in self.strategies:
            r=self.bt.run(df,d); ranked.append((r.total_return-r.max_drawdown*2,r,d))
        return max(ranked,key=lambda x:x[0])[2], regime
