from __future__ import annotations

from dataclasses import dataclass
from .backtest import BacktestEngine

@dataclass(frozen=True, slots=True)
class Regime:
    trend: str
    volatility: str

class RegimeDetector:
    def detect(self, df):
        ret=df.close.pct_change().dropna()
        if len(ret)<20: raise ValueError("At least 20 returns are required")
        vol=float(ret.std())
        baseline=float(ret.rolling(100,min_periods=20).std().median())
        trend="bull" if df.close.iloc[-1]>df.close.iloc[0] else "bear"
        return Regime(trend,"high" if baseline and vol>baseline*1.25 else "low" if baseline and vol<baseline*.75 else "normal")

class MetaStrategy:
    def __init__(self, strategies, backtester=None): self.strategies=list(strategies); self.bt=backtester or BacktestEngine()
    def choose(self, df):
        if not self.strategies: raise ValueError("No strategies available")
        regime=RegimeDetector().detect(df)
        ranked=[]
        for d in self.strategies:
            r=self.bt.run(df,d); ranked.append((r.total_return-r.max_drawdown*2,r,d))
        return max(ranked,key=lambda x:x[0])[2], regime
