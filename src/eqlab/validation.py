from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from .backtest import BacktestEngine

@dataclass(frozen=True, slots=True)
class RobustnessReport:
    walk_forward_mean: float
    walk_forward_std: float
    monte_carlo_p05: float
    monte_carlo_p50: float
    monte_carlo_p95: float
    parameter_stability: float
    regime_count: int
    overfit_penalty: float
    passed: bool

class RobustnessEngine:
    def __init__(self, backtester=None, seed=42): self.bt=backtester or BacktestEngine(); self.seed=seed
    def walk_forward(self, df, dna, folds=5):
        n=len(df); vals=[]
        for i in range(folds):
            a=int(i*n/(folds+1)); b=int((i+1)*n/(folds+1)); c=int((i+2)*n/(folds+1))
            if c>n or b-a<20 or c-b<10: continue
            vals.append(self.bt.run(df.iloc[a:b],dna).total_return)
        return vals
    def monte_carlo(self, returns, paths=500):
        rng=np.random.default_rng(self.seed); r=np.asarray(returns,float)
        if len(r)<2:return np.array([0.0])
        samples=rng.choice(r,(paths,len(r)),replace=True); wealth=np.prod(1+samples,axis=1)-1
        return np.quantile(wealth,[.05,.5,.95])
    def parameter_stability(self, df, dna):
        base=self.bt.run(df,dna).total_return; vals=[]
        for f in (-2,2):
            try:
                from .dna import StrategyDNA
                d=StrategyDNA(max(2,dna.fast+f),dna.slow,dna.rsi_period,dna.rsi_entry,dna.stop_atr,dna.take_atr,dna.allow_short)
                vals.append(self.bt.run(df,d).total_return)
            except ValueError: pass
        if not vals:return 0.0
        return float(max(0,1-np.std([base,*vals])/(abs(base)+1e-9)))
    def evaluate(self, train, validation, dna):
        wf=self.walk_forward(train,dna); vr=self.bt.run(validation,dna); mc=self.monte_carlo(pd.Series(self.bt.run(train,dna).total_return).repeat(max(2,vr.trades)).to_numpy())
        stability=self.parameter_stability(validation,dna)
        wf_mean=float(np.mean(wf)) if wf else -1.0; wf_std=float(np.std(wf)) if wf else 999.0
        penalty=max(0.0, abs(self.bt.run(train,dna).total_return-vr.total_return)-0.15)
        passed=wf_mean>0 and vr.max_drawdown<.5 and stability>.2 and penalty<.5
        return RobustnessReport(wf_mean,wf_std,float(mc[0]),float(mc[1]),float(mc[2]),stability,2,penalty,passed)

class OOSGate:
    """Final evaluation gate. It intentionally exposes no optimization method."""
    def __init__(self, backtester=None): self.bt=backtester or BacktestEngine()
    def evaluate(self, oos, dna): return self.bt.run(oos,dna)
