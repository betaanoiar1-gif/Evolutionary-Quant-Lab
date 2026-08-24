from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .backtest import BacktestEngine
from .overfitting import OverfittingDetector

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
    def __init__(self, backtester=None, seed=42): self.bt=backtester or BacktestEngine(); self.seed=seed; self.overfit=OverfittingDetector()
    def walk_forward(self, df, dna, folds=5):
        n=len(df); vals=[]
        for i in range(folds):
            a=int(i*n/(folds+1)); b=int((i+1)*n/(folds+1)); c=int((i+2)*n/(folds+1))
            if c>n or b-a<20 or c-b<10: continue
            vals.append(self.bt.run(df.iloc[b:c],dna).total_return)
        return vals
    def monte_carlo(self, trade_returns, paths=500):
        rng=np.random.default_rng(self.seed); r=np.asarray(trade_returns,float)
        if len(r)<2:return np.array([0.,0.,0.])
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
        train_report=self.bt.run(train,dna); vr=self.bt.run(validation,dna); wf=self.walk_forward(train,dna); mc=self.monte_carlo(train_report.trade_returns); stability=self.parameter_stability(validation,dna)
        wf_mean=float(np.mean(wf)) if wf else -1.; wf_std=float(np.std(wf)) if wf else 999.
        assessment=self.overfit.assess(train_report.total_return,vr.total_return,dna.complexity)
        passed=wf_mean>0 and vr.max_drawdown<.5 and stability>.2 and not assessment.flagged and len(train_report.trade_returns)>=10
        return RobustnessReport(wf_mean,wf_std,float(mc[0]),float(mc[1]),float(mc[2]),stability,2,assessment.score,passed)

class OOSGate:
    """Final evaluation gate. It intentionally exposes no optimization method."""
    def __init__(self, backtester=None): self.bt=backtester or BacktestEngine()
    def evaluate(self, oos, dna): return self.bt.run(oos,dna)
