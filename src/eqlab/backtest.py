from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import pandas as pd

from .features import build_features

@dataclass(frozen=True, slots=True)
class CostModel:
    fee_rate: float = 0.0004
    slippage_bps: float = 2.0
    funding_rate: float = 0.0
    def __post_init__(self):
        if min(self.fee_rate, self.slippage_bps, self.funding_rate) < 0: raise ValueError("Costs cannot be negative")
    def execution_price(self, price: float, side: int) -> float:
        return price * (1 + side * self.slippage_bps / 10000)
    def trade_cost(self, notional: float) -> float:
        return abs(notional) * self.fee_rate

@dataclass(frozen=True, slots=True)
class RiskModel:
    risk_per_trade: float = .01
    max_exposure: float = 1.0
    def size(self, equity: float, entry: float, stop: float) -> float:
        if equity <= 0 or entry <= 0 or stop <= 0: return 0.0
        risk_cash = equity * self.risk_per_trade
        distance = abs(entry-stop)
        if distance == 0: return 0.0
        return min(risk_cash/distance, equity*self.max_exposure/entry)

@dataclass(frozen=True, slots=True)
class PerformanceReport:
    initial: float
    final: float
    total_return: float
    max_drawdown: float
    sharpe: float
    sortino: float
    trades: int
    win_rate: float
    profit_factor: float
    net_profit: float
    fees: float
    slippage: float
    complexity: float

class BacktestEngine:
    def __init__(self, costs=None, risk=None): self.costs=costs or CostModel(); self.risk=risk or RiskModel()
    def run(self, df: pd.DataFrame, dna, initial: float=10_000) -> PerformanceReport:
        x=build_features(df,dna).dropna().reset_index(drop=True)
        equity=initial; peak=initial; maxdd=0.0; pos=0; entry=stop=target=0.0; size=0.0
        wins=[]; gross_win=gross_loss=0.0; fees=slip=0.0; curve=[]
        for i,row in x.iterrows():
            price=float(row.close); a=float(row.atr)
            if pos:
                hit_stop = row.low <= stop if pos==1 else row.high >= stop
                hit_target = row.high >= target if pos==1 else row.low <= target
                if hit_stop or hit_target:
                    exit_raw=stop if hit_stop else target; side=-pos
                    exit_price=self.costs.execution_price(exit_raw, side)
                    pnl=pos*size*(exit_price-entry)
                    c=self.costs.trade_cost(size*exit_price); fees+=c; slip+=abs(exit_price-exit_raw)*size
                    equity += pnl-c; wins.append(pnl-c)
                    if pnl-c>=0:gross_win+=pnl-c
                    else:gross_loss+=-(pnl-c)
                    pos=0; size=0
            if pos==0 and i>0:
                prev=x.iloc[i-1]
                long_sig=prev.fast>prev.slow and prev.rsi>dna.rsi_entry
                short_sig=dna.allow_short and prev.fast<prev.slow and prev.rsi<100-dna.rsi_entry
                if long_sig or short_sig:
                    pos=1 if long_sig else -1; raw=price; entry=self.costs.execution_price(raw,pos)
                    stop=entry-pos*dna.stop_atr*a; target=entry+pos*dna.take_atr*a
                    size=self.risk.size(equity,entry,stop)
                    c=self.costs.trade_cost(size*entry); fees+=c; equity-=c
            curve.append(equity); peak=max(peak,equity); maxdd=max(maxdd,(peak-equity)/peak if peak else 0)
        if pos:
            raw=float(x.iloc[-1].close); exit_price=self.costs.execution_price(raw,-pos); pnl=pos*size*(exit_price-entry); c=self.costs.trade_cost(size*exit_price); equity+=pnl-c; fees+=c; wins.append(pnl-c)
            if pnl-c>=0:gross_win+=pnl-c
            else:gross_loss+=-(pnl-c)
        returns=pd.Series(curve).pct_change().dropna()
        sharpe=float(np.sqrt(252)*returns.mean()/returns.std()) if returns.std()>0 else 0.0
        neg=returns[returns<0]; sortino=float(np.sqrt(252)*returns.mean()/neg.std()) if len(neg)>1 and neg.std()>0 else 0.0
        pf=gross_win/gross_loss if gross_loss else (math.inf if gross_win else 0.0)
        return PerformanceReport(initial,equity,equity/initial-1,maxdd,sharpe,sortino,len(wins),sum(v>0 for v in wins)/len(wins) if wins else 0,pf,equity-initial,fees,slip,dna.complexity)
