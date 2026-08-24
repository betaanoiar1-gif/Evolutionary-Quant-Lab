from __future__ import annotations

import random
from dataclasses import dataclass
from .dna import StrategyDNA
from .backtest import BacktestEngine, PerformanceReport

class StrategyGenerator:
    def __init__(self, seed=42): self.rng=random.Random(seed)
    def generate(self, n=100):
        if n<1: raise ValueError("n must be positive")
        seen=set(); out=[]
        while len(out)<n:
            d=StrategyDNA(self.rng.randint(5,40), self.rng.randint(50,200), self.rng.randint(7,30), self.rng.uniform(40,60), self.rng.uniform(1,4), self.rng.uniform(1.5,6), self.rng.random()<.5)
            if d.fingerprint() not in seen: seen.add(d.fingerprint()); out.append(d)
        return out

class EvolutionEngine:
    def __init__(self, seed=42): self.rng=random.Random(seed)
    def mutate(self,d):
        vals=d.to_dict(); key=self.rng.choice(["fast","slow","rsi_period","rsi_entry","stop_atr","take_atr"])
        if key in ("fast","slow","rsi_period"): vals[key]=max(2,int(vals[key]+self.rng.randint(-5,5)))
        elif key=="rsi_entry": vals[key]=min(70,max(30,vals[key]+self.rng.uniform(-5,5)))
        else: vals[key]=max(.5,vals[key]+self.rng.uniform(-.5,.5))
        vals["fast"]=min(vals["fast"],vals["slow"]-1)
        vals["slow"]=max(vals["slow"],vals["fast"]+1)
        return StrategyDNA(**vals)
    def crossover(self,a,b):
        fields=a.to_dict(); other=b.to_dict()
        for k in fields:
            if self.rng.random()<.5: fields[k]=other[k]
        fields["fast"]=min(fields["fast"],fields["slow"]-1)
        fields["slow"]=max(fields["slow"],fields["fast"]+1)
        return StrategyDNA(**fields)

@dataclass(frozen=True, slots=True)
class Candidate:
    dna: StrategyDNA
    report: PerformanceReport
    score: float

def score(report: PerformanceReport) -> float:
    if report.trades < 5: return -1e9
    return (report.sharpe + report.sortino + report.total_return*3 + report.win_rate - report.max_drawdown*3) - .001*report.complexity

class SearchEngine:
    def __init__(self, backtester=None, seed=42): self.bt=backtester or BacktestEngine(); self.ev=EvolutionEngine(seed); self.gen=StrategyGenerator(seed)
    def evaluate(self, df, population):
        return [self._candidate(df,d) for d in population]
    def _candidate(self,df,d):
        report=self.bt.run(df,d); return Candidate(d,report,score(report))
    def evolve(self, df, population, generations=5, elite=10):
        if not population: raise ValueError("population must not be empty")
        if elite<1: raise ValueError("elite must be positive")
        pop=list(population); archive=[]
        for _ in range(generations):
            ranked=sorted(self.evaluate(df,pop), key=lambda c:c.score, reverse=True); keep=min(elite,len(ranked)); archive.extend(ranked[:keep])
            parents=[c.dna for c in ranked[:keep]]; pop=parents[:]
            while len(pop)<len(population):
                child=self.ev.crossover(self.ev.rng.choice(parents),self.ev.rng.choice(parents)); pop.append(self.ev.mutate(child))
        return sorted(archive,key=lambda c:c.score,reverse=True)
