"""Reproducible smoke run of the complete development-to-OOS pipeline."""
import numpy as np
import pandas as pd
from eqlab.environment import EnvironmentEngine
from eqlab.data import DataEngine
from eqlab.pipeline import Laboratory


def synthetic_data(n=3000, seed=42):
    rng=np.random.default_rng(seed); ret=rng.normal(0.0001,0.01,n); close=100*np.cumprod(1+ret); open_=close*(1+rng.normal(0,.001,n)); high=np.maximum(open_,close)*(1+np.abs(rng.normal(0,.004,n))); low=np.minimum(open_,close)*(1-np.abs(rng.normal(0,.004,n)))
    return pd.DataFrame({"timestamp":pd.date_range("2020-01-01",periods=n,freq="h",tz="UTC"),"open":open_,"high":high,"low":low,"close":close,"volume":rng.uniform(100,1000,n)})


def main():
    EnvironmentEngine(42).validate(); data=DataEngine().normalize(synthetic_data()); split=DataEngine().temporal_split(data); lab=Laboratory(42)
    population=lab.gen.generate(100); survivors=lab.screen(split.train,population); evolved=lab.evolve(split.train,survivors or population,generations=3,elite=min(10,max(1,len(survivors or population)))); robust=lab.robust_rank(split.train,split.validation,evolved,top_n=min(10,len(evolved)))
    passed=[x for x in robust if x.robustness.passed]
    if not passed: print({"generated":len(population),"fast_survivors":len(survivors),"robust":0,"status":"NO_ROBUST_STRATEGY"}); return
    winner=passed[0].candidate.dna; oos=lab.final_evaluate(split.oos,winner)
    print({"generated":len(population),"fast_survivors":len(survivors),"robust":len(passed),"oos_return":oos.total_return,"oos_drawdown":oos.max_drawdown,"oos_trades":oos.trades})

if __name__ == "__main__": main()
