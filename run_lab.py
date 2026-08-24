"""Small smoke-run entry point for local/Colab execution."""
import numpy as np
import pandas as pd
from eqlab.environment import EnvironmentEngine
from eqlab.data import DataEngine
from eqlab.pipeline import Laboratory


def synthetic_data(n=3000, seed=42):
    rng=np.random.default_rng(seed)
    ret=rng.normal(0.0001,0.01,n)
    close=100*np.cumprod(1+ret)
    return pd.DataFrame({
        "timestamp":pd.date_range("2020-01-01",periods=n,freq="h",tz="UTC"),
        "open":close*(1+rng.normal(0,.001,n)),
        "high":close*(1+np.abs(rng.normal(0,.004,n))),
        "low":close*(1-np.abs(rng.normal(0,.004,n))),
        "close":close,
        "volume":rng.uniform(100,1000,n),
    })


def main():
    EnvironmentEngine(42).validate()
    data=DataEngine().normalize(synthetic_data())
    split=DataEngine().temporal_split(data)
    lab=Laboratory(42)
    population=lab.gen.generate(100)
    survivors=lab.screen(split.train,population)
    ranked=lab.full_rank(split.train,survivors)
    robust=lab.robustness(split.train,split.validation,ranked,top_n=min(10,len(ranked)))
    passed=[(c,r) for c,r in robust if r.passed]
    if passed:
        winner=max(passed,key=lambda x:x[1].walk_forward_mean)[0]
        oos=lab.final_evaluate(split.oos,winner.dna)
        print({"generated":len(population),"survivors":len(survivors),"robust":len(passed),"oos_return":oos.total_return,"oos_drawdown":oos.max_drawdown})
    else:
        print({"generated":len(population),"survivors":len(survivors),"robust":0,"status":"NO_ROBUST_STRATEGY"})

if __name__ == "__main__": main()
