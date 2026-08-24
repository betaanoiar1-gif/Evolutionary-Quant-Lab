import numpy as np
import pandas as pd
from eqlab.data import DataEngine
from eqlab.dna import StrategyDNA
from eqlab.backtest import BacktestEngine, CostModel, RiskModel
from eqlab.evolution import StrategyGenerator, EvolutionEngine
from eqlab.validation import RobustnessEngine, OOSGate


def market(n=500):
    rng=np.random.default_rng(7); ret=rng.normal(.0003,.01,n); close=100*np.cumprod(1+ret); ts=pd.date_range("2020-01-01",periods=n,freq="h",tz="UTC")
    return pd.DataFrame({"timestamp":ts,"open":close,"high":close*1.005,"low":close*.995,"close":close,"volume":rng.uniform(10,100,n)})

def test_data_validation_and_split():
    df=market(); d=DataEngine(); x=d.normalize(df); s=d.temporal_split(x)
    assert len(s.train)+len(s.validation)+len(s.oos)==len(x)
    assert s.train.timestamp.max()<s.validation.timestamp.min()<s.oos.timestamp.min()

def test_dna_and_backtest_are_deterministic():
    df=market(); dna=StrategyDNA(10,50)
    a=BacktestEngine(CostModel(.0004,2),RiskModel(.01)).run(df,dna)
    b=BacktestEngine(CostModel(.0004,2),RiskModel(.01)).run(df,dna)
    assert a.final==b.final

def test_generator_unique():
    xs=StrategyGenerator(3).generate(30); assert len({x.fingerprint() for x in xs})==30

def test_evolution_preserves_schema():
    dna=StrategyDNA(10,50); e=EvolutionEngine(4)
    assert e.mutate(dna).slow>e.mutate(dna).fast

def test_oos_gate_does_not_optimize():
    df=market(); split=DataEngine().temporal_split(df); dna=StrategyDNA(10,50)
    report=OOSGate().evaluate(split.oos,dna)
    assert report.initial==10000


def test_robustness_runs():
    df=market(); s=DataEngine().temporal_split(df); r=RobustnessEngine().evaluate(s.train,s.validation,StrategyDNA(10,50))
    assert 0<=r.parameter_stability<=1
