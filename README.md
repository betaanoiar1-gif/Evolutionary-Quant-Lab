# Evolutionary Quant Strategy Laboratory

A modular quantitative research laboratory for generating, screening, backtesting, evolving, and rigorously validating systematic trading strategies.

## Research architecture

Environment → Data Validation → Temporal Split → Strategy DNA → Feature Registry → Fast Screening → Full Event-Driven Backtest → Costs/Risk → Evolution → Robust Ranking/Archive → Walk-Forward/Monte Carlo/Parameter Stability/Regime/Overfitting Checks → isolated OOS → Meta-Strategy.

## Hard research rules

- No random train/validation/OOS shuffling.
- OOS is inaccessible to development, mutation, crossover, ranking, and parameter optimization APIs.
- Signals use completed information and execute on a subsequent bar.
- Fees, slippage, and optional perpetual funding are modeled at execution level.
- Position sizing is stop-distance based and constrained by per-strategy risk and exposure DNA.
- Fast screening is deliberately separate from the detailed execution simulator.
- Raw return alone cannot qualify a strategy.
- Complexity and multiple-hypothesis pressure are penalized.
- Robustness requires walk-forward evidence, parameter stability, regime consistency, and Monte Carlo survival.
- Meta-strategy fitting is development-only; it does not score candidates on the observations it is asked to trade.
- Random seeds are explicit for reproducibility.
- Final OOS is an evaluation artifact, never an optimization signal.

## Implemented foundation

The repository currently contains the environment/data layer, strict OHLCV validation, temporal isolation, versioned Strategy DNA, a feature registry with trend/momentum/volatility/volume features, fast screening, chronological event-driven backtesting, fee/slippage/funding accounting, stop-distance risk sizing, evolutionary generation/mutation/crossover, diversity-aware search, robustness analysis, multiple-testing overfit penalty, ranking, archive, regime/meta primitives, reporting, regression tests, and a Colab runner.

## Local/Colab verification

```bash
pip install -e '.[dev]'
pytest -q
python -m compileall -q src
python run_lab.py
```

Google Colab is execution-only. Clone the repository, install it in editable mode, and run the supplied notebook/runner.

GitHub Actions automatically runs pytest and compilation checks on pushes and pull requests to `main`.
