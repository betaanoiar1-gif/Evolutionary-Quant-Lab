# Evolutionary Quant Strategy Laboratory

A modular quantitative research laboratory for generating, screening, backtesting, evolving, and rigorously validating systematic trading strategies.

## Architecture

Environment → Data Validation → Temporal Split → Strategy DNA → Feature Library → Fast Screening → Full Backtest → Costs/Risk → Evolution → Ranking/Archive → Walk-Forward/Monte Carlo/Parameter Stability/Regime/Overfitting Checks → isolated OOS → Meta-Strategy.

## Hard research rules

- No random train/validation/OOS shuffling.
- OOS is inaccessible to development and optimization APIs.
- Execution is chronological; signals use completed information and execute on a subsequent bar.
- Fees, slippage and optional perpetual funding are modeled in execution.
- Risk sizing is stop-distance based and exposure constrained.
- Fast screening is separate from the detailed backtester.
- Robustness is required; raw return alone cannot qualify a strategy.
- Complexity is penalized.
- Random seeds are explicit for reproducibility.
- A final OOS result is an evaluation artifact, not an optimization signal.

## Current implementation

Environment engine, OHLCV loader/validator, temporal splitter, Strategy DNA, technical features, fast and full backtesting, fee/slippage/funding model, risk engine, strategy generation, mutation/crossover evolution, robustness checks, isolated OOS gate, ranking, archive, regime/meta-strategy primitives, reporting, tests, and a Colab runner are implemented.

## Run

```bash
pip install -e '.[dev]'
pytest -q
python run_lab.py
```

Google Colab is execution-only; use `notebooks/00_colab_runner.ipynb`.

GitHub Actions runs pytest and compilation checks on pushes and pull requests to `main`.
