# Evolutionary Quant Strategy Laboratory

A modular quantitative research laboratory for generating, backtesting, evolving, and rigorously validating systematic trading strategies.

## Principles

- Modular, independently testable engines.
- No look-ahead bias.
- Strict Train / Validation / OOS separation.
- OOS is never used for development or parameter optimization.
- Realistic fees, slippage, and execution modeling.
- Robustness over raw return.
- Reproducible experiments.

## Implemented

Environment, validated OHLCV data ingestion, temporal splitting, Strategy DNA, technical features, realistic cost/risk models, deterministic backtesting, evolutionary search, robustness validation, isolated OOS evaluation, ranking, archive, and regime/meta-strategy primitives.

## Verification

GitHub Actions runs pytest and Python compilation checks on every push and pull request to `main`.

## Status

Core research laboratory implementation in progress.
