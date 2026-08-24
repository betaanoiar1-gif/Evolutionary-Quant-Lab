from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import build_features


@dataclass(frozen=True, slots=True)
class CostModel:
    fee_rate: float = 0.0004
    slippage_bps: float = 2.0
    funding_rate: float = 0.0
    funding_interval_bars: int = 8

    def __post_init__(self) -> None:
        if self.fee_rate < 0 or self.slippage_bps < 0 or self.funding_rate < 0:
            raise ValueError("Costs cannot be negative")
        if self.funding_interval_bars < 1:
            raise ValueError("funding_interval_bars must be positive")

    def execution_price(self, price: float, side: int) -> float:
        if price <= 0 or side not in (-1, 1):
            raise ValueError("price must be positive and side must be +/-1")
        return price * (1 + side * self.slippage_bps / 10000)

    def trade_cost(self, notional: float) -> float:
        return abs(notional) * self.fee_rate


@dataclass(frozen=True, slots=True)
class RiskModel:
    risk_per_trade: float = .01
    max_exposure: float = 1.0

    def __post_init__(self) -> None:
        if not 0 < self.risk_per_trade <= 1 or not 0 < self.max_exposure <= 1:
            raise ValueError("Risk settings must be in (0, 1]")

    def size(self, equity: float, entry: float, stop: float) -> float:
        if equity <= 0 or entry <= 0 or stop <= 0:
            return 0.0
        distance = abs(entry - stop)
        if distance == 0:
            return 0.0
        risk_size = equity * self.risk_per_trade / distance
        exposure_size = equity * self.max_exposure / entry
        return min(risk_size, exposure_size)


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
    funding: float = 0.0
    exposure: float = 0.0
    turnover: float = 0.0
    expectancy: float = 0.0
    trade_returns: tuple[float, ...] = ()


class BacktestEngine:
    def __init__(self, costs: CostModel | None = None, risk: RiskModel | None = None) -> None:
        self.costs = costs or CostModel()
        self.risk = risk or RiskModel()

    @staticmethod
    def _signals(row: pd.Series, dna) -> tuple[bool, bool]:
        long_signal = row.fast > row.slow and row.rsi > dna.rsi_entry
        short_signal = dna.allow_short and row.fast < row.slow and row.rsi < (100 - dna.rsi_entry)
        if dna.use_adx_filter:
            long_signal = long_signal and row.adx >= dna.adx_min
            short_signal = short_signal and row.adx >= dna.adx_min
        if dna.use_volume_filter:
            volume_ok = row.volume >= row.volume_ma * dna.volume_multiplier
            long_signal = long_signal and volume_ok
            short_signal = short_signal and volume_ok
        return bool(long_signal), bool(short_signal)

    def run(self, df: pd.DataFrame, dna, initial: float = 10_000) -> PerformanceReport:
        if initial <= 0:
            raise ValueError("initial capital must be positive")
        x = build_features(df, dna).dropna().reset_index(drop=True)
        if len(x) < 2:
            return PerformanceReport(initial, initial, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, dna.complexity)

        equity = float(initial)
        peak = equity
        max_dd = 0.0
        pos = 0
        entry = stop = target = size = 0.0
        bars_in_trade = 0
        cooldown = 0
        trade_returns: list[float] = []
        gross_win = gross_loss = fees = slippage = funding = turnover = 0.0
        curve = [equity]
        exposure_bars = 0

        def close_position(raw_price: float, direction: int) -> None:
            nonlocal equity, pos, size, fees, slippage, turnover, gross_win, gross_loss
            exit_price = self.costs.execution_price(raw_price, -direction)
            pnl = direction * size * (exit_price - entry)
            fee = self.costs.trade_cost(size * exit_price)
            net = pnl - fee
            starting_equity = equity
            equity += net
            fees += fee
            slippage += abs(exit_price - raw_price) * size
            turnover += abs(size * exit_price)
            gross_win += max(net, 0.0)
            gross_loss += max(-net, 0.0)
            trade_returns.append(net / max(starting_equity, 1e-12))
            pos = 0
            size = 0.0

        for i, row in x.iterrows():
            price = float(row.close)
            if pos:
                exposure_bars += 1
                bars_in_trade += 1
                if i % self.costs.funding_interval_bars == 0:
                    charge = abs(pos * size * price) * self.costs.funding_rate
                    equity -= charge
                    funding += charge
                hit_stop = bool(row.low <= stop) if pos == 1 else bool(row.high >= stop)
                hit_target = bool(row.high >= target) if pos == 1 else bool(row.low <= target)
                timed_exit = dna.max_bars_in_trade > 0 and bars_in_trade >= dna.max_bars_in_trade
                if hit_stop or hit_target or timed_exit:
                    raw_exit = stop if hit_stop else target if hit_target else price
                    close_position(float(raw_exit), pos)
                    cooldown = dna.cooldown_bars
                    bars_in_trade = 0

            if pos == 0:
                if cooldown:
                    cooldown -= 1
                elif i > 0:
                    previous = x.iloc[i - 1]
                    long_signal, short_signal = self._signals(previous, dna)
                    if long_signal or short_signal:
                        direction = 1 if long_signal else -1
                        entry_price = self.costs.execution_price(price, direction)
                        stop_price = entry_price - direction * dna.stop_atr * float(row.atr)
                        target_price = entry_price + direction * dna.take_atr * float(row.atr)
                        position_size = self.risk.size(equity, entry_price, stop_price)
                        if position_size > 0:
                            entry, stop, target, size, pos = entry_price, stop_price, target_price, position_size, direction
                            fee = self.costs.trade_cost(size * entry)
                            equity -= fee
                            fees += fee
                            turnover += abs(size * entry)

            curve.append(equity)
            peak = max(peak, equity)
            max_dd = max(max_dd, (peak - equity) / peak if peak else 0.0)

        if pos:
            close_position(float(x.iloc[-1].close), pos)
            curve.append(equity)

        returns = pd.Series(curve).pct_change().dropna()
        sharpe = float(np.sqrt(252) * returns.mean() / returns.std()) if returns.std() > 0 else 0.0
        negative = returns[returns < 0]
        sortino = float(np.sqrt(252) * returns.mean() / negative.std()) if len(negative) > 1 and negative.std() > 0 else 0.0
        profit_factor = gross_win / gross_loss if gross_loss else (math.inf if gross_win else 0.0)
        trades = len(trade_returns)
        expectancy = float(np.mean(trade_returns)) if trade_returns else 0.0
        return PerformanceReport(
            initial, equity, equity / initial - 1, max_dd, sharpe, sortino, trades,
            sum(v > 0 for v in trade_returns) / trades if trades else 0.0,
            profit_factor, equity - initial, fees, slippage, dna.complexity, funding,
            exposure_bars / max(len(x), 1), turnover, expectancy, tuple(trade_returns),
        )
