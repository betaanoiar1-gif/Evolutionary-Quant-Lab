from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class StrategyDNA:
    """Versioned, serializable strategy genome.

    The genome describes research hypotheses and risk constraints. It does not
    contain execution state, market data, or fitted OOS information.
    """

    fast: int
    slow: int
    rsi_period: int = 14
    rsi_entry: float = 50.0
    stop_atr: float = 2.0
    take_atr: float = 3.0
    allow_short: bool = True
    trend_filter: str = "ema_cross"
    momentum_filter: str = "rsi"
    volatility_filter: str = "atr"
    use_adx_filter: bool = False
    adx_min: float = 20.0
    use_volume_filter: bool = False
    volume_multiplier: float = 1.0
    cooldown_bars: int = 0
    max_bars_in_trade: int = 0
    risk_per_trade: float = 0.01
    max_exposure: float = 1.0
    version: int = 2

    def __post_init__(self) -> None:
        if self.fast < 2 or self.slow <= self.fast:
            raise ValueError("Require 2 <= fast < slow")
        if not 2 <= self.rsi_period <= 100:
            raise ValueError("Invalid RSI period")
        if not 1 <= self.rsi_entry <= 99:
            raise ValueError("Invalid RSI threshold")
        if self.stop_atr <= 0 or self.take_atr <= 0:
            raise ValueError("ATR multipliers must be positive")
        if self.adx_min < 0:
            raise ValueError("ADX threshold cannot be negative")
        if self.volume_multiplier <= 0:
            raise ValueError("Volume multiplier must be positive")
        if self.cooldown_bars < 0 or self.max_bars_in_trade < 0:
            raise ValueError("Bar limits cannot be negative")
        if not 0 < self.risk_per_trade <= 1:
            raise ValueError("risk_per_trade must be in (0, 1]")
        if not 0 < self.max_exposure <= 1:
            raise ValueError("max_exposure must be in (0, 1]")
        if self.version < 1:
            raise ValueError("Invalid DNA version")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "StrategyDNA":
        return cls(**payload)

    def fingerprint(self) -> str:
        raw = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()

    @property
    def complexity(self) -> float:
        components = 1 + int(self.use_adx_filter) + int(self.use_volume_filter)
        parameter_cost = 0.01 * (self.fast + self.slow) + 0.02 * self.rsi_period
        return float(components + parameter_cost + 0.01 * self.cooldown_bars)
