from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class StrategyDNA:
    fast: int
    slow: int
    rsi_period: int = 14
    rsi_entry: float = 50.0
    stop_atr: float = 2.0
    take_atr: float = 3.0
    allow_short: bool = True
    version: int = 1

    def __post_init__(self) -> None:
        if self.fast < 2 or self.slow <= self.fast:
            raise ValueError("Require 2 <= fast < slow")
        if not 2 <= self.rsi_period <= 100:
            raise ValueError("Invalid RSI period")
        if not 1 <= self.rsi_entry <= 99:
            raise ValueError("Invalid RSI threshold")
        if self.stop_atr <= 0 or self.take_atr <= 0:
            raise ValueError("ATR multipliers must be positive")

    def to_dict(self) -> dict:
        return asdict(self)

    def fingerprint(self) -> str:
        raw = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()

    @property
    def complexity(self) -> float:
        return 1.0 + 0.01 * self.slow + 0.02 * self.rsi_period
