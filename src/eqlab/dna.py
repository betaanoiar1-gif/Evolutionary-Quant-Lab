from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Condition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    feature: str = Field(min_length=1, max_length=100)
    operator: Literal[">", ">=", "<", "<=", "==", "!=", "crosses_above", "crosses_below"]
    value: float | str


class EntryLogic(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    long: tuple[Condition, ...] = ()
    short: tuple[Condition, ...] = ()
    mode: Literal["all", "any"] = "all"


class ExitLogic(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    stop_loss_atr: float | None = Field(default=None, gt=0)
    take_profit_atr: float | None = Field(default=None, gt=0)
    trailing_atr: float | None = Field(default=None, gt=0)
    max_bars: int | None = Field(default=None, gt=0)


class RiskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    risk_per_trade: float = Field(gt=0, le=0.1)
    max_position_fraction: float = Field(gt=0, le=1)


class StrategyDNA(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    strategy_id: str = Field(min_length=1, max_length=128)
    timeframe: str = Field(min_length=1, max_length=20)
    features: tuple[str, ...] = ()
    entry: EntryLogic
    exit: ExitLogic
    risk: RiskConfig
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("features")
    @classmethod
    def unique_features(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("Duplicate features are not allowed")
        return value

    def canonical_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    def to_json(self) -> str:
        return json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))

    def fingerprint(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @property
    def complexity(self) -> int:
        conditions = len(self.entry.long) + len(self.entry.short)
        exits = sum(x is not None for x in (
            self.exit.stop_loss_atr,
            self.exit.take_profit_atr,
            self.exit.trailing_atr,
            self.exit.max_bars,
        ))
        return len(self.features) + conditions + exits


def strategy_from_json(payload: str) -> StrategyDNA:
    return StrategyDNA.model_validate_json(payload)
