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


_LEGACY_KEYS = {
    "fast", "slow", "rsi_period", "rsi_entry", "stop_atr", "take_atr",
    "allow_short", "trend_filter", "momentum_filter", "volatility_filter",
    "use_adx_filter", "adx_min", "use_volume_filter", "volume_multiplier",
    "cooldown_bars", "max_bars_in_trade", "risk_per_trade", "max_exposure", "version",
}


def _legacy_payload(args: tuple[Any, ...], values: dict[str, Any]) -> dict[str, Any]:
    if args:
        if len(args) != 2:
            raise TypeError("Legacy StrategyDNA positional form requires exactly fast and slow")
        if "fast" in values or "slow" in values:
            raise TypeError("fast/slow cannot be supplied both positionally and by keyword")
        values = {"fast": args[0], "slow": args[1], **values}
    if not (set(values) & _LEGACY_KEYS):
        return values
    if "fast" not in values or "slow" not in values:
        raise TypeError("Legacy StrategyDNA requires fast and slow")
    return values


class StrategyDNA(BaseModel):
    """Strict versioned strategy genome with backward-compatible engine accessors."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    strategy_id: str = Field(min_length=1, max_length=128)
    timeframe: str = Field(min_length=1, max_length=20)
    features: tuple[str, ...] = ()
    entry: EntryLogic
    exit: ExitLogic
    risk: RiskConfig
    metadata: dict[str, str] = Field(default_factory=dict)

    def __init__(self, *args: Any, **data: Any) -> None:
        values = _legacy_payload(args, data)
        if set(values) & _LEGACY_KEYS:
            fast = int(values.get("fast", 10))
            slow = int(values.get("slow", 50))
            rsi_period = int(values.get("rsi_period", 14))
            rsi_entry = float(values.get("rsi_entry", 50.0))
            stop_atr = float(values.get("stop_atr", 2.0))
            take_atr = float(values.get("take_atr", 3.0))
            allow_short = bool(values.get("allow_short", True))
            trend_filter = str(values.get("trend_filter", "ema_cross"))
            momentum_filter = str(values.get("momentum_filter", "rsi"))
            volatility_filter = str(values.get("volatility_filter", "atr"))
            use_adx = bool(values.get("use_adx_filter", False))
            adx_min = float(values.get("adx_min", 20.0))
            use_volume = bool(values.get("use_volume_filter", False))
            volume_multiplier = float(values.get("volume_multiplier", 1.0))
            cooldown = int(values.get("cooldown_bars", 0))
            max_bars = int(values.get("max_bars_in_trade", 0))
            risk_per_trade = float(values.get("risk_per_trade", 0.01))
            max_exposure = float(values.get("max_exposure", 1.0))
            version = int(values.get("version", 2))
            if fast < 2 or slow <= fast:
                raise ValueError("Require 2 <= fast < slow")
            if not 2 <= rsi_period <= 100:
                raise ValueError("Invalid RSI period")
            if not 1 <= rsi_entry <= 99:
                raise ValueError("Invalid RSI threshold")
            if cooldown < 0 or max_bars < 0:
                raise ValueError("Bar limits cannot be negative")
            if not 0 < risk_per_trade <= 0.1:
                raise ValueError("risk_per_trade must be in (0, 0.1]")
            if not 0 < max_exposure <= 1:
                raise ValueError("max_exposure must be in (0, 1]")
            legacy = {
                "fast": fast, "slow": slow, "rsi_period": rsi_period, "rsi_entry": rsi_entry,
                "stop_atr": stop_atr, "take_atr": take_atr, "allow_short": allow_short,
                "trend_filter": trend_filter, "momentum_filter": momentum_filter,
                "volatility_filter": volatility_filter, "use_adx_filter": use_adx,
                "adx_min": adx_min, "use_volume_filter": use_volume,
                "volume_multiplier": volume_multiplier, "cooldown_bars": cooldown,
                "max_bars_in_trade": max_bars, "risk_per_trade": risk_per_trade,
                "max_exposure": max_exposure, "version": version,
            }
            raw_id = json.dumps(legacy, sort_keys=True, separators=(",", ":"))
            strategy_id = "legacy-" + hashlib.sha256(raw_id.encode()).hexdigest()[:24]
            features = [f"{trend_filter}:{fast}:{slow}", f"{momentum_filter}:{rsi_period}", volatility_filter]
            if use_adx:
                features.append(f"adx:{adx_min:g}")
            if use_volume:
                features.append(f"volume:{volume_multiplier:g}")
            values = {
                "strategy_id": strategy_id,
                "timeframe": "1h",
                "features": tuple(features),
                "entry": EntryLogic(
                    long=(Condition(feature=f"{trend_filter}_fast", operator=">", value=f"{trend_filter}_slow"),),
                    short=(Condition(feature=f"{trend_filter}_fast", operator="<", value=f"{trend_filter}_slow"),) if allow_short else (),
                ),
                "exit": ExitLogic(stop_loss_atr=stop_atr, take_profit_atr=take_atr, max_bars=max_bars or None),
                "risk": RiskConfig(risk_per_trade=risk_per_trade, max_position_fraction=max_exposure),
                "metadata": {"_compat_legacy": "1", "_legacy": json.dumps(legacy, sort_keys=True, separators=(",", ":"))},
            }
        super().__init__(**values)

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

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StrategyDNA":
        if set(payload) & _LEGACY_KEYS:
            return cls(**payload)
        return cls.model_validate(payload)

    def to_dict(self) -> dict[str, Any]:
        """Legacy engine serialization; new schema serialization is canonical_dict()."""
        if self.metadata.get("_compat_legacy") == "1":
            return json.loads(self.metadata["_legacy"])
        return self.canonical_dict()

    def fingerprint(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @property
    def _legacy(self) -> dict[str, Any]:
        if self.metadata.get("_compat_legacy") == "1":
            return json.loads(self.metadata["_legacy"])
        return {
            "fast": 2, "slow": 3, "rsi_period": 14, "rsi_entry": 50.0,
            "stop_atr": self.exit.stop_loss_atr or 2.0,
            "take_atr": self.exit.take_profit_atr or 3.0,
            "allow_short": bool(self.entry.short), "trend_filter": "schema",
            "momentum_filter": "rsi", "volatility_filter": "atr", "use_adx_filter": False,
            "adx_min": 20.0, "use_volume_filter": False, "volume_multiplier": 1.0,
            "cooldown_bars": 0, "max_bars_in_trade": self.exit.max_bars or 0,
            "risk_per_trade": self.risk.risk_per_trade,
            "max_exposure": self.risk.max_position_fraction, "version": self.schema_version,
        }

    def __getattr__(self, name: str) -> Any:
        if name in _LEGACY_KEYS:
            return self._legacy.get(name)
        raise AttributeError(name)

    @property
    def complexity(self) -> float:
        if self.metadata.get("_compat_legacy") == "1":
            x = self._legacy
            components = 1 + int(x["use_adx_filter"]) + int(x["use_volume_filter"])
            parameter_cost = 0.01 * (x["fast"] + x["slow"]) + 0.02 * x["rsi_period"]
            return float(components + parameter_cost + 0.01 * x["cooldown_bars"])
        conditions = len(self.entry.long) + len(self.entry.short)
        exits = sum(x is not None for x in (
            self.exit.stop_loss_atr, self.exit.take_profit_atr,
            self.exit.trailing_atr, self.exit.max_bars,
        ))
        return len(self.features) + conditions + exits


def strategy_from_json(payload: str) -> StrategyDNA:
    return StrategyDNA.model_validate_json(payload)
