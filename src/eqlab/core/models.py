"""Foundational immutable domain models.

These models intentionally contain no trading logic. They define stable contracts
that later engines can depend on without coupling to implementation details.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Mapping


class DatasetSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    OOS = "oos"


@dataclass(frozen=True, slots=True)
class TimeRange:
    """Inclusive temporal boundaries for a dataset segment."""

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if self.start >= self.end:
            raise ValueError("TimeRange.start must be earlier than TimeRange.end")


@dataclass(frozen=True, slots=True)
class DatasetMetadata:
    """Metadata describing a validated market dataset."""

    symbol: str
    timeframe: str
    row_count: int
    time_range: TimeRange
    fingerprint: str
    extra: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError("symbol must not be empty")
        if not self.timeframe.strip():
            raise ValueError("timeframe must not be empty")
        if self.row_count < 1:
            raise ValueError("row_count must be positive")
        if not self.fingerprint.strip():
            raise ValueError("fingerprint must not be empty")


@dataclass(frozen=True, slots=True)
class SplitBoundary:
    """A deterministic boundary between two temporal dataset segments."""

    train_end: datetime
    validation_end: datetime
    oos_end: datetime

    def __post_init__(self) -> None:
        if not self.train_end < self.validation_end < self.oos_end:
            raise ValueError("Split boundaries must be strictly chronological")


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Generic structured validation result."""

    passed: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @classmethod
    def success(cls, warnings: tuple[str, ...] = ()) -> "ValidationResult":
        return cls(passed=True, warnings=warnings)

    @classmethod
    def failure(cls, *errors: str) -> "ValidationResult":
        if not errors:
            raise ValueError("At least one validation error is required")
        return cls(passed=False, errors=tuple(errors))
