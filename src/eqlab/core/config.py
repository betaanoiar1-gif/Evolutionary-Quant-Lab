"""Configuration primitives for reproducible experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Minimal global experiment configuration.

    Later engines will extend this through dedicated configuration objects rather
    than adding mutable global settings.
    """

    seed: int = 42
    timezone: str = "UTC"

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not self.timezone.strip():
            raise ValueError("timezone must not be empty")
