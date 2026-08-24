from datetime import datetime, timezone

import pytest

from eqlab.core.config import ExperimentConfig
from eqlab.core.models import DatasetMetadata, TimeRange, ValidationResult


def test_experiment_config_is_deterministic_and_validates_seed() -> None:
    config = ExperimentConfig(seed=123)
    assert config.seed == 123
    assert config.timezone == "UTC"

    with pytest.raises(ValueError):
        ExperimentConfig(seed=-1)


def test_time_range_requires_strict_order() -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 2, tzinfo=timezone.utc)
    assert TimeRange(start, end).start == start

    with pytest.raises(ValueError):
        TimeRange(end, start)


def test_dataset_metadata_validates_required_fields() -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 2, tzinfo=timezone.utc)
    metadata = DatasetMetadata(
        symbol="BTCUSDT",
        timeframe="1h",
        row_count=24,
        time_range=TimeRange(start, end),
        fingerprint="abc123",
    )
    assert metadata.row_count == 24

    with pytest.raises(ValueError):
        DatasetMetadata("", "1h", 24, TimeRange(start, end), "abc123")


def test_validation_result_helpers() -> None:
    success = ValidationResult.success()
    assert success.passed is True
    assert success.errors == ()

    failure = ValidationResult.failure("bad timestamp")
    assert failure.passed is False
    assert failure.errors == ("bad timestamp",)
