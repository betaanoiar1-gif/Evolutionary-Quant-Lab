import json

import pytest
from pydantic import ValidationError

from eqlab.dna import (
    Condition,
    EntryLogic,
    ExitLogic,
    RiskConfig,
    StrategyDNA,
    strategy_from_json,
)


def sample() -> StrategyDNA:
    return StrategyDNA(
        strategy_id="test-001",
        timeframe="1h",
        features=("ema_50", "rsi_14"),
        entry=EntryLogic(
            long=(Condition(feature="ema_50", operator=">", value="ema_200"),),
            short=(Condition(feature="ema_50", operator="<", value="ema_200"),),
        ),
        exit=ExitLogic(stop_loss_atr=2.0, take_profit_atr=3.0),
        risk=RiskConfig(risk_per_trade=0.01, max_position_fraction=0.25),
    )


def test_valid_dna_and_complexity():
    dna = sample()
    assert dna.schema_version == 1
    assert dna.complexity == 6


def test_round_trip_json():
    dna = sample()
    restored = strategy_from_json(dna.to_json())
    assert restored == dna
    assert json.loads(dna.to_json()) == dna.canonical_dict()


def test_fingerprint_is_deterministic():
    assert sample().fingerprint() == sample().fingerprint()
    assert len(sample().fingerprint()) == 64


def test_duplicate_features_rejected():
    with pytest.raises(ValidationError, match="Duplicate features"):
        sample.model_construct() if False else StrategyDNA(
            strategy_id="x",
            timeframe="1h",
            features=("rsi_14", "rsi_14"),
            entry=EntryLogic(),
            exit=ExitLogic(),
            risk=RiskConfig(risk_per_trade=0.01, max_position_fraction=0.5),
        )


def test_unknown_fields_rejected():
    payload = sample().model_dump()
    payload["oos_score"] = 99.0
    with pytest.raises(ValidationError):
        StrategyDNA.model_validate(payload)


def test_risk_bounds_rejected():
    with pytest.raises(ValidationError):
        RiskConfig(risk_per_trade=0.2, max_position_fraction=0.5)
    with pytest.raises(ValidationError):
        RiskConfig(risk_per_trade=0.01, max_position_fraction=1.1)
