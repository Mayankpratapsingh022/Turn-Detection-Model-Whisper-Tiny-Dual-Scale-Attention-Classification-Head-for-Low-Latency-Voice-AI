from pathlib import Path

import pytest
from pydantic import ValidationError

from turn_detector.config import AppConfig, DataConfig


def test_default_config_is_valid() -> None:
    config = AppConfig.from_yaml(Path("configs/default.yaml"))
    assert config.data.languages == ("hin", "eng")
    assert config.train.gradient_accumulation_steps == 8
    assert config.model.architecture == "dual_scale"
    assert config.train.hindi_sampling_fraction == 0.50
    assert config.tracking.enabled is False


def test_runpod_config_enables_tracking() -> None:
    config = AppConfig.from_yaml(Path("configs/runpod.yaml"))
    assert config.tracking.enabled is True
    assert config.tracking.project == "hinglish-turn-detector"
    assert config.train.mixed_precision == "bf16"


def test_experiment_config_inherits_default() -> None:
    config = AppConfig.from_yaml(Path("configs/experiments/e2_global.yaml"))
    assert config.model.architecture == "global"
    assert config.model.sample_rate == 16_000
    assert config.train.focused_sampling is False
    assert config.train.include_causal_pauses is False


def test_other_training_languages_are_rejected() -> None:
    with pytest.raises(ValidationError, match="restricted"):
        DataConfig(languages=("hin", "eng", "fra"))
