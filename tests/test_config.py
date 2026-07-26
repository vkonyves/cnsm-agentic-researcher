from pathlib import Path

from cnsm_agentic.config import load_config


def test_load_llm_config() -> None:
    config = load_config(Path("configs/pilot_llm.yaml"))
    assert config.experiment_family == "llm_benchmark"
    assert config.candidate_resources


def test_load_tabular_config() -> None:
    config = load_config(Path("configs/pilot_tabular.yaml"))
    assert config.experiment_family == "tabular_ml"
