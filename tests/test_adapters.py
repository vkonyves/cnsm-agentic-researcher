from cnsm_agentic.adapters import get_adapter


def test_known_adapters() -> None:
    assert get_adapter("llm_benchmark")
    assert get_adapter("tabular_ml")
