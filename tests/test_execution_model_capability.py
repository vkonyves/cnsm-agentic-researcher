import pytest

from cnsm_agentic.autonomous_research.final_pipeline import (
    create_feasible_experiment_plan,
)


def test_execution_model_capability_parameter_is_required():
    import inspect

    signature = inspect.signature(
        create_feasible_experiment_plan
    )

    assert "available_execution_models" in signature.parameters
