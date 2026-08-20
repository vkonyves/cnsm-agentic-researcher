from cnsm_agentic.autonomous_research.analysis_executors import (
    clear_registered_analysis_executors,
    registered_analysis_families,
)
from cnsm_agentic.autonomous_research.execution_adapters import (
    clear_registered_adapters,
    registered_adapter_families,
)
from cnsm_agentic.autonomous_research.final_pipeline import (
    FinalAutonomousResearchPipeline,
)


def test_pipeline_registers_builtin_execution_and_analysis_components():
    clear_registered_adapters()
    clear_registered_analysis_executors()

    FinalAutonomousResearchPipeline(
        model="gpt-5-mini",
        development_rehearsal=True,
    )

    assert "hosted_netops_gvr_v1" in registered_adapter_families()
    assert (
        "paired_binary_analysis_v1"
        in registered_analysis_families()
    )
