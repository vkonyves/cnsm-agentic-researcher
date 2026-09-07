from pathlib import Path


SCRIPT = Path(
    "scripts/recover_final_manuscript.py"
)


def test_publication_recovery_does_not_reenter_science_pipeline():
    text = SCRIPT.read_text(
        encoding="utf-8"
    )

    forbidden = (
        "FinalAutonomousResearchPipeline(",
        "AutonomousDiscoveryPipeline(",
        "EXPERIMENT_PLANNER",
        "ANALYSIS_PLANNER",
        "PREREGISTRATION_AGENT",
        "DESIGN_REPAIR_AGENT",
        "resolve_adapter(",
        "resolve_analysis_executor(",
    )

    for token in forbidden:
        assert token not in text


def test_publication_recovery_explicitly_preserves_science():
    text = SCRIPT.read_text(
        encoding="utf-8"
    )

    required = (
        '"scientific_execution_reused": True',
        '"scientific_execution_rerun": False',
        '"experiment_rerun": False',
        '"analysis_rerun": False',
        '"recovery_scope"',
        "cited_record_ids changed",
    )

    for token in required:
        assert token in text
