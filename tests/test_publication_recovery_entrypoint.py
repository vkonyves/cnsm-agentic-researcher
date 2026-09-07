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


def test_publication_recovery_copies_frozen_provenance():
    text = SCRIPT.read_text(
        encoding="utf-8"
    )

    required = (
        '"freeze_manifest.json"',
        '"master_prompt.sha256"',
        '"master_prompt.txt"',
        '"intervention_policy.json"',
        '"capability_manifest.json"',
        '"paper_run_constraints.json"',
        '"execution_manifest.json"',
        '"results.json"',
        '"contamination_summary.csv"',
        "recovery_provenance_dir",
        "recovery_execution_dir",
        "recovery_analysis_dir",
    )

    for token in required:
        assert token in text


def test_publication_recovery_loads_repository_environment():
    text = SCRIPT.read_text(
        encoding="utf-8"
    )

    assert "from dotenv import load_dotenv" in text
    assert 'ENV_PATH = REPO_ROOT / ".env"' in text
    assert "load_dotenv(ENV_PATH)" in text
