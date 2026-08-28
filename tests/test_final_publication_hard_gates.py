from pathlib import Path


def test_publication_sanity_gate_precedes_artifact_gate():
    source = Path(
        "src/cnsm_agentic/autonomous_research/final_pipeline.py"
    ).read_text(encoding="utf-8")

    sanity = source.index(
        "publication_sanity_audit ="
    )
    artifact = source.index(
        "artifact_reference_audit ="
    )

    assert sanity < artifact

    assert (
        "MANUSCRIPT_FINAL_DETERMINISTIC_AUDIT_FAILED"
        in source
    )

    assert (
        "Deterministic manuscript publication-sanity audit failed."
        in source
    )

    assert (
        "Deterministic manuscript artifact-reference audit failed."
        in source
    )


def test_final_deterministic_gate_blocks_before_final_judge():
    source = Path(
        "src/cnsm_agentic/autonomous_research/final_pipeline.py"
    ).read_text(encoding="utf-8")

    failure_state = source.index(
        "MANUSCRIPT_FINAL_DETERMINISTIC_AUDIT_FAILED"
    )

    final_judge = source.index(
        "final_report = await run_agent(\n"
        "            FINAL_JUDGE,"
    )

    assert failure_state < final_judge
