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
        "MANUSCRIPT_PUBLICATION_SANITY_AUDIT_FAILED"
        in source
    )

    assert (
        "MANUSCRIPT_ARTIFACT_REFERENCE_AUDIT_FAILED"
        in source
    )
