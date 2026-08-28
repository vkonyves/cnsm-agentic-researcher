from pathlib import Path


SOURCE = Path(
    "src/cnsm_agentic/autonomous_research/final_pipeline.py"
).read_text(encoding="utf-8")


def test_publication_remediation_occurs_before_final_judge():
    remediation = SOURCE.index(
        "Deterministic publication hygiene remediation"
    )
    final_judge = SOURCE.index(
        "final_report = await run_agent(\n"
        "            FINAL_JUDGE,"
    )

    assert remediation < final_judge


def test_both_audits_are_rerun_after_remediation():
    marker = SOURCE.index(
        "Deterministic publication hygiene remediation"
    )

    tail = SOURCE[marker:]

    assert tail.count(
        "audit_manuscript_publication_sanity("
    ) >= 1

    assert tail.count(
        "audit_manuscript_artifact_references("
    ) >= 1

    assert (
        "publication_sanity_audit_post_remediation.json"
        in tail
    )

    assert (
        "artifact_reference_audit_post_remediation.json"
        in tail
    )


def test_remediation_freezes_scientific_content():
    assert "SCIENTIFIC CONTENT IS FROZEN" in SOURCE
    assert "Do not perform or invent any new" in SOURCE
    assert (
        "deterministic_publication_hygiene_remediation"
        in SOURCE
    )


def test_failed_post_remediation_audits_block_ready():
    assert (
        "MANUSCRIPT_FINAL_DETERMINISTIC_AUDIT_FAILED"
        in SOURCE
    )
