from pathlib import Path


SOURCE = Path(
    "src/cnsm_agentic/autonomous_research/final_pipeline.py"
).read_text(encoding="utf-8")


def test_underfill_recovery_occurs_after_hygiene_remediation():
    hygiene = SOURCE.index(
        "Deterministic publication hygiene remediation"
    )
    recovery = SOURCE.index(
        "Post-hygiene scientific underfill recovery"
    )

    assert hygiene < recovery


def test_underfill_recovery_requires_successful_compile_and_underfill():
    assert (
        'publication_validation.get("compile_status")'
        in SOURCE
    )
    assert (
        "post_hygiene_page_count"
        in SOURCE
    )
    assert (
        "post_hygiene_maximum_pages"
        in SOURCE
    )
    assert (
        "post_hygiene_page_count"
        in SOURCE
        and "< post_hygiene_maximum_pages"
        in SOURCE
    )


def test_underfill_recovery_freezes_science():
    assert (
        "SCIENTIFIC CONTENT IS FROZEN."
        in SOURCE
    )
    assert (
        "post_hygiene_scientific_underfill_recovery"
        in SOURCE
    )
    assert (
        "genuine scientific explanation and synthesis"
        in SOURCE
    )
    assert (
        "Do not add generic filler"
        in SOURCE
    )


def test_underfill_recovery_uses_bounded_scientific_context():
    assert (
        'manuscript_revision_context['
        in SOURCE
    )
    assert (
        '"evidence_synthesis"'
        in SOURCE
    )
    assert (
        '"manuscript_evidence_bundle"'
        in SOURCE
    )
    assert (
        '"verified_records"'
        in SOURCE
    )


def test_underfill_recovery_rerenders_before_final_audits():
    recovery = SOURCE.index(
        "Post-hygiene scientific underfill recovery"
    )

    rerender_marker = SOURCE.index(
        "post_hygiene_underfill_recovery.json",
        recovery,
    )

    final_audit_comment = SOURCE.index(
        "post-remediation candidate",
        recovery,
    )

    assert recovery < rerender_marker < final_audit_comment
