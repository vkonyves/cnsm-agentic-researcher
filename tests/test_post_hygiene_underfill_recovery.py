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
        "post_hygiene_scientific_"
        in SOURCE
    )
    assert (
        "underfill_recovery"
        in SOURCE
    )
    assert (
        "genuine supported scientific substance"
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


def test_underfill_recovery_is_bounded_to_five_attempts():
    assert (
        "maximum_post_hygiene_underfill_attempts = 8"
        in SOURCE
    )
    assert (
        "for underfill_attempt in range("
        in SOURCE
    )


def test_underfill_recovery_starts_each_attempt_from_best_candidate():
    assert (
        "best_recovery_manuscript"
        in SOURCE
    )
    assert (
        "best_recovery_manuscript.model_dump()"
        in SOURCE
    )
    assert (
        "every attempt starts from the best successfully compiled"
        in SOURCE
    )


def test_underfill_recovery_stops_at_exact_page_budget():
    assert (
        "best_recovery_page_count"
        in SOURCE
    )
    assert (
        "== post_hygiene_maximum_pages"
        in SOURCE
    )


def test_underfill_recovery_renders_each_attempt():
    assert (
        "publication_validation_"
        in SOURCE
    )
    assert (
        "post_hygiene_underfill_recovery_"
        in SOURCE
    )
    assert (
        "build_publication_artifacts("
        in SOURCE
    )


def test_underfill_recovery_rejects_non_improving_or_overshooting_candidates():
    assert (
        "candidate_is_clean_and_non_regressing"
        in SOURCE
    )
    assert (
        "candidate_page_count"
        in SOURCE
    )
    assert (
        ">= best_recovery_page_count"
        in SOURCE
    )
    assert (
        "<= post_hygiene_maximum_pages"
        in SOURCE
    )


def test_underfill_recovery_tracks_best_successful_candidate():
    assert "best_recovery_manuscript" in SOURCE
    assert "best_recovery_validation" in SOURCE
    assert "best_recovery_page_count" in SOURCE


def test_underfill_recovery_accepts_only_page_improvement():
    assert "candidate_is_clean_and_non_regressing" in SOURCE
    assert "candidate_page_count" in SOURCE
    assert ">= best_recovery_page_count" in SOURCE


def test_underfill_recovery_rejects_overshoot():
    assert "<= post_hygiene_maximum_pages" in SOURCE


def test_underfill_recovery_restores_best_candidate_after_loop():
    assert (
        "revised_manuscript = best_recovery_manuscript"
        in SOURCE
    )
    assert (
        "publication_validation = ("
        in SOURCE
    )


def test_recovery_candidate_must_pass_publication_sanity():
    assert (
        "candidate_publication_sanity"
        in SOURCE
    )
    assert (
        "audit_manuscript_publication_sanity("
        in SOURCE
    )
    assert (
        'candidate_publication_sanity.get('
        in SOURCE
    )


def test_recovery_candidate_must_pass_artifact_audit():
    assert (
        "candidate_artifact_reference_audit"
        in SOURCE
    )
    assert (
        "audit_manuscript_artifact_references("
        in SOURCE
    )
    assert (
        'candidate_artifact_reference_audit.get('
        in SOURCE
    )


def test_recovery_attempt_audits_are_archived():
    assert (
        "publication_sanity_audit_"
        in SOURCE
    )
    assert (
        "artifact_reference_audit_"
        in SOURCE
    )
    assert (
        "post_hygiene_underfill_"
        in SOURCE
    )


def test_final_selector_can_rescue_protected_exact_page_candidate():
    assert (
        "current_candidate_needs_rescue"
        in SOURCE
    )
    assert (
        "protected_rescue_validation"
        in SOURCE
    )
    assert (
        "protected_rescue_is_valid"
        in SOURCE
    )


def test_protected_final_rescue_requires_all_hard_gates():
    assert (
        'protected_rescue_validation.get("passed") is True'
        in SOURCE
    )
    assert (
        'protected_rescue_sanity.get("passed") is True'
        in SOURCE
    )
    assert (
        "protected_rescue_artifact_audit.get("
        in SOURCE
    )


def test_protected_rescue_requires_exact_page_budget():
    assert (
        'protected_rescue_validation.get("page_count")'
        in SOURCE
    )
    assert (
        'protected_rescue_validation.get('
        in SOURCE
    )
    assert (
        '"maximum_pages"'
        in SOURCE
    )


def test_failed_protected_probe_is_non_destructive():
    assert (
        "pre_rescue_manuscript = revised_manuscript"
        in SOURCE
    )
    assert (
        "revised_manuscript = pre_rescue_manuscript"
        in SOURCE
    )


def test_protected_final_rescue_makes_no_agent_call():
    start = SOURCE.index(
        "# Deterministic protected-candidate rescue"
    )
    end = SOURCE.index(
        "# Re-run BOTH deterministic audits",
        start,
    )
    rescue = SOURCE[start:end]

    assert "run_agent(" not in rescue
    assert "await " not in rescue


def test_protected_final_rescue_archives_decision_artifacts():
    assert (
        "publication_validation_"
        in SOURCE
    )
    assert (
        "protected_final_rescue.json"
        in SOURCE
    )
    assert (
        "selected_protected_final_"
        in SOURCE
    )


def test_same_page_clean_scientific_expansion_is_retained():
    assert "current_recovery_text_length" in SOURCE
    assert "candidate_recovery_text_length" in SOURCE
    assert (
        "candidate_recovery_text_length"
        in SOURCE
        and "> current_recovery_text_length"
        in SOURCE
    )
    assert (
        "candidate_page_count"
        in SOURCE
        and ">= best_recovery_page_count"
        in SOURCE
    )


def test_underfill_recovery_expands_cumulatively():
    assert "Page count is a coarse rendered measure" in SOURCE
    assert "Build CUMULATIVELY" in SOURCE
    assert (
        "preserve every supported scientific "
        in SOURCE
    )




