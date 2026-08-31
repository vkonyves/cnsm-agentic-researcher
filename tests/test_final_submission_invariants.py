from pathlib import Path


PIPELINE = Path(
    "src/cnsm_agentic/autonomous_research/final_pipeline.py"
)


def source() -> str:
    return PIPELINE.read_text(encoding="utf-8")


def test_exact_page_submission_is_hard_final_invariant():
    text = source()

    assert (
        'publication_validation.get("page_count")\n'
        '            != publication_validation.get("maximum_pages")'
        in text
    )

    assert "Final manuscript is not an exact-page submission" in text


def test_empty_citation_set_does_not_expand_to_all_records():
    text = source()

    assert "if not cited_record_ids:" in text
    assert (
        "Never interpret an empty citation"
        in text
    )


def test_best_page_selection_handles_none_page_count():
    current_page_count = None
    best_page_count = 5

    comparable_current_page_count = (
        current_page_count
        if isinstance(current_page_count, int)
        else -1
    )

    assert best_page_count > comparable_current_page_count


def test_publication_hygiene_removes_non_bmp_unicode_for_pdflatex():
    source = Path(
        "src/cnsm_agentic/autonomous_research/final_pipeline.py"
    ).read_text(encoding="utf-8")

    assert r'[\U00010000-\U0010FFFF]' in source


def test_adapter_does_not_claim_deterministic_model_sampling():
    source = Path(
        "src/cnsm_agentic/autonomous_research/"
        "hosted_netops_adapter.py"
    ).read_text(encoding="utf-8")

    assert '"guarantees_deterministic_model_sampling": False' in source
    assert '"generation_semantics": (' in source
    assert '"shared_initial_candidate"' in source


def test_preregistration_rejects_unsupported_sampling_determinism():
    source = Path(
        "src/cnsm_agentic/autonomous_research/final_pipeline.py"
    ).read_text(encoding="utf-8")

    assert "deterministic_sampling_claims = (" in source
    assert '"temperature=0"' in source
    assert '"guarantees_deterministic_model_sampling"' in source
    assert "shared_initial_candidate" in source


def test_preregistration_rejects_unfrozen_holdout_seed_claim():
    source = Path(
        "src/cnsm_agentic/autonomous_research/final_pipeline.py"
    ).read_text(encoding="utf-8")

    assert '"holdout" in prereg_text' in source
    assert '"fixed seed" in prereg_text' in source
    assert 'adapter_contract.get("holdout_selection_seed") is None' in source



def test_publication_audit_normalizes_allowbreak_inside_sha256():
    source = Path(
        "src/cnsm_agentic/autonomous_research/final_pipeline.py"
    ).read_text(encoding="utf-8")

    assert "hash_scan_tex" in source
    assert r'\\allowbreak\{\}' in source


def test_publication_audit_rejects_raw_bracketed_doi_citations():
    source = Path(
        "src/cnsm_agentic/autonomous_research/final_pipeline.py"
    ).read_text(encoding="utf-8")

    assert "bracketed_doi_citations" in source
    assert "bracketed_doi_citation_count" in source
    assert "standard IEEE" in source


def test_shared_initial_wording_policy_is_applied_to_author_and_reviser():
    source = Path(
        "src/cnsm_agentic/autonomous_research/final_agents.py"
    ).read_text(encoding="utf-8")

    assert source.count(
        "baseline single-shot/initial validity"
    ) == 2
    assert source.count(
        "final guarded-pipeline validity"
    ) >= 2


def test_disclosure_condensation_policy_is_applied_twice():
    source = Path(
        "src/cnsm_agentic/autonomous_research/final_agents.py"
    ).read_text(encoding="utf-8")

    assert source.count(
        "Keep the Disclosure Statement concise and publication-facing."
    ) == 2
    assert source.count(
        "provider-call totals, stage counts, sampling-analysis discussion"
    ) == 2
    assert source.count(
        "second Methods or Results section"
    ) >= 2


def test_peer_reviewer_checks_citation_semantics():
    source = Path(
        "src/cnsm_agentic/autonomous_research/final_agents.py"
    ).read_text(encoding="utf-8")

    assert (
        "Verify citation semantics, not merely reference existence."
        in source
    )



def test_disclosure_has_dedicated_master_prompt_preserving_sanitizer():
    source = Path(
        "src/cnsm_agentic/autonomous_research/final_pipeline.py"
    ).read_text(encoding="utf-8")

    assert "def sanitize_disclosure_text" in source
    assert 'master_prompt_path = "provenance/master_prompt.txt"' in source
    assert "master_prompt_sha" in source
    assert 'key == "disclosure_statement"' in source

    # Disclosure must no longer be exempted wholesale from publication
    # metadata sanitation.
    assert '                preserve_field = key in {\n                    "cited_record_ids",\n                    "disclosure_statement",\n                }\n' not in source


def test_disclosure_sanitizer_removes_secondary_full_sha256_values():
    source = Path(
        "src/cnsm_agentic/autonomous_research/final_pipeline.py"
    ).read_text(encoding="utf-8")

    assert "secondary machine-provenance" in source
    assert "MASTERPROMPTSHA256TOKEN" in source
    assert "MASTERPROMPTPATHTOKEN" in source



def test_ordinary_manuscript_prose_removes_full_sha256_digests():
    source = Path(
        "src/cnsm_agentic/autonomous_research/final_pipeline.py"
    ).read_text(encoding="utf-8")

    assert (
        "Full cryptographic digests are machine provenance"
        in source
    )
    assert (
        "master-prompt digest in Disclosure is protected"
        in source
    )



def test_final_peer_review_closure_uses_authoritative_manuscript():
    source = Path(
        "src/cnsm_agentic/autonomous_research/final_pipeline.py"
    ).read_text(encoding="utf-8")

    assert "review_final_closure.json" in source
    assert "final_authoritative_closure_review" in source
    assert "final_authoritative_closure_instruction" in source
    assert "latest_peer_review = final_closure_review" in source

    closure_pos = source.index(
        "final_closure_review = await run_agent("
    )
    final_judge_pos = source.index(
        "final_report = await run_agent(",
        closure_pos,
    )

    assert closure_pos < final_judge_pos


def test_final_closure_receives_final_deterministic_audits():
    source = Path(
        "src/cnsm_agentic/autonomous_research/final_pipeline.py"
    ).read_text(encoding="utf-8")

    closure_start = source.index(
        "final_closure_review = await run_agent("
    )
    closure_end = source.index(
        "expected_type=PeerReviewReport",
        closure_start,
    )
    closure = source[closure_start:closure_end]

    assert '"publication_validation"' in closure
    assert '"publication_sanity_audit"' in closure
    assert '"artifact_reference_audit"' in closure
    assert '"manuscript"' in closure
    assert "revised_manuscript.model_dump()" in closure


def test_final_closure_is_after_deterministic_failure_return():
    source = Path(
        "src/cnsm_agentic/autonomous_research/final_pipeline.py"
    ).read_text(encoding="utf-8")

    deterministic_failure = source.index(
        "MANUSCRIPT_FINAL_DETERMINISTIC_AUDIT_FAILED"
    )
    closure = source.index(
        "final_closure_review = await run_agent("
    )

    assert deterministic_failure < closure



def test_shared_initial_policy_forbids_claimed_prompting_effect():
    source = Path(
        "src/cnsm_agentic/autonomous_research/final_agents.py"
    ).read_text(encoding="utf-8")

    assert source.count(
        "executed paired experiment measured an independent effect of"
    ) >= 2
    assert source.count(
        "without repair versus that candidate after the permitted"
    ) >= 2
    assert (
        "independent effect of constrained prompting"
        in source
    )


def test_final_agents_require_full_ieee_bibliography():
    source = Path(
        "src/cnsm_agentic/autonomous_research/final_agents.py"
    ).read_text(encoding="utf-8")

    assert source.count(
        "not an acceptable final bibliography"
    ) >= 2
    assert (
        "Numbered bare DOI URLs, arXiv URLs, OpenAlex identifiers"
        in source
    )


def test_disclosure_cannot_repeat_methods_and_results_accounting():
    source = Path(
        "src/cnsm_agentic/autonomous_research/final_agents.py"
    ).read_text(encoding="utf-8")

    assert source.count(
        "must not function as a second Methods or Results section"
    ) >= 2
    assert (
        "It must not become a second Methods or Results"
        in source
    )


def test_unmeasured_operational_benefits_must_be_qualified():
    source = Path(
        "src/cnsm_agentic/autonomous_research/final_agents.py"
    ).read_text(encoding="utf-8")

    assert source.count(
        "Do not present unmeasured operational benefits as empirical findings."
    ) >= 2
    assert (
        "Reject empirical-sounding claims of reduced human triage time"
        in source
    )



def test_post_hygiene_recovery_can_seed_from_sanitized_protected_checkpoint():
    source = Path(
        "src/cnsm_agentic/autonomous_research/final_pipeline.py"
    ).read_text(encoding="utf-8")

    seed_pos = source.index(
        "sanitized_protected_recovery_seed = ("
    )
    recovery_pos = source.index(
        "# Monotonic bounded post-hygiene scientific",
        seed_pos,
    )

    assert seed_pos < recovery_pos
    assert (
        "sanitize_structured_manuscript_publication_metadata("
        in source[seed_pos:recovery_pos]
    )
    assert (
        "selected_post_hygiene_"
        in source[seed_pos:recovery_pos]
    )
    assert (
        "protected_recovery_seed.json"
        in source[seed_pos:recovery_pos]
    )


def test_protected_recovery_seed_is_deterministic_only():
    source = Path(
        "src/cnsm_agentic/autonomous_research/final_pipeline.py"
    ).read_text(encoding="utf-8")

    start = source.index(
        "# Deterministic protected seed for post-hygiene recovery"
    )
    end = source.index(
        "# Monotonic bounded post-hygiene scientific",
        start,
    )
    seed_block = source[start:end]

    assert "run_agent(" not in seed_block
    assert "build_publication_artifacts(" in seed_block
    assert "audit_manuscript_publication_sanity(" in seed_block
    assert "audit_manuscript_artifact_references(" in seed_block


def test_protected_final_rescue_remains_agent_free_after_seed_change():
    source = Path(
        "src/cnsm_agentic/autonomous_research/final_pipeline.py"
    ).read_text(encoding="utf-8")

    start = source.index(
        "# Deterministic protected-candidate rescue"
    )
    end = source.index(
        "# Re-run BOTH deterministic audits",
        start,
    )

    assert "run_agent(" not in source[start:end]
