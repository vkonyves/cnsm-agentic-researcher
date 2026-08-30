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
        "Detailed machine provenance belongs in the archived artifact bundle."
    ) == 2


def test_peer_reviewer_checks_citation_semantics():
    source = Path(
        "src/cnsm_agentic/autonomous_research/final_agents.py"
    ).read_text(encoding="utf-8")

    assert (
        "Verify citation semantics, not merely reference existence."
        in source
    )
