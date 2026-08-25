from pathlib import Path


def test_final_pipeline_has_bounded_review_revision_loop():
    source = Path(
        "src/cnsm_agentic/autonomous_research/"
        "final_pipeline.py"
    ).read_text()

    assert "maximum_peer_review_rounds = 5" in source
    assert "latest_peer_review" in source
    assert 'f"review_{review_round:02d}.json"' in source
    assert "revision_rounds_dir" in source

    # A review may finalise only when the reviewer accepts
    # and no critical issue or required revision remains.
    assert "review_is_finalisable = (" in source
    assert (
        "latest_peer_review.accept_for_finalisation"
        in source
    )
    assert (
        "and not latest_peer_review.critical_issues"
        in source
    )
    assert (
        "and not latest_peer_review.required_revisions"
        in source
    )
    assert "if review_is_finalisable:" in source

    # Final readiness must use the latest review rather than
    # a permanently rejected first-round review.
    marker = "Final autonomous readiness judgement"
    assert marker in source

    final_judge_section = source.split(
        marker,
        1,
    )[1]

    assert '"peer_review"' in final_judge_section
    assert "latest_peer_review" in final_judge_section
    assert "peer_review.model_dump()" not in final_judge_section


def test_final_pipeline_requires_full_five_page_budget():
    source = Path(
        "src/cnsm_agentic/autonomous_research/"
        "final_pipeline.py"
    ).read_text()

    assert "maximum_format_revision_rounds = 7" in source

    assert (
        'publication_validation.get(\n'
        '                    "uses_full_page_budget",'
        in source
    )

    assert "if uses_full_page_budget:" in source

    # Underfilled papers must be expanded rather than accepted merely
    # because they are below the maximum page limit.
    assert "pages_missing = maximum_pages - page_count" in source
    assert (
        "occupies only "
        in source
    )
    assert (
        "of the required {maximum_pages} IEEE pages"
        in source
    )
    assert (
        "Do not merely rephrase existing text"
        in source
    )
    assert (
        "add materially useful scientific "
        in source
    )
    assert (
        "exactly {maximum_pages} pages"
        in source
    )

    # Oversized papers must still be compacted.
    assert (
        "exceeding the frozen IEEE "
        in source
    )
    assert (
        "Shorten and compact the manuscript"
        in source
    )


def test_final_judge_requires_exact_page_budget():
    source = Path(
        "src/cnsm_agentic/autonomous_research/"
        "final_agents.py"
    ).read_text()

    assert (
        "publication_validation.uses_full_page_budget=true"
        in source
    )
    assert (
        "publication_validation.page_count equals "
        in source
    )
    assert (
        "publication_validation.maximum_pages"
        in source
    )
    assert (
        "A paper with fewer pages "
        in source
    )
    assert (
        "than the frozen maximum must be treated as failing the exact-page "
        in source
    )
    assert (
        "publication gate even when within_page_limit=true"
        in source
    )
    assert (
        "publication_validation.passed=true"
        in source
    )


def test_peer_reviewer_does_not_require_unavailable_post_lock_work():
    source = Path(
        "src/cnsm_agentic/autonomous_research/"
        "final_agents.py"
    ).read_text()

    assert "Do not require new post-lock experiments" in source

    assert (
        "retrospective modification of the sealed preregistration"
        in source
    )

    assert (
        "convert the actionable requirement"
        in source
    )

    assert (
        "into an explicit manuscript clarification, limitation, deviation"
        in source
    )

    assert (
        "Do not require the manuscript to fabricate repository URLs"
        in source
    )

    assert (
        "reported transparently as unexecuted or"
        in source
    )

    assert (
        "unavailable, with its consequence for interpretation stated clearly"
        in source
    )


def test_manuscript_payload_compacts_verified_records():
    source = Path(
        "src/cnsm_agentic/autonomous_research/"
        "final_pipeline.py"
    ).read_text()

    assert (
        "def compact_verified_records_for_manuscript("
        in source
    )
    assert (
        "compact_verified_records_for_manuscript("
        in source
    )
    assert (
        '"abstract"'
        in source
    )
    assert (
        "value[:1200]"
        in source
    )


def test_context_window_failure_is_not_retried():
    source = Path(
        "src/cnsm_agentic/autonomous_research/"
        "final_pipeline.py"
    ).read_text()

    assert "context_length_exceeded" in source
    assert "exceeds the context window" in source
    assert (
        "identical retries "
        in source
    )


def test_manuscript_payload_compacts_execution_manifest():
    source = Path(
        "src/cnsm_agentic/autonomous_research/"
        "final_pipeline.py"
    ).read_text()

    assert (
        "def compact_execution_manifest_for_manuscript("
        in source
    )
    assert (
        '"artifact_hashes"'
        in source
    )
    assert (
        '"artifact_hash_summary"'
        in source
    )
    assert (
        '"artifact_count"'
        in source
    )
    assert (
        "compact_execution_manifest_for_manuscript("
        in source
    )