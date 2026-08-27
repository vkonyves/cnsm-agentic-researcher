from pathlib import Path


def _final_pipeline_source() -> str:
    return Path(
        "src/cnsm_agentic/autonomous_research/"
        "final_pipeline.py"
    ).read_text(encoding="utf-8")


def _final_agents_source() -> str:
    return Path(
        "src/cnsm_agentic/autonomous_research/"
        "final_agents.py"
    ).read_text(encoding="utf-8")


def test_final_pipeline_has_bounded_review_revision_loop():
    source = _final_pipeline_source()

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
    assert (
        "peer_review.model_dump()"
        not in final_judge_section
    )


def test_final_pipeline_requires_full_five_page_budget():
    source = _final_pipeline_source()

    assert "maximum_format_revision_rounds = 16" in source

    assert (
        'publication_validation.get(\n'
        '                    "uses_full_page_budget",'
        in source
    )

    assert "if uses_full_page_budget:" in source

    # Underfilled papers must be expanded rather than accepted merely
    # because they are below the maximum page limit.
    assert (
        "pages_missing = maximum_pages - page_count"
        in source
    )
    assert "occupies only " in source
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
        "target approximately one page above the final budget"
        in source
        or
        "target six compiled pages"
        in source
        or
        "overshoot"
        in source.lower()
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


def test_underfill_revision_uses_overshoot_first_convergence():
    source = _final_pipeline_source()

    assert "overshoot_target_pages = maximum_pages + 1" in source
    assert "minimum_expansion_words = max(" in source
    assert "2300" in source
    assert "preferred_expansion_words = max(" in source
    assert "2600" in source
    assert "Do NOT target {maximum_pages} pages from below" in source
    assert "target approximately " in source
    assert "compaction from above" in source


def test_final_judge_requires_exact_page_budget():
    source = _final_agents_source()

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
        "than the frozen maximum must be treated as "
        "failing the exact-page "
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
    source = _final_agents_source()

    assert (
        "Do not require new post-lock experiments"
        in source
    )

    assert (
        "retrospective modification of the sealed preregistration"
        in source
    )

    assert (
        "convert the actionable requirement"
        in source
    )

    assert (
        "into an explicit manuscript clarification, limitation, "
        "deviation"
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
        "unavailable, with its consequence for interpretation "
        "stated clearly"
        in source
    )


def test_manuscript_payload_compacts_verified_records():
    source = _final_pipeline_source()

    assert (
        "def compact_verified_records_for_manuscript("
        in source
    )
    assert (
        "compact_verified_records_for_manuscript("
        in source
    )
    assert '"abstract"' in source
    assert "value[:1200]" in source


def test_context_window_failure_is_not_retried():
    source = _final_pipeline_source()

    assert "context_length_exceeded" in source
    assert "exceeds the context window" in source
    assert "identical retries " in source


def test_manuscript_payload_compacts_execution_manifest():
    source = _final_pipeline_source()

    assert (
        "def compact_execution_manifest_for_manuscript("
        in source
    )
    assert '"artifact_hashes"' in source
    assert '"artifact_hash_summary"' in source
    assert '"artifact_count"' in source
    assert (
        "compact_execution_manifest_for_manuscript("
        in source
    )


def test_format_revision_performs_only_one_revision_per_round():
    source = _final_pipeline_source()

    start = source.index(
        "maximum_format_revision_rounds = 16"
    )

    # Stop before the separate terminal-review/revision loop.
    end = source.index(
        "maximum_terminal_revision_rounds = 2",
        start,
    )

    format_section = source[start:end]

    assert (
        format_section.count(
            "revised_manuscript = await run_agent("
        )
        == 1
    )

    assert (
        '"revision_instruction": (\n'
        '                        revision_instruction'
        in format_section
    )


def test_format_revision_preserves_best_under_limit_candidate():
    source = _final_pipeline_source()

    start = source.index(
        "maximum_format_revision_rounds = 16"
    )
    end = source.index(
        "maximum_terminal_revision_rounds = 2",
        start,
    )

    format_section = source[start:end]

    assert (
        "best_manuscript = revised_manuscript"
        in format_section
    )
    assert (
        "best_publication_validation"
        in format_section
    )
    assert (
        "best_page_count = -1"
        in format_section
    )
    assert (
        "current_page_count > best_page_count"
        in format_section
    )
    assert (
        "current_page_count <= current_maximum_pages"
        in format_section
    )
    # When an expansion overshoots the page budget, the current
    # overfull candidate must remain available as the next revision base
    # so it can be compacted down toward exactly five pages.
    assert (
        "revision_base_manuscript"
        in format_section
    )
    assert (
        "revision_base_manuscript = revised_manuscript"
        in format_section
        or
        "revision_base_manuscript = ("
        in format_section
    )
    assert (
        "revised_manuscript = best_manuscript"
        in format_section
    )

    assert (
        "revised_manuscript.manuscript"
        not in format_section
    )

    assert (
        "revised_manuscript.model_dump()"
        in format_section
    )


def test_best_format_candidate_is_rerendered_before_terminal_review():
    source = _final_pipeline_source()

    start = source.index(
        "if (\n"
        "            best_publication_validation is not None"
    )
    end = source.index(
        "maximum_terminal_revision_rounds = 2",
        start,
    )

    selection_section = source[start:end]

    assert (
        "revised_manuscript = best_manuscript"
        in selection_section
    )

    assert (
        "publication_validation = (\n"
        "                build_publication_artifacts("
        in selection_section
    )

    assert (
        '"revised_package.json"'
        in selection_section
    )


def test_manuscript_revision_receives_compact_evidence_bundle():
    source = _final_pipeline_source()

    assert (
        "def build_manuscript_evidence_bundle("
        in source
    )
    assert '"artifact_examples"' in source
    assert '"analysis_artifacts"' in source
    assert '"manuscript_evidence_bundle"' in source
    assert '"model_configuration"' in source
    assert (
        '"initial_master_prompt_reference"'
        in source
    )
    assert '"representative_tasks"' in source
    assert '"shared-initial"' in source
    assert '"condition_summary.csv"' in source


def test_manuscript_evidence_bundle_is_archived():
    source = _final_pipeline_source()

    assert (
        '"manuscript_evidence_bundle.json"'
        in source
    )


def test_underfill_revisions_are_cumulative():
    source = _final_pipeline_source()

    assert (
        "Each underfill revision must be "
        in source
    )
    assert (
        "cumulative: retain existing Methods, Results, tables"
        in source
    )
    assert (
        "do not shorten, "
        in source
    )


def test_terminal_peer_review_runs_after_format_revision():
    source = _final_pipeline_source()

    format_pos = source.index(
        "maximum_format_revision_rounds = 16"
    )
    terminal_pos = source.index(
        "maximum_terminal_revision_rounds = 2",
        format_pos,
    )

    assert format_pos < terminal_pos

    assert '"review_terminal.json"' in source
    assert (
        '"Terminal AI peer review "'
        in source
    )
    assert (
        "maximum_peer_review_rounds\n"
        "                        + terminal_round"
        in source
    )


def test_terminal_peer_review_can_drive_bounded_revisions():
    source = _final_pipeline_source()

    start = source.index(
        "maximum_terminal_revision_rounds = 2"
    )
    end = source.index(
        "if publication_validation is None:",
        start,
    )

    terminal_section = source[start:end]

    assert (
        "maximum_terminal_revision_rounds = 2"
        in terminal_section
    )
    assert (
        "for terminal_round in range("
        in terminal_section
    )
    assert (
        "maximum_terminal_revision_rounds + 2"
        in terminal_section
    )

    assert (
        '"review_terminal_"'
        in terminal_section
    )
    assert (
        '"terminal_revised_package_"'
        in terminal_section
    )
    assert (
        '"publication_validation_terminal_"'
        in terminal_section
    )

    assert (
        "not latest_peer_review.critical_issues"
        in terminal_section
    )
    assert (
        "not latest_peer_review.required_revisions"
        in terminal_section
    )

    assert (
        "terminal_round\n"
        "                > maximum_terminal_revision_rounds"
        in terminal_section
    )

    assert (
        "Terminal peer-review manuscript revision "
        in terminal_section
    )


def test_terminal_peer_review_receives_evidence_bundle():
    source = _final_pipeline_source()

    terminal_start = source.index(
        "maximum_terminal_revision_rounds = 2"
    )

    terminal_end = source.index(
        "if publication_validation is None:",
        terminal_start,
    )

    terminal_section = source[
        terminal_start:terminal_end
    ]

    assert (
        '"manuscript_evidence_bundle"'
        in terminal_section
    )
    assert (
        '"evidence_verification"'
        in terminal_section
    )
    assert (
        '"preregistration"'
        in terminal_section
    )
    assert (
        '"execution_manifest"'
        in terminal_section
    )
    assert (
        '"analysis_results"'
        in terminal_section
    )
    assert (
        '"deterministic_reconciliation"'
        in terminal_section
    )


def test_terminal_revision_receives_latest_peer_review():
    source = _final_pipeline_source()

    start = source.index(
        "maximum_terminal_revision_rounds = 2"
    )
    end = source.index(
        "if publication_validation is None:",
        start,
    )

    terminal_section = source[start:end]

    assert (
        '"peer_review": (\n'
        "                        latest_peer_review.model_dump()"
        in terminal_section
    )

    assert (
        '"publication_validation": (\n'
        "                        publication_validation"
        in terminal_section
    )

    assert (
        '"revision_round": (\n'
        '                        "terminal_"'
        in terminal_section
    )


def test_terminal_revision_is_rerendered_and_archived():
    source = _final_pipeline_source()

    start = source.index(
        "maximum_terminal_revision_rounds = 2"
    )
    end = source.index(
        "if publication_validation is None:",
        start,
    )

    terminal_section = source[start:end]

    assert (
        '"terminal_revised_package_"'
        in terminal_section
    )

    assert (
        '"publication_validation_terminal_"'
        in terminal_section
    )

    assert (
        "build_publication_artifacts("
        in terminal_section
    )

    # The conventional latest-manuscript alias must track terminal
    # revisions as well.
    assert (
        '"revised_package.json"'
        in terminal_section
    )


def test_terminal_review_compatibility_alias_is_written():
    source = _final_pipeline_source()

    start = source.index(
        "maximum_terminal_revision_rounds = 2"
    )
    end = source.index(
        "if publication_validation is None:",
        start,
    )

    terminal_section = source[start:end]

    assert (
        '"review_terminal.json"'
        in terminal_section
    )

    assert (
        "latest_peer_review"
        in terminal_section
    )


def test_terminal_review_precedes_final_judge():
    source = _final_pipeline_source()

    terminal_pos = source.index(
        '"review_terminal.json"'
    )

    final_marker_pos = source.index(
        "Final autonomous readiness judgement",
        terminal_pos,
    )

    assert terminal_pos < final_marker_pos


def test_final_publication_validation_alias_is_written():
    source = _final_pipeline_source()

    terminal_pos = source.index(
        "maximum_terminal_revision_rounds = 2"
    )

    final_alias_pos = source.index(
        '"publication_validation.json"',
        terminal_pos,
    )

    final_judge_pos = source.index(
        "Final autonomous readiness judgement",
        final_alias_pos,
    )

    assert final_alias_pos < final_judge_pos


def test_regular_peer_review_receives_evidence_bundle():
    source = _final_pipeline_source()

    review_loop_start = source.index(
        "for review_round in range("
    )

    review_loop_end = source.index(
        "maximum_format_revision_rounds",
        review_loop_start,
    )

    review_section = source[
        review_loop_start:review_loop_end
    ]

    assert (
        '"manuscript_evidence_bundle"'
        in review_section
    )


def test_terminal_review_uses_closure_mode_after_first_round():
    source = _final_pipeline_source()

    assert (
        'previous_terminal_review: PeerReviewReport | None = None'
        in source
    )

    assert (
        '"full_terminal_review"'
        in source
    )

    assert (
        '"closure_review"'
        in source
    )

    assert (
        '"review_mode": ('
        in source
    )

    assert (
        '"previous_terminal_review": ('
        in source
    )

    assert (
        "previous_terminal_review = latest_peer_review"
        in source
    )


def test_peer_reviewer_does_not_turn_outcomes_into_revision_defects():
    source = _final_agents_source()

    assert (
        "unfavorable scientific outcomes are not themselves"
        in source
    )

    assert (
        "scientific result or limitation"
        in source
    )

    assert (
        "Do not use raw execution artifacts as authorization"
        in source
    )

    assert (
        'review_mode="closure_review"'
        in source
    )

    assert (
        "do not introduce a new"
        in source
    )


def test_terminal_revision_has_bounded_page_convergence_loop():
    source = _final_pipeline_source()

    start = source.index(
        "maximum_terminal_revision_rounds = 2"
    )
    end = source.index(
        "if publication_validation is None:",
        start,
    )

    terminal_section = source[start:end]

    normalized_terminal_section = " ".join(
        terminal_section.split()
    )

    assert (
        "maximum_terminal_format_rounds = 10"
        in terminal_section
    )

    assert (
        "for terminal_format_round in range("
        in terminal_section
    )

    assert (
        "terminal_compile_status"
        in terminal_section
    )

    assert (
        "terminal_page_count"
        in terminal_section
    )

    assert (
        "terminal_maximum_pages"
        in terminal_section
    )

    assert (
        "terminal_page_count == terminal_maximum_pages"
        in normalized_terminal_section
    )

    assert (
        '"terminal_format_revised_package_"'
        in terminal_section
    )

    assert (
        '"publication_validation_terminal_format_"'
        in terminal_section
    )

    assert (
        "Terminal manuscript page convergence "
        in terminal_section
    )

    assert (
        "terminal_format_instruction"
        in terminal_section
    )

    assert (
        "Do not invent evidence"
        in terminal_section
    )

    assert (
        "best_terminal_manuscript"
        in terminal_section
    )

    assert (
        "best_terminal_publication_validation"
        in terminal_section
    )

    assert (
        "best_terminal_size"
        in terminal_section
    )

    assert (
        "terminal_revision_base_manuscript"
        in terminal_section
    )

    assert (
        "current_terminal_candidate_is_better"
        in terminal_section
    )

    assert (
        '"publication_validation_terminal_best_"'
        in terminal_section
    )


def test_terminal_page_convergence_occurs_after_terminal_revision_render():
    source = _final_pipeline_source()

    start = source.index(
        "maximum_terminal_revision_rounds = 2"
    )
    end = source.index(
        "if publication_validation is None:",
        start,
    )

    terminal_section = source[start:end]

    terminal_revision_pos = terminal_section.index(
        '"terminal_revised_package_"'
    )

    first_terminal_validation_pos = terminal_section.index(
        '"publication_validation_terminal_"'
    )

    terminal_format_loop_pos = terminal_section.index(
        "maximum_terminal_format_rounds = 10"
    )

    assert (
        terminal_revision_pos
        < first_terminal_validation_pos
        < terminal_format_loop_pos
    )

def test_preregistration_blocks_unsupported_scientific_execution_stages():
    source = _final_pipeline_source()
    start = source.index(
        "def preregistration_execution_contract_issues("
    )
    end = source.index(
        "def canonicalize_preregistration_execution_contract(",
        start,
    )
    section = source[start:end]
    assert "supports_multi_model_consensus" in section
    assert "3-model consensus" in section
    assert "supports_simulated_human_gate" in section
    assert "supports_prompt_family_stratification" in section
    assert "benign" in section
    assert "ambiguous" in section
    assert "adversarial" in section
    assert "execute multi-model consensus." in section
    assert "generate or record those strata." in section


def test_preregistration_repair_prompt_forbids_unsupported_stages():
    source = _final_pipeline_source()
    assert "consensus stages, ensembles, multi-model " in source
    assert "voting, simulated-human gates, prompt-family strata" in source
    assert "and task strata that the selected adapter actually executes " in source
    assert "and records. Set task_count exactly to " in source


def test_underfill_prompts_require_structural_additions():
    source = _final_pipeline_source()
    assert "Make structural additions rather than primarily rewriting" in source
    assert "eight distinct new artifact-grounded paragraphs" in source
    assert "page-convergence control, not permission to pad" in source
    assert "aim to cross the boundary" in source


def test_terminal_underfill_targets_deliberate_overshoot():
    source = _final_pipeline_source()
    assert "terminal_overshoot_target_pages" in source
    assert "terminal_minimum_words = max(" in source
    assert "terminal_preferred_words = max(" in source
    assert "Do not try to approach five pages cautiously from below" in source
    assert "compact the actual over-limit candidate" in source


def test_overfill_compaction_uses_actual_overlimit_candidate():
    source = _final_pipeline_source()
    assert "if page_count > maximum_pages" in source
    assert "revision_base_manuscript = (" in source
    assert "revised_manuscript" in source
    assert "if terminal_page_count > terminal_maximum_pages:" in source
    assert "terminal_revision_base_manuscript = revised_manuscript" in source


def test_page_convergence_counts_scientific_section_words():
    source = _final_pipeline_source()
    assert "def manuscript_section_word_count(" in source
    assert 'payload.get("sections", {})' in source
    assert "best_section_word_count" in source
    assert "current_section_word_count" in source
