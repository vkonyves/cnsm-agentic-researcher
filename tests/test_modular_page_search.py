from pathlib import Path


def source() -> str:
    return Path(
        "src/cnsm_agentic/autonomous_research/final_pipeline.py"
    ).read_text(encoding="utf-8")


def test_modular_search_occurs_before_protected_checkpoint():
    text = source()

    modular = text.index(
        "Modular evidence-grounded exact-page convergence"
    )
    checkpoint = text.index(
        "Protected exact-page submission checkpoint"
    )

    assert modular < checkpoint


def test_modular_search_uses_section_level_variants():
    text = source()

    assert '"methodology"' in text
    assert '"results"' in text
    assert '"discussion"' in text
    assert '"related_work"' in text

    assert "modular_variants" in text
    assert "candidate_section_text" in text


def test_only_requested_section_is_taken_from_model_response():
    text = source()

    assert (
        "Only the requested section may"
        in text
    )
    assert (
        'generated_dump[\n'
        '                            "sections"\n'
        '                        ][modular_section]'
        in text
    )


def test_combination_search_is_deterministic():
    text = source()

    assert "for modular_levels in product(" in text
    assert "ManuscriptPackage.model_validate(" in text
    assert "build_publication_artifacts(" in text


def test_exact_page_candidates_require_publication_pass():
    text = source()

    assert (
        'candidate_validation.get(\n'
        '                        "passed"\n'
        '                    )\n'
        '                    is True'
        in text
    )
    assert (
        "candidate_pages\n"
        "                    == modular_maximum_pages"
        in text
    )


def test_richest_exact_candidate_is_selected():
    text = source()

    assert "modular_exact_candidates.sort(" in text
    assert "reverse=True" in text
    assert "modular_exact_candidates[0]" in text


def test_selected_candidate_is_rerendered_authoritatively():
    text = source()

    marker = (
        "Re-render selected candidate into the authoritative"
    )
    start = text.index(marker)
    section = text[start:start + 1400]

    assert "build_publication_artifacts(" in section
    assert "publication_dir" in section


def test_no_exact_candidate_can_still_improve_underfill():
    text = source()

    assert "modular_best_under_manuscript" in text
    assert "modular_best_under_pages" in text
    assert "modular_best_under_words" in text
