from cnsm_agentic.autonomous_research.publication_renderer import (
    _render_paragraphs,
)


def test_long_machine_identifier_gets_break_opportunities():
    rendered = _render_paragraphs(
        "Preregistration: "
        "netops_prereg_C1_validator_feedback_repair_v2_2026-09-01."
    )

    assert (
        r"netops\_\allowbreak{}prereg\_\allowbreak{}C1"
        in rendered
    )
    assert (
        r"validator\_\allowbreak{}feedback\_\allowbreak{}repair"
        in rendered
    )
    assert (
        r"2026-\allowbreak{}09-\allowbreak{}01"
        in rendered
    )


def test_shared_controlled_fault_identifier_gets_break_opportunities():
    rendered = _render_paragraphs(
        "Mode: shared_controlled_fault_candidate semantics."
    )

    assert (
        r"shared\_\allowbreak{}controlled\_\allowbreak{}fault"
        in rendered
    )


def test_ordinary_short_hyphenated_prose_is_unchanged():
    rendered = _render_paragraphs(
        "This is a paired-design experiment."
    )

    assert "paired-design" in rendered
    assert r"paired-\allowbreak{}design" not in rendered
