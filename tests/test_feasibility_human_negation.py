from cnsm_agentic.autonomous_research.feasibility import (
    HUMAN_DEPENDENCY_PATTERNS,
    _find_patterns,
    _prepare_human_dependency_scan_text,
)


def dependencies(text: str) -> set[str]:
    cleaned = _prepare_human_dependency_scan_text(text)
    return set(
        _find_patterns(
            cleaned,
            HUMAN_DEPENDENCY_PATTERNS,
        )
    )


def test_coordinated_human_dependencies_are_negated():
    cases = [
        "No human annotation or manual adjudication is required.",
        "No manual review or human annotation is needed.",
        "This does not require human annotation or manual adjudication.",
        "The experiment runs without human review or manual adjudication.",
    ]

    for text in cases:
        found = dependencies(text)
        assert "human annotation" not in found
        assert "manual adjudication" not in found
        assert "human review" not in found
        assert "manual review" not in found


def test_positive_human_dependencies_remain_forbidden():
    cases = [
        (
            "Ambiguous outputs require manual adjudication.",
            "manual adjudication",
        ),
        (
            "The evaluation uses human annotation.",
            "human annotation",
        ),
        (
            "Failures undergo human review.",
            "human review",
        ),
        (
            "The final sample requires manual review.",
            "manual review",
        ),
    ]

    for text, expected in cases:
        assert expected in dependencies(text)


def test_r38_false_positive_sentence_is_removed():
    text = (
        "All execution uses hosted_model_api and CPU-executable "
        "deterministic verifiers; no human annotation or manual "
        "adjudication is required."
    )

    assert dependencies(text) == set()
