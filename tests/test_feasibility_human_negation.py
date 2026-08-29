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


def test_r40_no_human_imputation_or_manual_adjudication_is_negated():
    text = "No human imputation or manual adjudication is used."

    cleaned = _prepare_human_dependency_scan_text(text)

    assert "human imputation" not in cleaned.lower()
    assert "manual adjudication" not in cleaned.lower()


def test_r40_do_not_perform_human_adjudication_is_negated():
    text = (
        "If verifier and simulator disagree, do not perform "
        "human adjudication."
    )

    cleaned = _prepare_human_dependency_scan_text(text)

    assert "human adjudication" not in cleaned.lower()


def test_positive_human_imputation_remains_detectable():
    text = "Missing values require human imputation."

    cleaned = _prepare_human_dependency_scan_text(text)

    assert "human imputation" in cleaned.lower()


def test_positive_human_adjudication_remains_detectable():
    text = "Ambiguous cases require human adjudication."

    cleaned = _prepare_human_dependency_scan_text(text)

    assert "human adjudication" in cleaned.lower()


from cnsm_agentic.autonomous_research.feasibility import (
    _find_positive_human_dependencies,
)


def positive_dependencies(text: str) -> set[str]:
    return set(
        _find_positive_human_dependencies(
            text,
            HUMAN_DEPENDENCY_PATTERNS,
        )
    )


def test_clause_aware_negation_matrix():
    cases = [
        "No human annotation is required.",
        "No human annotation is used.",
        "No human annotation will be performed.",
        "Human annotation is not required.",
        "Human annotation will not be used.",
        "The workflow does not require human annotation.",
        "The experiment runs without human annotation.",
        "Do not perform human annotation.",
        "We do not use human annotation.",
        "No manual review is required.",
        "Manual review is not required.",
        "The pipeline operates without manual review.",
        "No manual adjudication will be performed.",
        "Manual adjudication will not be used.",
        "The workflow does not rely on human review.",
        (
            "No manual imputation or human annotation "
            "will be performed."
        ),
        (
            "Automated deterministic audit is used; "
            "no human annotation is performed."
        ),
        (
            "All audit runs are automated and results are "
            "reported. No manual human audit is performed."
        ),
    ]

    for case in cases:
        assert positive_dependencies(case) == set(), case


def test_clause_aware_positive_dependency_matrix():
    cases = [
        (
            "Human annotation is required.",
            "human annotation",
        ),
        (
            "The evaluation uses human annotation.",
            "human annotation",
        ),
        (
            "Two human annotators label every artifact.",
            "human annotation",
        ),
        (
            "Manual adjudication resolves disagreements.",
            "manual adjudication",
        ),
        (
            "Ambiguous cases require manual review.",
            "manual review",
        ),
        (
            "Failures undergo human review.",
            "human review",
        ),
        (
            "A third rater resolves ties.",
            "third rater",
        ),
        (
            "The external validator approves each result.",
            "external validator",
        ),
    ]

    for text, expected in cases:
        assert expected in positive_dependencies(text), text


def test_negated_dependency_does_not_hide_positive_later_clause():
    text = (
        "No human annotation is used; "
        "manual review is required for ambiguous outputs."
    )

    found = positive_dependencies(text)

    assert "human annotation" not in found
    assert "manual review" in found


def test_positive_dependency_does_not_get_hidden_by_later_denial():
    text = (
        "Human annotation is required for the primary labels; "
        "no manual adjudication is used."
    )

    found = positive_dependencies(text)

    assert "human annotation" in found
    assert "manual adjudication" not in found


def test_r44_exact_missingness_sentence_is_not_dependency():
    text = (
        "No manual imputation or human annotation will "
        "be performed."
    )

    assert "human annotation" not in positive_dependencies(text)


def test_r44_exact_audit_sentence_is_not_dependency():
    text = (
        "Automated deterministic audit: 5% sample of episodes "
        "are re-processed using alternative tokenizer and provenance "
        "shingle size to detect sensitivity; all audit runs are automated "
        "and results are reported. No manual human audit is performed."
    )

    assert positive_dependencies(text) == set()


def test_passive_purpose_denial_is_not_human_dependency():
    cases = [
        "The flag is logged for analysis, not used for manual adjudication.",
        "The score is diagnostic only and not used for human annotation.",
        "The automatic flag is not intended for manual review.",
        "The metric is not required for human annotation.",
    ]

    for case in cases:
        assert positive_dependencies(case) == set(), case


def test_positive_used_for_human_dependency_still_detected():
    cases = [
        (
            "The flag is used for manual adjudication.",
            "manual adjudication",
        ),
        (
            "The score is used for human annotation.",
            "human annotation",
        ),
        (
            "Ambiguous outputs are intended for manual review.",
            "manual review",
        ),
    ]

    for text, expected in cases:
        found = positive_dependencies(text)
        assert expected in found, text


def test_r59_contamination_wording_is_feasible():
    text = (
        "If verifier signals unreachable required invariants or "
        "contradictory constraints, mark as 'verifier-contradiction' "
        "and log for analysis (not used for manual adjudication). "
        "No human annotation is required; all treatments and flags "
        "are deterministic and automated."
    )

    found = positive_dependencies(text)

    assert "manual adjudication" not in found
    assert "human annotation" not in found


def test_autonomous_scoring_allows_explicitly_prohibited_human_adjudication():
    from cnsm_agentic.autonomous_research.feasibility import (
        feasibility_report,
    )

    design = {
        "missingness_plan": (
            "Unparseable outputs are automatically flagged. "
            "Multiple imputation is not used because human "
            "adjudication is prohibited; instead deterministic "
            "best-case and worst-case bounds are reported."
        ),
        "transformation_validation": {
            "audit_sample_policy": (
                "No human annotation is used; validation is "
                "fully automated."
            )
        },
    }

    report = feasibility_report(
        design=design,
        capability_manifest={
            "autonomous_scoring_required": True,
            "human_scientific_labour_allowed": False,
            "human_annotation_allowed": False,
        },
    )

    assert not any(
        "autonomous-scoring requirement" in issue
        for issue in report.get("issues", [])
    )


def test_autonomous_scoring_rejects_positive_human_adjudication():
    from cnsm_agentic.autonomous_research.feasibility import (
        feasibility_report,
    )

    design = {
        "analysis_plan": (
            "Ambiguous harmful outcomes are resolved by human "
            "adjudication before the primary analysis."
        )
    }

    report = feasibility_report(
        design=design,
        capability_manifest={
            "autonomous_scoring_required": True,
        },
    )

    assert any(
        "autonomous-scoring requirement" in issue
        for issue in report.get("issues", [])
    )


def test_autonomous_scoring_rejects_manual_scoring():
    from cnsm_agentic.autonomous_research.feasibility import (
        feasibility_report,
    )

    design = {
        "analysis_plan": (
            "Each generated configuration receives manual scoring "
            "for operational correctness."
        )
    }

    report = feasibility_report(
        design=design,
        capability_manifest={
            "autonomous_scoring_required": True,
        },
    )

    assert any(
        "autonomous-scoring requirement" in issue
        for issue in report.get("issues", [])
    )


def test_autonomous_scoring_negation_does_not_hide_later_positive_dependency():
    from cnsm_agentic.autonomous_research.feasibility import (
        feasibility_report,
    )

    design = {
        "analysis_plan": (
            "Human adjudication is prohibited for parser failures; "
            "however, final operational correctness uses manual scoring."
        )
    }

    report = feasibility_report(
        design=design,
        capability_manifest={
            "autonomous_scoring_required": True,
        },
    )

    issues = report.get("issues", [])

    assert any(
        "autonomous-scoring requirement" in issue
        and "manual scoring" in issue
        for issue in issues
    )
