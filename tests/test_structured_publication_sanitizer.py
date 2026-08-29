

def test_protected_rescue_audits_sanitized_candidate():
    from pathlib import Path

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
    rescue = source[start:end]

    assert (
        "sanitized_protected_submission_manuscript"
        in rescue
    )

    assert (
        "manuscript=(\n"
        "                            "
        "sanitized_protected_submission_manuscript"
        in rescue
    )


def test_protected_rescue_promotes_sanitized_candidate():
    from pathlib import Path

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
    rescue = source[start:end]

    assert (
        "revised_manuscript = (\n"
        "                        "
        "sanitized_protected_submission_manuscript"
        in rescue
    )
