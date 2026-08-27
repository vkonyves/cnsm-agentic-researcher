from pathlib import Path


def pipeline_source() -> str:
    return Path(
        "src/cnsm_agentic/autonomous_research/pipeline.py"
    ).read_text(encoding="utf-8")


def agents_source() -> str:
    return Path(
        "src/cnsm_agentic/autonomous_research/agents.py"
    ).read_text(encoding="utf-8")


def test_candidate_critic_declares_candidate_id_immutable():
    source = agents_source()

    assert (
        "candidate_id is an immutable foreign key"
        in source
    )
    assert (
        "Never rename, normalize, suffix, clone, duplicate"
        in source
    )
    assert "exactly one review " in source
    assert "for every supplied candidate" in source


def test_candidate_critic_has_bounded_identity_repair():
    source = pipeline_source()

    assert (
        "async def _run_candidate_critic_with_coverage_repair("
        in source
    )
    assert "maximum_attempts: int = 3" in source
    assert "allowed_candidate_ids" in source
    assert "missing_review_ids" in source
    assert "unknown_review_ids" in source
    assert "candidate_criticism_repair" in source


def test_candidate_critic_identity_repair_does_not_remap_ids():
    source = pipeline_source()

    start = source.index(
        "async def _run_candidate_critic_with_coverage_repair("
    )
    end = source.index(
        "class AutonomousDiscoveryPipeline:",
        start,
    )
    section = source[start:end]

    assert (
        "_validate_review_coverage("
        in section
    )
    assert (
        "Copy candidate_id values verbatim"
        in section
    )
    assert "Copy candidate_id values verbatim. Do not " in section
    assert "rename, suffix, normalize, clone, infer, or " in section


def test_candidate_critic_attempts_are_archived():
    source = pipeline_source()

    assert '"critic_review_attempts"' in source
    assert (
        'f"critic_reviews_attempt_{attempt:02d}.json"'
        in source
    )
    assert (
        'f"coverage_attempt_{attempt:02d}.json"'
        in source
    )


def test_normal_and_regeneration_paths_use_coverage_repair():
    source = pipeline_source()

    assert (
        source.count(
            "await _run_candidate_critic_with_coverage_repair("
        )
        >= 2
    )


def test_strict_review_coverage_validator_remains():
    source = pipeline_source()

    assert (
        "if reviewed_ids != generated_ids:"
        in source
    )
    assert (
        "Critic reviews do not cover exactly "
        in source
    )
