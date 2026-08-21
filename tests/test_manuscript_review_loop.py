from pathlib import Path


def test_final_pipeline_has_bounded_review_revision_loop():
    source = Path(
        "src/cnsm_agentic/autonomous_research/"
        "final_pipeline.py"
    ).read_text()

    assert "maximum_peer_review_rounds = 3" in source
    assert "latest_peer_review" in source
    assert 'f"review_{review_round:02d}.json"' in source
    assert "revision_rounds_dir" in source

    # Final readiness must use the latest review rather than
    # a permanently rejected first-round review.
    final_judge_section = source.split(
        "# 14. Final autonomous readiness judgement",
        1,
    )[1]

    assert '"peer_review"' in final_judge_section
    assert "latest_peer_review" in final_judge_section
    assert "peer_review.model_dump()" not in final_judge_section
