from pathlib import Path


SOURCE = Path(
    "src/cnsm_agentic/autonomous_research/final_pipeline.py"
).read_text(encoding="utf-8")


def _agent_payload(agent_name: str, occurrence: int = 0) -> str:
    marker = f"{agent_name},"
    positions = []
    start = 0

    while True:
        pos = SOURCE.find(marker, start)
        if pos < 0:
            break
        positions.append(pos)
        start = pos + len(marker)

    # Ignore import/registration occurrences by selecting runtime calls
    runtime = [
        pos
        for pos in positions
        if "run_agent(" in SOURCE[max(0, pos - 120):pos]
    ]

    pos = runtime[occurrence]

    next_call = SOURCE.find(
        "await run_agent(",
        pos + len(marker),
    )

    if next_call < 0:
        next_call = len(SOURCE)

    return SOURCE[pos:next_call]


def test_peer_reviewers_do_not_receive_full_execution_manifest():
    for occurrence in (0, 1):
        block = _agent_payload(
            "PEER_REVIEWER",
            occurrence,
        )

        assert (
            "_compact_execution_manifest_for_manuscript("
            in block
        )


def test_final_judge_does_not_receive_full_execution_manifest():
    block = _agent_payload(
        "FINAL_JUDGE",
        0,
    )

    assert (
        "_compact_execution_manifest_for_manuscript("
        in block
    )


def test_analysis_planner_uses_its_bounded_projection():
    payload_start = SOURCE.index(
        "analysis_payload: dict[str, Any] = {"
    )

    planner_call = SOURCE.index(
        "candidate_analysis_plan = await run_agent(",
        payload_start,
    )

    block = SOURCE[
        payload_start:planner_call
    ]

    assert (
        "_compact_execution_manifest_for_analysis_planning("
        in block
    )
