import json
from pathlib import Path

from cnsm_agentic.autonomous_research.final_pipeline import (
    _compact_execution_manifest_for_analysis_planning,
)


SOURCE = Path(
    "src/cnsm_agentic/autonomous_research/final_pipeline.py"
).read_text(encoding="utf-8")


def test_analysis_planning_removes_only_large_hash_inventory():
    manifest = {
        "status": "COMPLETED",
        "adapter_family": "hosted_netops_gvr_v1",
        "planned_episode_count": 200,
        "completed_episode_count": 200,
        "model_calls_used": 203,
        "result_schema_id": "paired-binary-v1",
        "results_path": "execution/results.json",
        "warnings": [],
        "artifact_hashes": {
            f"execution/artifact/{i}.json": "a" * 64
            for i in range(5000)
        },
    }

    full_size = len(
        json.dumps(
            manifest,
            ensure_ascii=False,
        )
    )

    compact = (
        _compact_execution_manifest_for_analysis_planning(
            manifest
        )
    )

    compact_size = len(
        json.dumps(
            compact,
            ensure_ascii=False,
        )
    )

    assert "artifact_hashes" not in compact
    assert compact["artifact_hashes_total_count"] == 5000

    assert compact["status"] == manifest["status"]
    assert (
        compact["adapter_family"]
        == manifest["adapter_family"]
    )
    assert (
        compact["planned_episode_count"]
        == manifest["planned_episode_count"]
    )
    assert (
        compact["completed_episode_count"]
        == manifest["completed_episode_count"]
    )
    assert (
        compact["model_calls_used"]
        == manifest["model_calls_used"]
    )
    assert (
        compact["result_schema_id"]
        == manifest["result_schema_id"]
    )
    assert (
        compact["results_path"]
        == manifest["results_path"]
    )
    assert compact["warnings"] == manifest["warnings"]

    assert compact_size < full_size / 20


def test_analysis_planner_uses_compact_execution_manifest():
    payload_start = SOURCE.index(
        "analysis_payload: dict[str, Any] = {"
    )
    planner_call = SOURCE.index(
        "candidate_analysis_plan = await run_agent(",
        payload_start,
    )

    payload = SOURCE[
        payload_start:planner_call
    ]

    assert (
        "_compact_execution_manifest_for_analysis_planning("
        in payload
    )


def test_deterministic_analysis_validation_still_uses_full_manifest():
    compatibility = SOURCE.index(
        "analysis_compatibility_issues("
    )

    block = SOURCE[
        compatibility:compatibility + 700
    ]

    assert (
        "execution_manifest=execution_manifest"
        in block
    )
