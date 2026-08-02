from pathlib import Path

from cnsm_agentic.autonomous_research.analysis_executors import (
    PairedBinaryAnalysisExecutor,
    validate_analysis_results,
)
from cnsm_agentic.autonomous_research.execution_adapters import (
    validate_execution_manifest,
)
from cnsm_agentic.autonomous_research.hosted_netops_adapter import (
    HostedNetOpsGVRAdapter,
    hosted_netops_plan_issues,
)
from cnsm_agentic.autonomous_research.model_providers import (
    ModelCallRequest,
    ModelCallResult,
)

from cnsm_agentic.autonomous_research.netops_generate_validate_repair import (
    generate_task,
    render_reference_configuration,
)


class FakeHostedProvider:
    provider_name = "openai_responses"

    def __init__(self) -> None:
        self.requests: list[ModelCallRequest] = []

    def call(self, request: ModelCallRequest) -> ModelCallResult:
        self.requests.append(request)
        task_number = int(request.metadata["task_id"].split("-")[1])
        stage = request.metadata["stage"]
        task = generate_task(task_number)
        reference = render_reference_configuration(task)

        if stage == "shared_initial_generation":
            lines = reference.splitlines()
            if task_number == 1:
                parts = lines[0].split()
                parts[-1] = "down" if parts[-1] == "up" else "up"
                lines[0] = " ".join(parts)
            else:
                lines = lines[:-1]
            text = "\n".join(lines)
        else:
            text = reference

        return ModelCallResult(
            provider="openai_responses",
            requested_model=request.model,
            resolved_model="gpt-test-2026-08-02",
            response_text=text,
            response_id=f"resp-{len(self.requests)}",
            request_id=f"req-{len(self.requests)}",
            input_tokens=20,
            output_tokens=12,
            total_tokens=32,
            latency_ms=5,
            attempt_count=1,
            cache_status="MISS",
            cache_key_sha256=request.cache_key_sha256(),
        )


def _plan(task_count: int = 2):
    return {
        "study_id": "hosted-pilot-test",
        "adapter_family": "hosted_netops_gvr_v1",
        "result_schema_id": "paired_binary_episode_v1",
        "result_schema_version": "1.0",
        "conditions": ["baseline", "guarded"],
        "design": "paired_binary",
        "task_count": task_count,
        "task_indices": list(range(1, task_count + 1)),
        "estimated_model_calls": task_count * 2,
        "maximum_model_calls": task_count * 2,
        "task_families": ["intent_configuration_repair_v1"],
        "transformations": {
            "baseline": "direct_configuration_generation_v1",
            "guarded": "generate_validate_repair_v1",
        },
        "model_provider": "openai_responses",
        "model_name": "gpt-test",
        "model_version": "gpt-test-2026-08-02",
        "deterministic_automated_scoring": True,
        "requires_human_scientific_labour": False,
        "execution_mode": "scientific_pilot",
        "maximum_attempts_per_call": 1,
        "max_output_tokens": 200,
        "reasoning_effort": "minimal",
        "temperature": None,
    }


def test_hosted_plan_contract_accepts_bounded_pilot() -> None:
    assert hosted_netops_plan_issues(_plan()) == []


def test_hosted_plan_rejects_inexact_call_ceiling() -> None:
    plan = _plan()
    plan["maximum_model_calls"] = 3
    assert any(
        "exact pilot ceiling" in issue
        for issue in hosted_netops_plan_issues(plan)
    )


def test_hosted_plan_rejects_duplicate_task_indices() -> None:
    plan = _plan()
    plan["task_indices"] = [7, 7]
    assert any(
        "task_indices" in issue
        for issue in hosted_netops_plan_issues(plan)
    )


def test_mock_hosted_execution_and_analysis(tmp_path: Path) -> None:
    provider = FakeHostedProvider()
    adapter = HostedNetOpsGVRAdapter(provider=provider)
    plan = _plan(task_count=2)
    plan["task_indices"] = [7, 8]
    execution_dir = tmp_path / "execution"

    manifest = adapter.execute(
        plan=plan,
        preregistration={
            "study_id": plan["study_id"],
            "primary_estimand": (
                "paired_success_rate_difference_guarded_minus_baseline"
            ),
        },
        output_dir=execution_dir,
    )

    assert manifest["model_calls_used"] == 4
    assert manifest["completed_episode_count"] == 4
    assert manifest["failed_episode_count"] == 0
    assert len(provider.requests) == 4
    assert validate_execution_manifest(
        manifest,
        plan=plan,
        output_dir=execution_dir,
        maximum_model_calls=4,
    ) == []

    raw_rows = [
        __import__("json").loads(line)
        for line in (execution_dir / "raw_results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    baseline = [row for row in raw_rows if row["condition"] == "baseline"]
    guarded = [row for row in raw_rows if row["condition"] == "guarded"]
    assert [row["score"] for row in baseline] == [0, 0]
    assert [row["score"] for row in guarded] == [1, 1]
    assert all(
        (tmp_path / row["scoring_artifact_path"]).is_file()
        for row in raw_rows
    )
    assert len(list((execution_dir / "provider_calls").glob("*.json"))) == 4
    assert len(list((execution_dir / "responses").glob("*shared-initial.txt"))) == 2
    for pair_id in {"pair-000007", "pair-000008"}:
        pair_rows = [row for row in raw_rows if row["pair_id"] == pair_id]
        assert len({row["shared_initial_candidate_sha256"] for row in pair_rows}) == 1
        assert len({row["shared_initial_provider_trace_sha256"] for row in pair_rows}) == 1
        baseline_row = next(row for row in pair_rows if row["condition"] == "baseline")
        guarded_row = next(row for row in pair_rows if row["condition"] == "guarded")
        assert not (baseline_row["score"] == 1 and guarded_row["score"] != 1)

    analysis_plan = {
        "analysis_executor": "paired_binary_analysis_v1",
        "study_id": plan["study_id"],
        "estimand": (
            "paired_success_rate_difference_guarded_minus_baseline"
        ),
        "failed_call_treatment": "complete_pair_primary",
        "bootstrap_seed": 7,
        "bootstrap_resamples": 100,
        "confidence_level": 0.95,
    }
    results = PairedBinaryAnalysisExecutor().execute(
        analysis_plan=analysis_plan,
        preregistration={},
        execution_manifest=manifest,
        run_dir=tmp_path,
    )
    assert validate_analysis_results(
        results,
        run_dir=tmp_path,
        execution_manifest=manifest,
    ) == []
    primary = results["confirmatory_results"][0]
    assert primary["complete_pair_count"] == 2
    assert primary["n_10"] == 2
    assert primary["n_01"] == 0
    assert primary["estimate"] == 1.0
