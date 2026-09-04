from cnsm_agentic.autonomous_research.hosted_netops_validator_feedback_adapter import (
    BLIND_REPAIR_TRANSFORMATION,
    GUIDED_REPAIR_TRANSFORMATION,
    HostedNetOpsValidatorFeedbackAdapter,
    validator_feedback_plan_issues,
)


def _plan(
    task_count: int = 40,
) -> dict:
    return {
        "study_id": "validator-feedback-test",
        "adapter_family": (
            "hosted_netops_validator_feedback_repair_v2"
        ),
        "result_schema_id": "paired_binary_episode_v1",
        "result_schema_version": "1.0",
        "conditions": [
            "baseline",
            "guarded",
        ],
        "design": "paired_binary",
        "task_count": task_count,
        "task_indices": list(
            range(
                1,
                task_count + 1,
            )
        ),
        "initial_generation_calls_per_task": 1,
        "maximum_repair_calls_per_task": 2,
        "maximum_model_calls_per_task": 3,
        "estimated_model_calls": task_count * 3,
        "maximum_model_calls": task_count * 3,
        "task_families": [
            "intent_configuration_repair_v1",
        ],
        "transformations": {
            "baseline":
                BLIND_REPAIR_TRANSFORMATION,
            "guarded":
                GUIDED_REPAIR_TRANSFORMATION,
        },
        "model_provider": "openai_responses",
        "model_name": "gpt-test",
        "model_version": "gpt-test-version",
        "deterministic_automated_scoring": True,
        "requires_human_scientific_labour": False,
        "execution_mode": "scientific_confirmatory",
        "maximum_attempts_per_call": 1,
        "max_output_tokens": 2000,
        "reasoning_effort": "minimal",
        "temperature": None,
    }


def test_validator_feedback_contract_accepts_40_tasks():
    assert validator_feedback_plan_issues(
        _plan()
    ) == []


def test_validator_feedback_requires_balanced_task_count():
    plan = _plan()
    plan["task_count"] = 41
    plan["task_indices"] = list(range(1, 42))
    plan["estimated_model_calls"] = 41 * 3
    plan["maximum_model_calls"] = 41 * 3

    issues = validator_feedback_plan_issues(
        plan
    )

    assert any(
        "divisible by 40" in issue
        for issue in issues
    )


def test_adapter_supports_valid_plan():
    adapter = HostedNetOpsValidatorFeedbackAdapter()

    assert adapter.supports(
        _plan()
    )


def test_mock_validator_feedback_execution_and_analysis(tmp_path):
    import json

    from cnsm_agentic.autonomous_research.analysis_executors import (
        PairedBinaryAnalysisExecutor,
        validate_analysis_results,
    )
    from cnsm_agentic.autonomous_research.execution_adapters import (
        validate_execution_manifest,
    )
    from cnsm_agentic.autonomous_research.model_providers import (
        ModelCallResult,
    )
    from cnsm_agentic.autonomous_research.netops_generate_validate_repair import (
        generate_task,
        render_reference_configuration,
    )

    class FakeProvider:
        provider_name = "openai_responses"

        def __init__(self):
            self.requests = []

        def call(self, request):
            self.requests.append(request)
            task_index = int(request.metadata["task_index"])
            task = generate_task(task_index)
            reference = render_reference_configuration(task)
            stage = request.metadata["stage"]

            if stage == "valid_source_generation":
                text = reference
            elif stage == "blind_controlled_fault_repair":
                text = request.prompt.split(
                    "Candidate:\n", 1
                )[1].split(
                    "\nReturn one corrected", 1
                )[0]
            elif stage == "validator_guided_controlled_fault_repair":
                text = reference
            else:
                raise AssertionError(stage)

            return ModelCallResult(
                provider="openai_responses",
                requested_model=request.model,
                resolved_model="gpt-test-2026-09-04",
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

    provider = FakeProvider()
    adapter = HostedNetOpsValidatorFeedbackAdapter(
        provider=provider
    )
    plan = _plan(task_count=40)
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

    assert manifest["source_valid_count"] == 40
    assert manifest["source_invalid_or_failed_count"] == 0
    assert manifest["model_calls_used"] == 120
    assert manifest["completed_episode_count"] == 80
    assert len(provider.requests) == 120

    assert validate_execution_manifest(
        manifest,
        plan=plan,
        output_dir=execution_dir,
        maximum_model_calls=120,
    ) == []

    rows = [
        json.loads(line)
        for line in (
            execution_dir / "raw_results.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    baseline = [
        row for row in rows
        if row["condition"] == "baseline"
    ]
    guarded = [
        row for row in rows
        if row["condition"] == "guarded"
    ]
    assert all(row["score"] == 0 for row in baseline)
    assert all(row["score"] == 1 for row in guarded)

    for pair_id in {row["pair_id"] for row in rows}:
        pair_rows = [
            row for row in rows
            if row["pair_id"] == pair_id
        ]
        assert len(pair_rows) == 2
        assert len({
            row["shared_initial_candidate_sha256"]
            for row in pair_rows
        }) == 1

    odd = [
        row for row in rows
        if row["task_id"] == "task-000001"
    ]
    even = [
        row for row in rows
        if row["task_id"] == "task-000002"
    ]
    assert next(
        row["condition_order"]
        for row in odd
        if row["condition"] == "baseline"
    ) == 1
    assert next(
        row["condition_order"]
        for row in even
        if row["condition"] == "baseline"
    ) == 2

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
    assert primary["complete_pair_count"] == 40
    assert primary["n_10"] == 40
    assert primary["n_01"] == 0
    assert primary["estimate"] == 1.0
