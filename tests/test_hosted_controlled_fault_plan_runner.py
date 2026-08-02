import json
from pathlib import Path

import pytest

from cnsm_agentic.autonomous_research.controlled_fault_experiment_plan import (
    generate_experiment_plan,
    write_experiment_plan,
)
from cnsm_agentic.autonomous_research.hosted_controlled_fault_plan_runner import (
    HostedControlledFaultPlanRunner,
)
from cnsm_agentic.autonomous_research.model_providers import ModelCallResult
from cnsm_agentic.autonomous_research.netops_generate_validate_repair import (
    render_reference_configuration,
    generate_task,
)


class FakeProvider:
    def __init__(
        self,
        responses: list[str | Exception],
    ) -> None:
        self.responses = list(responses)
        self.requests = []

    def call(self, request):
        self.requests.append(request)
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return ModelCallResult(
            provider="openai_responses",
            requested_model=request.model,
            resolved_model=request.model,
            response_text=item,
            request_id=f"req-{len(self.requests)}",
            response_id=f"resp-{len(self.requests)}",
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            latency_ms=10,
            attempt_count=1,
            cache_status="MISS",
            cache_key_sha256=f"hash-{len(self.requests)}",
        )


def _small_plan(
    tmp_path: Path,
    *,
    pair_count: int = 5,
) -> Path:
    plan = generate_experiment_plan(pair_count=40, seed=17)
    plan["pairs"] = plan["pairs"][:pair_count]
    plan["pair_count"] = pair_count
    plan["maximum_model_calls"] = pair_count * 2
    plan["fault_class_counts"] = {}
    plan["workflow_pattern_counts"] = {}

    for pair in plan["pairs"]:
        fault = pair["fault_class"]
        plan["fault_class_counts"][fault] = (
            plan["fault_class_counts"].get(fault, 0) + 1
        )

        pattern = pair["workflow_pattern"]
        plan["workflow_pattern_counts"][pattern] = (
            plan["workflow_pattern_counts"].get(pattern, 0) + 1
        )

    import hashlib

    canonical = json.dumps(
        {
            key: value
            for key, value in plan.items()
            if key != "plan_sha256"
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    plan["plan_sha256"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()

    path = tmp_path / f"small-plan-{pair_count}.json"
    path.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def test_plan_runner_completes_fake_five_pair_run(
    tmp_path: Path,
) -> None:
    plan_path = _small_plan(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    responses = []
    for pair in plan["pairs"]:
        reference = render_reference_configuration(
            generate_task(pair["task_index"])
        )
        responses.extend([reference, reference])

    provider = FakeProvider(responses)
    summary = HostedControlledFaultPlanRunner(
        provider=provider
    ).execute(
        plan_path=plan_path,
        run_dir=tmp_path / "run",
        model_name="fake-model",
        max_output_tokens=2000,
    )

    assert summary["execution_status"] == "COMPLETED"
    assert summary["terminal_pair_count"] == 5
    assert summary["complete_scientific_pair_count"] == 5
    assert summary["model_calls_used"] == 10
    assert summary["baseline_successes"] == 0
    assert summary["guarded_successes"] == 5
    assert len(provider.requests) == 10
    for request in provider.requests:
        assert isinstance(request.metadata["task_index"], str)
        assert request.metadata["task_index"].isdigit()


def test_resume_skips_completed_pair_after_process_interruption(
    tmp_path: Path,
) -> None:
    plan_path = _small_plan(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    first = plan["pairs"][0]

    first_reference = render_reference_configuration(
        generate_task(first["task_index"])
    )

    provider_one = FakeProvider(
        [
            first_reference,
            first_reference,
            KeyboardInterrupt(),
        ]
    )
    runner_one = HostedControlledFaultPlanRunner(provider=provider_one)
    with pytest.raises(KeyboardInterrupt):
        runner_one.execute(
            plan_path=plan_path,
            run_dir=tmp_path / "resume-run",
            model_name="fake-model",
            max_output_tokens=2000,
        )

    remaining_responses = []
    for pair in plan["pairs"][1:]:
        reference = render_reference_configuration(
            generate_task(pair["task_index"])
        )
        remaining_responses.extend([reference, reference])

    provider_two = FakeProvider(remaining_responses)
    summary_two = HostedControlledFaultPlanRunner(
        provider=provider_two
    ).execute(
        plan_path=plan_path,
        run_dir=tmp_path / "resume-run",
        model_name="fake-model",
        max_output_tokens=2000,
        resume=True,
    )

    assert summary_two["execution_status"] == "COMPLETED"
    assert summary_two["model_calls_used"] == 10
    assert len(provider_two.requests) == 8

    first_checkpoint = json.loads(
        (
            tmp_path
            / "resume-run/execution/checkpoints/pair-000001.json"
        ).read_text(encoding="utf-8")
    )
    assert first_checkpoint["guarded_stage"] == "COMPLETED"


def test_resume_does_not_retry_terminal_source_failure(
    tmp_path: Path,
) -> None:
    plan_path = _small_plan(tmp_path, pair_count=1)

    provider_one = FakeProvider(
        [RuntimeError("terminal source failure")]
    )
    summary_one = HostedControlledFaultPlanRunner(
        provider=provider_one
    ).execute(
        plan_path=plan_path,
        run_dir=tmp_path / "failed-run",
        model_name="fake-model",
        max_output_tokens=2000,
    )

    assert summary_one["model_calls_used"] == 1
    assert summary_one["execution_status"] == "COMPLETED"

    provider_two = FakeProvider([])
    summary_two = HostedControlledFaultPlanRunner(
        provider=provider_two
    ).execute(
        plan_path=plan_path,
        run_dir=tmp_path / "failed-run",
        model_name="fake-model",
        max_output_tokens=2000,
        resume=True,
    )

    assert summary_two["model_calls_used"] == 1
    assert len(provider_two.requests) == 0

def test_resume_rejects_model_or_token_changes(
    tmp_path: Path,
) -> None:
    plan_path = _small_plan(tmp_path, pair_count=2)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    first = plan["pairs"][0]
    reference = render_reference_configuration(
        generate_task(first["task_index"])
    )

    provider = FakeProvider(
        [
            reference,
            reference,
            KeyboardInterrupt(),
        ]
    )

    with pytest.raises(KeyboardInterrupt):
        HostedControlledFaultPlanRunner(
            provider=provider
        ).execute(
            plan_path=plan_path,
            run_dir=tmp_path / "incomplete",
            model_name="fake-model",
            max_output_tokens=2000,
        )

    with pytest.raises(ValueError, match="model_name"):
        HostedControlledFaultPlanRunner(
            provider=FakeProvider([])
        ).execute(
            plan_path=plan_path,
            run_dir=tmp_path / "incomplete",
            model_name="different-model",
            max_output_tokens=2000,
            resume=True,
        )

    with pytest.raises(ValueError, match="max_output_tokens"):
        HostedControlledFaultPlanRunner(
            provider=FakeProvider([])
        ).execute(
            plan_path=plan_path,
            run_dir=tmp_path / "incomplete",
            model_name="fake-model",
            max_output_tokens=1000,
            resume=True,
        )