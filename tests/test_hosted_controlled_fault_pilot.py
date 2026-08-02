from pathlib import Path

import json

from cnsm_agentic.autonomous_research.hosted_controlled_fault_pilot import (
    HOSTED_CONTROLLED_FAULT_ADAPTER_FAMILY,
    HostedControlledFaultPilot,
    _repair_prompt,
    _source_prompt,
    controlled_fault_plan_issues,
)
from cnsm_agentic.autonomous_research.model_providers import ModelCallResult

from cnsm_agentic.autonomous_research.netops_generate_validate_repair import (
    generate_task,
    inject_controlled_fault,
    render_reference_configuration,
)


class FakeProvider:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.requests = []

    def call(self, request):
        self.requests.append(request)
        response_text = self.responses.pop(0)
        return ModelCallResult(
            provider="openai_responses",
            requested_model=request.model,
            resolved_model=request.model,
            response_text=response_text,
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


def _plan() -> dict:
    return {
        "study_id": "controlled-test",
        "adapter_family": HOSTED_CONTROLLED_FAULT_ADAPTER_FAMILY,
        "execution_mode": "scientific_pilot",
        "conditions": ["baseline", "guarded"],
        "task_indices": [1],
        "maximum_model_calls": 2,
        "model_provider": "openai_responses",
        "model_name": "fake-model",
        "maximum_attempts_per_call": 1,
        "reasoning_effort": "minimal",
        "max_output_tokens": 2000,
    }


def test_plan_validation() -> None:
    assert controlled_fault_plan_issues(_plan()) == []
    bad = _plan()
    bad["maximum_model_calls"] = 1
    assert controlled_fault_plan_issues(bad)


def test_task_intent_is_preserved_verbatim_in_prompts() -> None:
    task = generate_task(8)

    injected = inject_controlled_fault(
        task,
        render_reference_configuration(task),
    )

    source_prompt = _source_prompt(task)
    repair_prompt = _repair_prompt(
        task,
        injected["injected_configuration"],
        injected["injected_validation"],
    )

    expected_line = f"Intent and safety policy: {task['intent']}"

    assert source_prompt.splitlines()[0] == expected_line
    assert repair_prompt.splitlines()[0] == expected_line

    assert "uplink1 is shut down" in source_prompt
    assert "must always be up. Then shut edge1 down" in source_prompt
    assert "is shutdown" not in source_prompt
    assert "up.Then" not in source_prompt


def test_hosted_controlled_fault_pilot(tmp_path: Path) -> None:
    source = "\n".join(
        [
            "interface edge1 admin down",
            "interface edge1 mtu 1600",
            "interface edge1 vlan 30",
            "interface edge1 admin up",
        ]
    )
    repaired = source
    provider = FakeProvider([source, repaired])

    summary = HostedControlledFaultPilot(provider=provider).execute(
        plan=_plan(),
        execution_dir=tmp_path / "execution",
    )

    assert summary["model_calls_used"] == 2
    assert summary["complete_pairs"] == 1
    assert summary["baseline_successes"] == 0
    assert summary["guarded_successes"] == 1
    assert summary["n_10_guarded_only"] == 1
    assert len(provider.requests) == 2

    assert (
        tmp_path / "execution/faults/task-000001-fault.json"
    ).exists()
    assert (
        tmp_path / "execution/prompts/task-000001-guarded-repair.txt"
    ).exists()

    fault_record = json.loads(
        (
            tmp_path
            / "execution/faults/task-000001-fault.json"
        ).read_text(encoding="utf-8")
    )

    assert fault_record["task_id"] == "task-000001"
    assert fault_record["pair_id"] == "pair-000001"

    baseline = json.loads(
        (
            tmp_path
            / "execution/scoring/task-000001-baseline.json"
        ).read_text(encoding="utf-8")
    )

    guarded = json.loads(
        (
            tmp_path
            / "execution/scoring/task-000001-guarded.json"
        ).read_text(encoding="utf-8")
    )

    assert (
        baseline["condition_input_candidate_sha256"]
        == guarded["condition_input_candidate_sha256"]
        == fault_record["shared_injected_candidate_sha256"]
    )