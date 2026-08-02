from pathlib import Path
from types import SimpleNamespace

import pytest

from cnsm_agentic.autonomous_research.model_providers import (
    JsonFileCallCache,
    ModelCallRequest,
    OpenAIResponsesProvider,
)


class FakeResponses:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, outcomes):
        self.responses = FakeResponses(outcomes)


def _response(text="interface eth1 admin up"):
    return SimpleNamespace(
        id="resp_test",
        model="gpt-test-2026-08-01",
        output_text=text,
        _request_id="req_test",
        usage=SimpleNamespace(
            input_tokens=12,
            output_tokens=6,
            total_tokens=18,
        ),
    )


def test_request_cache_key_is_deterministic() -> None:
    first = ModelCallRequest(
        provider="openai_responses",
        model="gpt-test",
        prompt="Generate config.",
        metadata={"b": "2", "a": "1"},
    )
    second = ModelCallRequest(
        provider="openai_responses",
        model="gpt-test",
        prompt="Generate config.",
        metadata={"a": "1", "b": "2"},
    )
    assert first.cache_key_sha256() == second.cache_key_sha256()


def test_openai_provider_records_provenance() -> None:
    client = FakeClient([_response()])
    provider = OpenAIResponsesProvider(
        client=client,
        maximum_attempts=1,
    )
    result = provider.call(
        ModelCallRequest(
            provider="openai_responses",
            model="gpt-test",
            prompt="Generate config.",
            instructions="Return configuration only.",
            max_output_tokens=100,
            reasoning_effort="minimal",
        )
    )

    assert result.response_text == "interface eth1 admin up"
    assert result.response_id == "resp_test"
    assert result.request_id == "req_test"
    assert result.requested_model == "gpt-test"
    assert result.resolved_model == "gpt-test-2026-08-01"
    assert result.input_tokens == 12
    assert result.output_tokens == 6
    assert result.total_tokens == 18
    assert result.attempt_count == 1
    assert result.cache_status == "MISS"
    assert client.responses.calls[0]["store"] is False
    assert client.responses.calls[0]["reasoning"] == {
        "effort": "minimal"
    }


def test_openai_provider_retries() -> None:
    sleeps = []
    client = FakeClient([
        RuntimeError("temporary"),
        _response(),
    ])
    provider = OpenAIResponsesProvider(
        client=client,
        maximum_attempts=2,
        retry_backoff_seconds=0.25,
        sleep_fn=sleeps.append,
    )

    result = provider.call(
        ModelCallRequest(
            provider="openai_responses",
            model="gpt-test",
            prompt="Generate config.",
        )
    )

    assert result.attempt_count == 2
    assert len(client.responses.calls) == 2
    assert sleeps == [0.25]


def test_openai_provider_cache_prevents_second_call(
    tmp_path: Path,
) -> None:
    cache = JsonFileCallCache(tmp_path / "cache")
    client = FakeClient([_response()])
    provider = OpenAIResponsesProvider(
        client=client,
        cache=cache,
        maximum_attempts=1,
    )
    request = ModelCallRequest(
        provider="openai_responses",
        model="gpt-test",
        prompt="Generate config.",
    )

    first = provider.call(request)
    second = provider.call(request)

    assert first.cache_status == "MISS"
    assert second.cache_status == "HIT"
    assert second.attempt_count == 0
    assert second.latency_ms == 0
    assert len(client.responses.calls) == 1
    assert second.response_text == first.response_text


def test_openai_provider_rejects_empty_output() -> None:
    response = _response("   ")
    response.status = "incomplete"
    response.incomplete_details = SimpleNamespace(
        reason="max_output_tokens"
    )
    provider = OpenAIResponsesProvider(
        client=FakeClient([response]),
        maximum_attempts=1,
    )

    with pytest.raises(
        RuntimeError,
        match="incomplete_reason='max_output_tokens'",
    ):
        provider.call(
            ModelCallRequest(
                provider="openai_responses",
                model="gpt-test",
                prompt="Generate config.",
            )
        )


def test_openai_provider_rejects_wrong_provider() -> None:
    provider = OpenAIResponsesProvider(
        client=FakeClient([_response()]),
        maximum_attempts=1,
    )

    with pytest.raises(ValueError, match="provider"):
        provider.call(
            ModelCallRequest(
                provider="other",
                model="gpt-test",
                prompt="Generate config.",
            )
        )
