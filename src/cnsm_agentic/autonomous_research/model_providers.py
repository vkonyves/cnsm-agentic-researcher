from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Protocol


@dataclass(frozen=True)
class ModelCallRequest:
    provider: str
    model: str
    prompt: str
    instructions: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    reasoning_effort: str | None = None
    metadata: dict[str, str] | None = None

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "prompt": self.prompt,
            "instructions": self.instructions,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "reasoning_effort": self.reasoning_effort,
            "metadata": dict(sorted((self.metadata or {}).items())),
        }

    def cache_key_sha256(self) -> str:
        payload = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return sha256(payload).hexdigest()


@dataclass(frozen=True)
class ModelCallResult:
    provider: str
    requested_model: str
    resolved_model: str
    response_text: str
    response_id: str | None
    request_id: str | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    latency_ms: int
    attempt_count: int
    cache_status: str
    cache_key_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HostedModelProvider(Protocol):
    provider_name: str

    def call(self, request: ModelCallRequest) -> ModelCallResult:
        ...


class JsonFileCallCache:
    """Small deterministic cache for bounded pilots.

    The cache stores only completed call results. Cache keys include the exact
    prompt, model, instructions, generation parameters, and metadata.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str) -> ModelCallResult | None:
        path = self._path(key)
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        data["cache_status"] = "HIT"
        data["latency_ms"] = 0
        data["attempt_count"] = 0
        return ModelCallResult(**data)

    def put(self, result: ModelCallResult) -> None:
        path = self._path(result.cache_key_sha256)
        stored = result.to_dict()
        stored["cache_status"] = "MISS"
        path.write_text(
            json.dumps(
                stored,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )


class OpenAIResponsesProvider:
    provider_name = "openai_responses"

    def __init__(
        self,
        *,
        client: Any | None = None,
        cache: JsonFileCallCache | None = None,
        maximum_attempts: int = 3,
        retry_backoff_seconds: float = 1.0,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        if maximum_attempts <= 0:
            raise ValueError("maximum_attempts must be positive")
        if retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be nonnegative")

        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "The openai package is required for hosted model calls."
                ) from exc
            client = OpenAI()

        self.client = client
        self.cache = cache
        self.maximum_attempts = maximum_attempts
        self.retry_backoff_seconds = retry_backoff_seconds
        self.sleep_fn = sleep_fn

    @staticmethod
    def _usage_value(usage: Any, field: str) -> int | None:
        value = getattr(usage, field, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        return None

    def call(self, request: ModelCallRequest) -> ModelCallResult:
        if request.provider != self.provider_name:
            raise ValueError(
                f"Request provider must be {self.provider_name!r}."
            )
        if not request.model.strip():
            raise ValueError("Model identifier must be non-empty.")
        if not request.prompt.strip():
            raise ValueError("Prompt must be non-empty.")

        cache_key = request.cache_key_sha256()
        if self.cache is not None:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        kwargs: dict[str, Any] = {
            "model": request.model,
            "input": request.prompt,
            "store": False,
        }
        if request.instructions is not None:
            kwargs["instructions"] = request.instructions
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            kwargs["max_output_tokens"] = request.max_output_tokens
        if request.reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": request.reasoning_effort}
        if request.metadata:
            kwargs["metadata"] = request.metadata

        last_error: Exception | None = None

        for attempt in range(1, self.maximum_attempts + 1):
            started_ns = time.perf_counter_ns()
            try:
                response = self.client.responses.create(**kwargs)
                latency_ms = max(
                    0,
                    (time.perf_counter_ns() - started_ns) // 1_000_000,
                )
                text = str(getattr(response, "output_text", "") or "")
                usage = getattr(response, "usage", None)
                if not text.strip():
                    incomplete = getattr(response, "incomplete_details", None)
                    incomplete_reason = getattr(incomplete, "reason", None)
                    response_id = getattr(response, "id", None)
                    status = getattr(response, "status", None)
                    input_tokens = self._usage_value(usage, "input_tokens")
                    output_tokens = self._usage_value(usage, "output_tokens")
                    total_tokens = self._usage_value(usage, "total_tokens")
                    raise RuntimeError(
                        "OpenAI response contained no output text"
                        f"; response_id={response_id!r}"
                        f"; status={status!r}"
                        f"; incomplete_reason={incomplete_reason!r}"
                        f"; input_tokens={input_tokens!r}"
                        f"; output_tokens={output_tokens!r}"
                        f"; total_tokens={total_tokens!r}."
                    )

                result = ModelCallResult(
                    provider=self.provider_name,
                    requested_model=request.model,
                    resolved_model=str(
                        getattr(response, "model", request.model)
                    ),
                    response_text=text,
                    response_id=(
                        str(getattr(response, "id"))
                        if getattr(response, "id", None)
                        else None
                    ),
                    request_id=(
                        str(getattr(response, "_request_id"))
                        if getattr(response, "_request_id", None)
                        else None
                    ),
                    input_tokens=self._usage_value(
                        usage, "input_tokens"
                    ),
                    output_tokens=self._usage_value(
                        usage, "output_tokens"
                    ),
                    total_tokens=self._usage_value(
                        usage, "total_tokens"
                    ),
                    latency_ms=int(latency_ms),
                    attempt_count=attempt,
                    cache_status="MISS",
                    cache_key_sha256=cache_key,
                )
                if self.cache is not None:
                    self.cache.put(result)
                return result
            except Exception as exc:
                last_error = exc
                if attempt == self.maximum_attempts:
                    break
                self.sleep_fn(
                    self.retry_backoff_seconds * (2 ** (attempt - 1))
                )

        assert last_error is not None
        raise RuntimeError(
            "Hosted model call failed after "
            f"{self.maximum_attempts} attempts: {last_error}"
        ) from last_error
