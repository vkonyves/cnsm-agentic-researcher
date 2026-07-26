from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from agents import (
    ModelRetryBackoffSettings,
    ModelRetrySettings,
    ModelSettings,
    retry_policies,
)
from agents.retry import RetryDecision, RetryPolicyContext


RetryObserver = Callable[
    [
        RetryPolicyContext,
        bool,
        float | None,
        str | None,
    ],
    None | Awaitable[None],
]


async def _call_maybe_async(
    function: Callable[..., Any],
    *args: Any,
) -> Any:
    """
    Call a synchronous or asynchronous function and return its result.
    """
    result = function(*args)

    if inspect.isawaitable(result):
        return await result

    return result


def _decision_fields(
    decision: bool | RetryDecision,
) -> tuple[bool, float | None, str | None]:
    """
    Extract logging-friendly fields from an SDK retry decision.
    """
    if isinstance(decision, RetryDecision):
        return (
            decision.retry,
            decision.delay,
            decision.reason,
        )

    return bool(decision), None, None


def _observable_retry_policy(
    observer: RetryObserver | None = None,
) -> Callable[
    [RetryPolicyContext],
    Awaitable[bool | RetryDecision],
]:
    """
    Build the normal retry policy while exposing retry decisions to an
    optional provenance observer.

    The original SDK decision object is returned unchanged so provider
    retry constraints and delay information remain intact.
    """
    base_policy = retry_policies.any(
        retry_policies.provider_suggested(),
        retry_policies.retry_after(),
        retry_policies.network_error(),
        retry_policies.http_status(
            [
                408,
                409,
                429,
                500,
                502,
                503,
                504,
            ]
        ),
    )

    async def policy(
        context: RetryPolicyContext,
    ) -> bool | RetryDecision:
        decision = await _call_maybe_async(
            base_policy,
            context,
        )

        should_retry, delay, reason = _decision_fields(
            decision
        )

        if observer is not None:
            await _call_maybe_async(
                observer,
                context,
                should_retry,
                delay,
                reason,
            )

        return decision

    return policy


def research_model_settings(
    retry_observer: RetryObserver | None = None,
) -> ModelSettings:
    """
    Return model settings for long-running autonomous research stages.

    Up to three retries are allowed for transient network, timeout,
    throttling and selected provider/server errors.
    """
    return ModelSettings(
        max_tokens=8000,
        retry=ModelRetrySettings(
            max_retries=3,
            backoff=ModelRetryBackoffSettings(
                initial_delay=2.0,
                max_delay=20.0,
                multiplier=2.0,
                jitter=True,
            ),
            policy=_observable_retry_policy(
                observer=retry_observer,
            ),
        ),
    )