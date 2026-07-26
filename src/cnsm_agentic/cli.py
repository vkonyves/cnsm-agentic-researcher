from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from agents import flush_traces
from agents.retry import RetryPolicyContext
from cnsm_agentic.model_policy import RetryObserver

from cnsm_agentic.config import load_config
from cnsm_agentic.critic import critique_research_plan
from cnsm_agentic.discovery import (
    assemble_discovery_report,
    discover_one_resource,
)
from cnsm_agentic.formulation import formulate_research_plan
from cnsm_agentic.openai_client import configure_openai_client
from cnsm_agentic.planner import generate_pilot_plan
from cnsm_agentic.provenance import RunStore, sha256_text
from cnsm_agentic.settings import load_settings


def make_run_id(run_name: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    safe_name = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in run_name
    )

    return f"{timestamp}-{safe_name}"


def make_group_id() -> str:
    return f"cnsm-{uuid.uuid4().hex}"


def require_api_key(settings: object) -> bool:
    if not settings.api_key_present:
        print(
            "OPENAI_API_KEY is missing. Add it to .env.",
            file=sys.stderr,
        )
        return False

    return True


async def run_plan(config_path: Path) -> int:
    settings = load_settings()

    if not require_api_key(settings):
        return 2

    # Installs the custom AsyncOpenAI client with the longer timeout.
    configure_openai_client()

    config_text = config_path.read_text(encoding="utf-8")
    config = load_config(config_path)

    run_id = make_run_id(config.run_name)
    store = RunStore(settings.artifact_root, run_id)

    store.write_text(
        "config.source.yaml",
        config_text,
    )

    store.write_json(
        "config.snapshot.json",
        config.model_dump(),
    )

    store.append_event(
        "run_started",
        {
            "command": "plan",
            "config_path": str(config_path),
            "config_sha256": sha256_text(config_text),
            "model": settings.worker_model,
            "experiment_family": config.experiment_family,
        },
    )

    try:
        plan = await generate_pilot_plan(
            config=config,
            model=settings.worker_model,
            max_turns=settings.max_agent_turns,
        )

        output_path = store.write_json(
            "pilot_plan.json",
            plan.model_dump(mode="json"),
        )

        store.append_event(
            "run_completed",
            {
                "output_path": str(output_path),
            },
        )

        print(
            json.dumps(
                plan.model_dump(mode="json"),
                indent=2,
                ensure_ascii=False,
            )
        )

        print(f"\nSaved run: {store.path}")
        return 0

    except Exception as exc:
        store.append_event(
            "run_failed",
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise

    finally:
        flush_traces()


async def run_research_pilot(config_path: Path) -> int:
    settings = load_settings()

    if not require_api_key(settings):
        return 2

    # Must be called after load_settings(), because load_settings()
    # loads values from .env.
    configure_openai_client()

    config_text = config_path.read_text(encoding="utf-8")
    config = load_config(config_path)

    group_id = make_group_id()
    run_id = make_run_id(config.run_name)

    store = RunStore(
        settings.artifact_root,
        run_id,
    )

    def make_retry_observer(
        *,
        stage: str,
        resource: str | None = None,
    ) -> RetryObserver:
        async def observe_retry(
            context: RetryPolicyContext,
            should_retry: bool,
            delay: float | None,
            reason: str | None,
        ) -> None:
            error = context.error
            normalized = getattr(
                context,
                "normalized",
                None,
            )

            store.append_event(
                "model_request_retry_decision",
                stage=stage,
                resource=resource,
                attempt=getattr(
                    context,
                    "attempt",
                    None,
                ),
                max_retries=getattr(
                    context,
                    "max_retries",
                    None,
                ),
                should_retry=should_retry,
                delay_seconds=delay,
                decision_reason=reason,
                error_type=(
                    type(error).__name__
                    if error is not None
                    else None
                ),
                error_message=(
                    str(error)
                    if error is not None
                    else None
                ),
                request_id=getattr(
                    normalized,
                    "request_id",
                    None,
                ),
                status_code=getattr(
                    normalized,
                    "status_code",
                    None,
                ),
                error_code=getattr(
                    normalized,
                    "error_code",
                    None,
                ),
                retry_after_seconds=getattr(
                    normalized,
                    "retry_after",
                    None,
                ),
                is_timeout=getattr(
                    normalized,
                    "is_timeout",
                    None,
                ),
                is_network_error=getattr(
                    normalized,
                    "is_network_error",
                    None,
                ),
            )

        return observe_retry










    store.write_text(
        "config.source.yaml",
        config_text,
    )

    store.write_json(
        "config.snapshot.json",
        config.model_dump(),
    )

    store.append_event(
        "run_started",
        {
            "command": "research-pilot",
            "config_path": str(config_path),
            "config_sha256": sha256_text(config_text),
            "group_id": group_id,
            "research_model": settings.research_model,
            "review_model": settings.review_model,
            "experiment_family": config.experiment_family,
        },
    )

    current_stage = "initialisation"

    try:
        # ----------------------------------------------------------
        # Stage 1: discovery and verification
        # ----------------------------------------------------------
        current_stage = "discovery"

        store.append_event(
            "stage_started",
            {
                "stage": current_stage,
            },
        )

        verified_resources = []
        failed_resources = []

        for index, resource in enumerate(
            config.candidate_resources,
            start=1,
        ):

            store.append_event(
                "resource_discovery_started",
                {
                    "stage": current_stage,
                    "resource": resource.name,
                    "resource_index": index,
                },
            )

            try:
                verified = await discover_one_resource(
                    config=config,
                    resource=resource,
                    model=settings.research_model,
                    max_turns=settings.max_agent_turns,
                    group_id=group_id,
                    retry_observer=make_retry_observer(
                        stage="discovery",
                        resource=resource.name,
                    ),
                )

                verified_resources.append(verified)

                resource_filename = (
                    resource.name.lower()
                    .replace(" ", "-")
                    .replace("/", "-")
                )

                resource_path = store.write_json(
                    f"discovery/resources/{resource_filename}.json",
                    verified.model_dump(mode="json"),
                )

                store.append_event(
                    "resource_discovery_completed",
                    {
                        "stage": current_stage,
                        "resource": resource.name,
                        "output_path": str(resource_path),
                    },
                )

            except Exception as resource_exc:
                failed_resources.append(resource.name)

                store.append_event(
                    "resource_discovery_failed",
                    {
                        "stage": current_stage,
                        "resource": resource.name,
                        "error_type": type(resource_exc).__name__,
                        "error": str(resource_exc),
                    },
                )

                print(
                    f"Discovery failed for {resource.name}: "
                    f"{type(resource_exc).__name__}: {resource_exc}",
                    file=sys.stderr,
                )

        discovery = assemble_discovery_report(
            config=config,
            resources=verified_resources,
            failed_resources=failed_resources,
        )

        discovery_path = store.write_json(
            "discovery/discovery_report.json",
            discovery.model_dump(mode="json"),
        )

        if not verified_resources:
            raise RuntimeError(
                "No candidate resource was successfully discovered. "
                "See the per-resource failure events."
            )

        store.append_event(
            "stage_completed",
            {
                "stage": current_stage,
                "output_path": str(discovery_path),
            },
        )

        # ----------------------------------------------------------
        # Stage 2: evidence-grounded formulation
        # ----------------------------------------------------------
        current_stage = "formulation"

        store.append_event(
            "stage_started",
            {
                "stage": current_stage,
            },
        )

        plan = await formulate_research_plan(
            config=config,
            discovery=discovery,
            model=settings.research_model,
            max_turns=settings.max_agent_turns,
            group_id=group_id,
            retry_observer=make_retry_observer(
                stage="formulation",
            ),
        )

        plan_path = store.write_json(
            "formulation/research_plan.json",
            plan.model_dump(mode="json"),
        )

        store.append_event(
            "stage_completed",
            {
                "stage": current_stage,
                "output_path": str(plan_path),
            },
        )

        # ----------------------------------------------------------
        # Stage 3: adversarial criticism
        # ----------------------------------------------------------
        current_stage = "critic"

        store.append_event(
            "stage_started",
            {
                "stage": current_stage,
            },
        )

        critic = await critique_research_plan(
            discovery=discovery,
            plan=plan,
            model=settings.review_model,
            max_turns=settings.max_agent_turns,
            group_id=group_id,
            retry_observer=make_retry_observer(
                stage="critic",
            ),
        )

        critic_path = store.write_json(
            "review/critic_report.json",
            critic.model_dump(mode="json"),
        )

        store.append_event(
            "stage_completed",
            {
                "stage": current_stage,
                "output_path": str(critic_path),
            },
        )

        # ----------------------------------------------------------
        # Final pilot summary
        # ----------------------------------------------------------
        current_stage = "summary"

        summary = {
            "recommended_candidate": critic.recommended_candidate,
            "proceed_to_dataset_acquisition": (
                critic.proceed_to_dataset_acquisition
            ),
            "blocking_issues": critic.blocking_issues,
            "run_directory": str(store.path),
            "group_id": group_id,
        }

        summary_path = store.write_json(
            "summary.json",
            summary,
        )

        store.append_event(
            "run_completed",
            {
                **summary,
                "summary_path": str(summary_path),
            },
        )

        print(
            json.dumps(
                summary,
                indent=2,
                ensure_ascii=False,
            )
        )

        print(f"\nSaved run: {store.path}")
        return 0

    except Exception as exc:
        store.append_event(
            "run_failed",
            {
                "stage": current_stage,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )

        print(
            f"\nResearch pilot failed during stage: {current_stage}",
            file=sys.stderr,
        )
        print(
            f"Failed run preserved at: {store.path}",
            file=sys.stderr,
        )

        raise

    finally:
        flush_traces()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CNSM agentic researcher",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    plan_parser = subparsers.add_parser(
        "plan",
        help="Run the original structured planning smoke test.",
    )

    plan_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the pilot YAML configuration.",
    )

    research_parser = subparsers.add_parser(
        "research-pilot",
        help=(
            "Run resource discovery, evidence-grounded formulation, "
            "and adversarial review."
        ),
    )

    research_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the research-pilot YAML configuration.",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "plan":
        raise SystemExit(
            asyncio.run(
                run_plan(args.config)
            )
        )

    if args.command == "research-pilot":
        raise SystemExit(
            asyncio.run(
                run_research_pilot(args.config)
            )
        )

    raise SystemExit(
        f"Unsupported command: {args.command}"
    )


if __name__ == "__main__":
    main()
