from __future__ import annotations

import json

from agents import Agent, RunConfig, Runner, WebSearchTool

from cnsm_agentic.config import PilotConfig, ResourceSpec
from cnsm_agentic.model_policy import (
    RetryObserver,
    research_model_settings,
)
from cnsm_agentic.schemas import (
    DiscoveryReport,
    VerifiedResource,
)


def build_resource_discovery_prompt(
    config: PilotConfig,
    resource: ResourceSpec,
) -> str:
    return f"""
Investigate exactly one candidate resource for an autonomous
network-management research project.

Research topic:
{config.topic_scope}

Candidate resource:
{json.dumps(resource.model_dump(), indent=2)}

Search priority:
1. Primary paper or official arXiv record.
2. Official repository.
3. Official dataset or project page.
4. Official documentation.
5. Secondary sources only when primary sources cannot answer a question.

Verify as many of the following as possible:
- exact paper identifier and title;
- official repository;
- concrete dataset URLs;
- dataset downloadability;
- licence;
- task types;
- availability of gold answers;
- availability of evaluation code;
- likely CPU feasibility.

Structured-output rules:

- Return information only for the single resource named above.
- repository_url must contain one complete http:// or https:// URL,
  or null.
- dataset_urls must contain only complete http:// or https:// URLs.
- Use an empty list if no concrete dataset URL is verified.
- Never place prose or repository-relative paths in dataset_urls.
- Put paths, filenames and explanations in dataset_location_notes.
- Every EvidenceSource.url must contain exactly one complete URL.
- Treat absence of evidence as unresolved.
- Never infer a licence merely because a repository is public.
- Every verified claim must be supported by an evidence source.
- Do not design experiments.
- Do not fabricate URLs or benchmark details.

Keep the result concise:
- at most 5 verified claims;
- at most 5 unresolved questions;
- at most 5 evidence sources;
- each notes field below 60 words;
- do not repeat facts between fields.
"""

async def discover_one_resource(
    config: PilotConfig,
    resource: ResourceSpec,
    model: str,
    max_turns: int,
    group_id: str,
    retry_observer: RetryObserver | None = None,
) -> VerifiedResource:
    agent = Agent(
        name=f"Resource Verifier: {resource.name}",
        model=model,
        model_settings=research_model_settings(
            retry_observer=retry_observer,
        ),
        instructions=(
            "Investigate only the supplied benchmark. "
            "Use primary and official sources wherever possible. "
            "Expose uncertainty and keep URLs separate from prose."
        ),
        tools=[
            WebSearchTool(
                search_context_size="low",
            )
        ],
        output_type=VerifiedResource,
    )

    result = await Runner.run(
        agent,
        build_resource_discovery_prompt(
            config,
            resource,
        ),
        max_turns=min(
            max_turns,
            8,
        ),
        run_config=RunConfig(
            workflow_name="CNSM single-resource discovery",
            group_id=group_id,
            trace_metadata={
                "stage": "discovery",
                "resource": resource.name,
            },
        ),
    )

    return result.final_output


def assemble_discovery_report(
    config: PilotConfig,
    resources: list[VerifiedResource],
    failed_resources: list[str],
) -> DiscoveryReport:
    searched_resources = [
        resource.name
        for resource in config.candidate_resources
    ]

    next_actions: list[str] = []

    for resource in resources:
        next_actions.extend(resource.unresolved_questions)

    if failed_resources:
        next_actions.append(
            "Retry discovery for resources that failed because of "
            "network or provider errors."
        )

    return DiscoveryReport(
        scope_summary=(
            "Independent, web-backed verification of candidate NetOps "
            "benchmarks. Each benchmark was investigated in a separate "
            "model call to reduce timeout risk."
        ),
        searched_resources=searched_resources,
        resources=resources,
        rejected_or_unresolved_leads=[
            f"Discovery failed before validation: {name}"
            for name in failed_resources
        ],
        discovery_limitations=[
            "Web discovery does not replace deterministic URL, licence, "
            "repository and downloaded-file validation.",
            "Publisher pages may be unavailable because of access controls.",
        ],
        next_verification_actions=next_actions,
    )
