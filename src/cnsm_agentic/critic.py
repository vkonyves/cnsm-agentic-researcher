from __future__ import annotations

import json

from agents import Agent, RunConfig, Runner

from cnsm_agentic.model_policy import (
    RetryObserver,
    research_model_settings,
)
from cnsm_agentic.schemas import (
    CriticReport,
    DiscoveryReport,
    ResearchPlan,
)


def build_critic_prompt(
    discovery: DiscoveryReport,
    plan: ResearchPlan,
) -> str:
    discovery_json = json.dumps(
        discovery.model_dump(mode="json"),
        indent=2,
    )

    plan_json = json.dumps(
        plan.model_dump(mode="json"),
        indent=2,
    )

    return f"""
Act as a hostile but fair scientific reviewer.

Discovery evidence:
{discovery_json}

Candidate plan:
{plan_json}

For each candidate:

- identify unsupported claims;
- check whether every proposed metric is computable;
- identify unavailable or inappropriate baselines;
- assess CPU and API feasibility;
- identify leakage risks;
- identify ambiguity and underspecified decisions;
- assess reproducibility;
- assess scientific novelty.

Require explicit repairs rather than silently fixing weak work.

Select a candidate only when the available evidence supports proceeding
to dataset acquisition. Otherwise, return no recommendation.
""".strip()


async def critique_research_plan(
    discovery: DiscoveryReport,
    plan: ResearchPlan,
    model: str,
    max_turns: int,
    group_id: str,
    retry_observer: RetryObserver | None = None,
) -> CriticReport:
    agent = Agent(
        name="Adversarial Scientific Reviewer",
        model=model,
        model_settings=research_model_settings(
            retry_observer=retry_observer,
        ),
        instructions=(
            "Act as an independent scientific reviewer. "
            "Be sceptical, evidence-driven and specific. "
            "Do not silently repair weak work."
        ),
        output_type=CriticReport,
    )

    result = await Runner.run(
        agent,
        build_critic_prompt(
            discovery=discovery,
            plan=plan,
        ),
        max_turns=max_turns,
        run_config=RunConfig(
            workflow_name="CNSM adversarial review",
            group_id=group_id,
            trace_metadata={
                "stage": "critic",
            },
        ),
    )

    return result.final_output