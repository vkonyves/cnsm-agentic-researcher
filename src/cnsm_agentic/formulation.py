from __future__ import annotations

import json
from agents import Agent, RunConfig, Runner
from cnsm_agentic.config import PilotConfig
from cnsm_agentic.schemas import DiscoveryReport, ResearchPlan
from cnsm_agentic.model_policy import (
    RetryObserver,
    research_model_settings,
)

def build_formulation_prompt(config: PilotConfig, discovery: DiscoveryReport) -> str:
    return f"""
Formulate evidence-grounded candidate experiments.
Topic: {config.topic_scope}
Experiment family: {config.experiment_family}
Discovery report: {json.dumps(discovery.model_dump(mode='json'), indent=2)}
Constraints: {json.dumps(config.pilot_constraints, indent=2)}

Produce 2 or 3 materially different candidates tied to verified resources. Use verified facts only as facts. List assumptions and required checks. Do not use synthetic data as the primary dataset. Operationalise every metric. Keep the work CPU-feasible. Make the recommendation provisional and list dependencies. Do not fabricate results, fields, licences or baselines.
"""

async def formulate_research_plan(
    config: PilotConfig,
    discovery: DiscoveryReport,
    model: str,
    max_turns: int,
    group_id: str,
    retry_observer: RetryObserver | None = None,
) -> ResearchPlan:
    agent = Agent(
        name="Evidence-Grounded Method Designer",
        model=model,
        model_settings=research_model_settings(
            retry_observer=retry_observer,
        ),
        instructions=(
            "Design testable scientific experiments only from supplied "
            "verified evidence. Expose assumptions and operationalise "
            "metrics."
        ),
        output_type=ResearchPlan,
    )

    result = await Runner.run(
        agent,
        build_formulation_prompt(
            config,
            discovery,
        ),
        max_turns=max_turns,
        run_config=RunConfig(
            workflow_name="CNSM experiment formulation",
            group_id=group_id,
            trace_metadata={
                "stage": "formulation",
            },
        ),
    )

    return result.final_output
