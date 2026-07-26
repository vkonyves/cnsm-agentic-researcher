from __future__ import annotations

import json

from agents import Agent, Runner

from cnsm_agentic.adapters import get_adapter
from cnsm_agentic.config import PilotConfig
from cnsm_agentic.schemas import PilotResearchPlan


def build_prompt(config: PilotConfig) -> str:
    adapter = get_adapter(config.experiment_family)
    resources = [resource.model_dump() for resource in config.candidate_resources]

    return f"""
You are performing an architecture smoke test for a future autonomous scientific
researcher. Generate a cautious pilot research plan, not a final scientific
decision.

Topic scope:
{config.topic_scope}

Experiment family:
{config.experiment_family}

Unverified candidate resources:
{json.dumps(resources, indent=2)}

Pilot constraints:
{json.dumps(config.pilot_constraints, indent=2)}

Adapter guidance:
{adapter.planning_guidance(config)}

Requirements:
- Produce 2 or 3 materially different candidate experiment directions.
- Tie every candidate explicitly to one or more named candidate resources.
- Do not construct a synthetic dataset as the primary scientific dataset.
- Synthetic data may be proposed only for a small architecture smoke test.
- Clearly distinguish:
  1. verified facts supplied in the configuration,
  2. unverified assumptions,
  3. required machine-verification actions.
- Do not claim that repositories, licences, datasets, task formats, gold labels,
  or evaluation code exist unless that information was supplied in the configuration.
- Do not recommend a final scientific direction without verified evidence.
- Make the recommendation explicitly provisional.
- Every metric must include:
  - a precise definition,
  - the required gold standard or reference,
  - whether it is automatically computable.
- Prefer deterministic and automatically computable metrics.
- Include an estimated API call pattern for each candidate.
- Keep every candidate feasible on a laptop without a GPU.
- Do not fabricate results, references, repository contents, licences, task counts,
  benchmark fields, or baseline performance.
- Human actions after final freeze must not include scientific editing, metric selection,
  result interpretation, citation correction, or code repair.
- The output must fit the supplied structured schema.
"""


async def generate_pilot_plan(
    config: PilotConfig,
    model: str,
    max_turns: int,
) -> PilotResearchPlan:
    agent = Agent(
        name="Pilot Research Planner",
        model=model,
        instructions=(
            "You design cautious, testable research plans. Never invent experimental "
            "results or claim that an unverified dataset is available."
        ),
        output_type=PilotResearchPlan,
    )
    result = await Runner.run(
        agent,
        build_prompt(config),
        max_turns=max_turns,
    )
    return result.final_output
