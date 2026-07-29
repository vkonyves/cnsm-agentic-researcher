from __future__ import annotations

from agents import Agent

from .schemas import CandidateSet, EvidenceSynthesis, QueryPlan, ReviewSet, SelectionDecision


QUERY_PLANNER = Agent(
    name="Literature Query Planner",
    model="gpt-5-mini",
    output_type=QueryPlan,
    instructions="Create a broad scholarly search plan from the programme. Do not assume a study candidate. Cover phenomenon, mechanisms, mitigations, benchmark validity, methods, and neighbouring work.",
)

EVIDENCE_SYNTHESISER = Agent(
    name="Evidence Synthesiser",
    model="gpt-5-mini",
    output_type=EvidenceSynthesis,
    instructions="Synthesize only claims supported by supplied record IDs. Distinguish absence of evidence from evidence of absence. Reject unsupported novelty claims.",
)

CANDIDATE_GENERATOR = Agent(
    name="Research Candidate Generator",
    model="gpt-5-mini",
    output_type=CandidateSet,
    instructions="Generate 3 to 6 distinct, feasible, falsifiable candidates grounded in evidence IDs. Do not use predetermined candidate IDs or preferred methods.",
)

CANDIDATE_CRITIC = Agent(
    name="Research Candidate Critic",
    model="gpt-5-mini",
    output_type=ReviewSet,
    instructions="Critically review every candidate for novelty, falsifiability, evidence support, causal interpretability, reproducibility, compute feasibility, and venue relevance. Use pass, repair, or veto.",
)

SELECTION_JUDGE = Agent(
    name="Evidence-Grounded Selection Judge",
    model="gpt-5-mini",
    output_type=SelectionDecision,
    instructions="Select one candidate only from the programme, evidence, candidates, and reviews. Do not prefer any candidate ID. Cite record IDs and list repairs required before preregistration.",
)
