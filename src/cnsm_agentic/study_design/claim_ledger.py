from __future__ import annotations

from dataclasses import asdict

from .models import Claim, StudyPlan


def initialise_claim_ledger(plan: StudyPlan) -> list[dict[str, object]]:
    claims = []
    for index, hypothesis in enumerate(plan.hypotheses, start=1):
        claims.append(Claim(
            claim_id=f"C-{index:03d}",
            claim_text=hypothesis.statement,
            claim_type=f"{hypothesis.kind}_hypothesis",
            supporting_artifacts=[],
            supporting_sources=[],
            statistical_status="proposed",
            scope={"study_id": plan.study_id, "selected_candidate_id": plan.selected_candidate_id},
            prohibited_generalizations=[
                "all LLMs unless multiple representative models are tested",
                "all NetOps tasks unless benchmark scope supports the claim",
                "causal mechanism without a mechanism-identifying design",
            ],
        ))
    return [asdict(claim) for claim in claims]
