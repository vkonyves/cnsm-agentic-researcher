from __future__ import annotations

from .models import CriticReview, ResearchCandidate, ResearchProgramme


def deterministic_critic(candidate: ResearchCandidate, programme: ResearchProgramme) -> CriticReview:
    missing = [c for c in programme.required_controls if c.casefold() not in candidate.design_summary.casefold()]
    leakage = 0.35 if "fine-tun" in candidate.design_summary.casefold() else 0.10
    overclaim = 0.30 if "single-model" in candidate.tags else 0.15
    weak = [] if "unchanged repeat" in candidate.design_summary.casefold() else ["No unchanged-repeat control"]
    stats = [] if "cluster" in candidate.design_summary.casefold() else ["No cluster-aware uncertainty"]
    feasibility = []
    if candidate.estimated_model_calls > 10000:
        feasibility.append("Model-call budget too high")
    veto = leakage > programme.critic_veto_thresholds["data_leakage_risk"] or overclaim > programme.critic_veto_thresholds["scope_overclaim_risk"]
    verdict = "veto" if veto else ("revise" if missing else "pass")
    return CriticReview(candidate.candidate_id, [], leakage, overclaim, weak, missing, stats, feasibility, verdict, "Acceptable for tournament scoring." if verdict == "pass" else "Revision or rejection required.")
