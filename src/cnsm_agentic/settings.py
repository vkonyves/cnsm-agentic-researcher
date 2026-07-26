from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    api_key_present: bool
    worker_model: str
    research_model: str
    lead_model: str
    review_model: str
    artifact_root: Path
    max_agent_turns: int
    run_budget_gbp: float


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        api_key_present=bool(os.getenv("OPENAI_API_KEY")),
        worker_model=os.getenv("OPENAI_WORKER_MODEL", "gpt-5-nano"),
        research_model=os.getenv("OPENAI_RESEARCH_MODEL", "gpt-5-mini"),
        lead_model=os.getenv("OPENAI_LEAD_MODEL", "gpt-5"),
        review_model=os.getenv("OPENAI_REVIEW_MODEL", "gpt-5"),
        artifact_root=Path(os.getenv("ARTIFACT_ROOT", "./runs")),
        max_agent_turns=int(os.getenv("MAX_AGENT_TURNS", "12")),
        run_budget_gbp=float(os.getenv("RUN_BUDGET_GBP", "5")),
    )
