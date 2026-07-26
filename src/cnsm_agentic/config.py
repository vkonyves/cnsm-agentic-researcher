from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from cnsm_agentic.schemas import ExperimentFamily


class ResourceSpec(BaseModel):
    name: str
    kind: str
    status: str


class PilotConfig(BaseModel):
    run_name: str
    experiment_family: ExperimentFamily
    model_role: str = "worker"
    topic_scope: str
    candidate_resources: list[ResourceSpec] = Field(min_length=1)
    pilot_constraints: dict[str, Any]


def load_config(path: Path) -> PilotConfig:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return PilotConfig.model_validate(payload)
