from __future__ import annotations

from pydantic import BaseModel, Field


class BaselineResult(BaseModel):
    baseline_name: str
    benchmark: str

    record_count: int
    correct_count: int
    accuracy: float

    prediction_counts: dict[str, int] = Field(
        default_factory=dict
    )

    correct_option_counts: dict[str, int] = Field(
        default_factory=dict
    )

    per_task_accuracy: dict[str, float] = Field(
        default_factory=dict
    )


class BaselineReport(BaseModel):
    benchmark: str
    dataset_path: str
    results: list[BaselineResult]
