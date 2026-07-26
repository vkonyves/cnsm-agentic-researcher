from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionRecord(BaseModel):
    question_id: str
    benchmark: str
    task_id: str

    gold_option: str
    raw_output: str

    parsed_option: str | None
    parse_status: str
    parser_rule: str | None = None

    is_correct: bool | None

    prompt_sha256: str
    generator_name: str


class PredictionRunReport(BaseModel):
    benchmark: str
    sample_path: str

    generator_name: str
    seed: int

    record_count: int
    parsed_count: int
    ambiguous_count: int
    unparsed_count: int
    empty_count: int

    scored_count: int
    correct_count: int
    accuracy_on_scored: float | None
    coverage: float

    parser_rule_counts: dict[str, int] = Field(
        default_factory=dict
    )

    parse_status_counts: dict[str, int] = Field(
        default_factory=dict
    )

    task_accuracy: dict[str, float | None] = Field(
        default_factory=dict
    )

    predictions_path: str
