from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class MCQARecord(BaseModel):
    benchmark: str

    question_id: str
    task_id: str
    task_name: str
    source_turn: int

    question: str
    options: dict[str, str]

    correct_option: str
    rationale: str
    rationale_tag: str
    difficulty: str

    source_file: str
    source_question_index: int

    @field_validator("question_id")
    @classmethod
    def validate_question_id(
        cls,
        value: str,
    ) -> str:
        if not value.strip():
            raise ValueError(
                "question_id must not be empty."
            )

        return value

    @field_validator("question")
    @classmethod
    def validate_question(
        cls,
        value: str,
    ) -> str:
        if not value.strip():
            raise ValueError(
                "question must not be empty."
            )

        return value.strip()

    @field_validator("options")
    @classmethod
    def validate_options(
        cls,
        value: dict[str, str],
    ) -> dict[str, str]:
        if len(value) < 2:
            raise ValueError(
                "Each MCQA record must contain "
                "at least two options."
            )

        normalised: dict[str, str] = {}

        for key, option_text in value.items():
            option_key = str(key).strip().upper()

            if not option_key:
                raise ValueError(
                    "Option labels must not be empty."
                )

            if not isinstance(option_text, str):
                raise ValueError(
                    f"Option {option_key!r} must be text."
                )

            if not option_text.strip():
                raise ValueError(
                    f"Option {option_key!r} must not be empty."
                )

            normalised[option_key] = (
                option_text.strip()
            )

        return normalised

    @field_validator("correct_option")
    @classmethod
    def validate_correct_option(
        cls,
        value: str,
    ) -> str:
        option = value.strip().upper()

        if not option:
            raise ValueError(
                "correct_option must not be empty."
            )

        return option


class NormalisationReport(BaseModel):
    benchmark: str

    source_root: str
    output_jsonl: str

    source_file_count: int
    question_file_count: int

    raw_question_count: int
    normalised_record_count: int

    duplicate_question_id_count: int
    invalid_record_count: int

    option_label_sets: dict[str, int] = Field(
        default_factory=dict
    )

    correct_option_counts: dict[str, int] = Field(
        default_factory=dict
    )

    task_counts: dict[str, int] = Field(
        default_factory=dict
    )

    difficulty_counts: dict[str, int] = Field(
        default_factory=dict
    )

    rationale_tag_counts: dict[str, int] = Field(
        default_factory=dict
    )

    invalid_records: list[dict[str, object]] = Field(
        default_factory=list
    )
