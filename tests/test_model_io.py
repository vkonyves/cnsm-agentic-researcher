from cnsm_agentic.benchmark_schemas import MCQARecord
from cnsm_agentic.model_io import format_mcqa_prompt

from cnsm_agentic.model_io import (
    parse_mcqa_output,
)


VALID = {"A", "B", "C", "D"}


def test_exact_label() -> None:
    result = parse_mcqa_output(
        "C",
        VALID,
    )

    assert result.parsed_option == "C"
    assert result.parse_status == "parsed"
    assert result.parser_rule == "exact_label"


def test_label_with_punctuation() -> None:
    result = parse_mcqa_output(
        "B.",
        VALID,
    )

    assert result.parsed_option == "B"
    assert result.parse_status == "parsed"


def test_answer_phrase() -> None:
    result = parse_mcqa_output(
        "The answer is D.",
        VALID,
    )

    assert result.parsed_option == "D"
    assert result.parse_status == "parsed"


def test_option_phrase() -> None:
    result = parse_mcqa_output(
        "I select option A.",
        VALID,
    )

    assert result.parsed_option == "A"
    assert result.parse_status == "parsed"


def test_single_standalone_label() -> None:
    result = parse_mcqa_output(
        "After considering the alternatives, C seems safest.",
        VALID,
    )

    assert result.parsed_option == "C"
    assert result.parse_status == "parsed"


def test_ambiguous_output() -> None:
    result = parse_mcqa_output(
        "It could be A or C.",
        VALID,
    )

    assert result.parsed_option is None
    assert result.parse_status == "ambiguous"


def test_invalid_label() -> None:
    result = parse_mcqa_output(
        "E",
        VALID,
    )

    assert result.parsed_option is None
    assert result.parse_status == "unparsed"


def test_empty_output() -> None:
    result = parse_mcqa_output(
        "   ",
        VALID,
    )

    assert result.parsed_option is None
    assert result.parse_status == "empty"


def test_conflicting_explicit_answers() -> None:
    result = parse_mcqa_output(
        "The answer is A, but option B is also possible.",
        VALID,
    )

    assert result.parsed_option is None
    assert result.parse_status == "ambiguous"
    


def test_prompt_formatter() -> None:
    record = MCQARecord(
        benchmark="6G-Bench",
        question_id="test-1",
        task_id="T1",
        task_name="Test Task",
        source_turn=1,
        question="Which option is correct?",
        options={
            "A": "First",
            "B": "Second",
            "C": "Third",
            "D": "Fourth",
        },
        correct_option="B",
        rationale="Because B is correct.",
        rationale_tag="TEST",
        difficulty="very_hard",
        source_file="example.json",
        source_question_index=0,
    )

    prompt = format_mcqa_prompt(record)

    assert "Which option is correct?" in prompt
    assert "A. First" in prompt
    assert "B. Second" in prompt
    assert "Return exactly one option label" in prompt

    # The gold answer must never leak into the prompt.
    assert "Because B is correct." not in prompt    
