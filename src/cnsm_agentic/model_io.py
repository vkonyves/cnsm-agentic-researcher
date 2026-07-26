from __future__ import annotations

import re

from pydantic import BaseModel

from cnsm_agentic.benchmark_schemas import MCQARecord


class ParsedMCQAOutput(BaseModel):
    raw_output: str
    parsed_option: str | None
    parse_status: str
    parser_rule: str | None = None


def format_mcqa_prompt(
    record: MCQARecord,
) -> str:
    option_lines = "\n".join(
        f"{label}. {text}"
        for label, text in sorted(
            record.options.items()
        )
    )

    valid_labels = ", ".join(
        sorted(record.options.keys())
    )

    return f"""
You are answering a multiple-choice question about network and service management.

Question:
{record.question}

Options:
{option_lines}

Return exactly one option label from:
{valid_labels}

Do not provide an explanation.
""".strip()


def parse_mcqa_output(
    raw_output: str,
    valid_options: set[str],
) -> ParsedMCQAOutput:
    text = raw_output.strip()

    if not text:
        return ParsedMCQAOutput(
            raw_output=raw_output,
            parsed_option=None,
            parse_status="empty",
        )

    valid = {
        option.upper()
        for option in valid_options
    }

    upper = text.upper().strip()

    # Rule 1: exact label.
    if upper in valid:
        return ParsedMCQAOutput(
            raw_output=raw_output,
            parsed_option=upper,
            parse_status="parsed",
            parser_rule="exact_label",
        )

    # Rule 2: exact label with punctuation.
    punctuation_match = re.fullmatch(
        r"\s*([A-Z])[\.\)\:\-]?\s*",
        upper,
    )

    if punctuation_match:
        candidate = punctuation_match.group(1)

        if candidate in valid:
            return ParsedMCQAOutput(
                raw_output=raw_output,
                parsed_option=candidate,
                parse_status="parsed",
                parser_rule="label_with_punctuation",
            )

    patterns = [
        (
            "answer_phrase",
            r"\b(?:THE\s+)?ANSWER\s+IS\s+([A-Z])\b",
        ),
        (
            "option_phrase",
            r"\bOPTION\s+([A-Z])\b",
        ),
        (
            "choice_phrase",
            r"\bCHOICE\s+([A-Z])\b",
        ),
        (
            "select_phrase",
            r"\b(?:SELECT|SELECTED|CHOOSE|CHOSEN)\s+([A-Z])\b",
        ),
    ]

    matches: list[
        tuple[str, str]
    ] = []

    for rule_name, pattern in patterns:
        for match in re.finditer(
            pattern,
            upper,
        ):
            candidate = match.group(1)

            if candidate in valid:
                matches.append(
                    (
                        rule_name,
                        candidate,
                    )
                )

    unique_candidates = {
        candidate
        for _, candidate in matches
    }

    if len(unique_candidates) == 1:
        selected = next(
            iter(unique_candidates)
        )

        matching_rules = sorted(
            {
                rule
                for rule, candidate in matches
                if candidate == selected
            }
        )

        return ParsedMCQAOutput(
            raw_output=raw_output,
            parsed_option=selected,
            parse_status="parsed",
            parser_rule="+".join(
                matching_rules
            ),
        )

    if len(unique_candidates) > 1:
        return ParsedMCQAOutput(
            raw_output=raw_output,
            parsed_option=None,
            parse_status="ambiguous",
            parser_rule="multiple_explicit_options",
        )

    # Rule 4: one standalone option letter anywhere.
    standalone_candidates = {
        match.group(1)
        for match in re.finditer(
            r"(?<![A-Z0-9])([A-Z])(?![A-Z0-9])",
            upper,
        )
        if match.group(1) in valid
    }

    if len(standalone_candidates) == 1:
        selected = next(
            iter(standalone_candidates)
        )

        return ParsedMCQAOutput(
            raw_output=raw_output,
            parsed_option=selected,
            parse_status="parsed",
            parser_rule="single_standalone_label",
        )

    if len(standalone_candidates) > 1:
        return ParsedMCQAOutput(
            raw_output=raw_output,
            parsed_option=None,
            parse_status="ambiguous",
            parser_rule="multiple_standalone_labels",
        )

    return ParsedMCQAOutput(
        raw_output=raw_output,
        parsed_option=None,
        parse_status="unparsed",
    )
