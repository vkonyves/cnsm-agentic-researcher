from __future__ import annotations

import json
import re
from typing import Any


HUMAN_DEPENDENCY_PATTERNS = {
    "human rater": (
        r"\bhuman raters?\b"
    ),
    "human annotation": (
        r"\bhuman annotat(?:ion|or|ors|e|ed|ing)\b"
    ),
    "human imputation": (
        r"\bhuman imput(?:ation|e|ed|ing)\b"
    ),
    "manual adjudication": (
        r"\bmanual adjudicat(?:ion|or|ors|e|ed|ing)\b"
    ),
    "third rater": (
        r"\bthird rater\b"
    ),
    "external validator": (
        r"\bexternal validators?\b"
    ),
    "external partner": (
        r"\bexternal partners?\b"
    ),
    "industry partner": (
        r"\bindustry partners?\b"
    ),
    "university partner": (
        r"\buniversity partners?\b"
    ),
    "NDA resource": (
        r"\bnda\b"
    ),
    "safety officer": (
        r"\bsafety officer\b"
    ),
    "human review": (
        r"\bhuman review\b"
    ),
    "manual review": (
        r"\bmanual review\b"
    ),
    "human audit": (
        r"\b(?:human|manual human) audit\b"
    ),
    "human scorer": (
        r"\bhuman scor(?:er|ers|ing)\b"
    ),
    "domain expert": (
        r"\bdomain experts?\b"
    ),
    "human evaluator": (
        r"\bhuman evaluat(?:or|ors|ion)\b"
    ),
}


NEGATED_HUMAN_DEPENDENCY_PATTERNS = (
    # Explicit negative-action statements, e.g.
    # "do not perform human adjudication".
    r"\bdo\s+not\s+"
    r"(?:perform|use|require|rely\s+on)\s+"
    r"(?:human\s+annotation|human\s+imput\w*|"
    r"human\s+adjudicat\w*|manual\s+adjudicat\w*|"
    r"human\s+review|manual\s+review)\b",

    # Coordinated explicit denials such as:
    # "no human annotation or manual adjudication is required".
    r"\bno\s+"
    r"(?:human\s+annotation|human\s+imput\w*|human\s+adjudicat\w*|manual\s+adjudicat\w*|human\s+review|manual\s+review)"
    r"(?:\s+or\s+"
    r"(?:human\s+annotation|human\s+imput\w*|human\s+adjudicat\w*|manual\s+adjudicat\w*|human\s+review|manual\s+review))+"
    r"\s+(?:is|are)\s+(?:required|needed|used)\b",

    # "does not require human annotation or manual adjudication".
    r"\bdoes\s+not\s+require\s+"
    r"(?:human\s+annotation|human\s+imput\w*|human\s+adjudicat\w*|manual\s+adjudicat\w*|human\s+review|manual\s+review)"
    r"(?:\s+or\s+"
    r"(?:human\s+annotation|human\s+imput\w*|human\s+adjudicat\w*|manual\s+adjudicat\w*|human\s+review|manual\s+review))+\b",

    # "without human review or manual adjudication".
    r"\bwithout\s+"
    r"(?:human\s+annotation|human\s+imput\w*|human\s+adjudicat\w*|manual\s+adjudicat\w*|human\s+review|manual\s+review)"
    r"(?:\s+or\s+"
    r"(?:human\s+annotation|human\s+imput\w*|human\s+adjudicat\w*|manual\s+adjudicat\w*|human\s+review|manual\s+review))+\b",
    # Coordinated negative requirements, e.g.
    # "no human annotation or manual adjudication is required".
    # Remove the whole denied dependency phrase before scanning for
    # positive human-scientific dependencies.
    r"\\bno[\\s-]+"
    r"(?:human[\\s-]+annotation"
    r"|manual[\\s-]+adjudicat(?:ion|or|ors|e|ed|ing)?"
    r"|human[\\s-]+review"
    r"|manual[\\s-]+review)"
    r"(?:[\\s,;/+-]+or[\\s,;/+-]+"
    r"(?:human[\\s-]+annotation"
    r"|manual[\\s-]+adjudicat(?:ion|or|ors|e|ed|ing)?"
    r"|human[\\s-]+review"
    r"|manual[\\s-]+review))+"
    r"(?:[\\s-]+(?:is|are))?"
    r"(?:[\\s-]+(?:required|needed|used))?\\b",

    # Coordinated "does not require ..." construction.
    r"\\bdoes[\\s-]+not[\\s-]+require[\\s-]+"
    r"(?:human[\\s-]+annotation"
    r"|manual[\\s-]+adjudicat(?:ion|or|ors|e|ed|ing)?"
    r"|human[\\s-]+review"
    r"|manual[\\s-]+review)"
    r"(?:[\\s,;/+-]+or[\\s,;/+-]+"
    r"(?:human[\\s-]+annotation"
    r"|manual[\\s-]+adjudicat(?:ion|or|ors|e|ed|ing)?"
    r"|human[\\s-]+review"
    r"|manual[\\s-]+review))+\\b",

    # Coordinated "without A or B" construction.
    r"\\bwithout[\\s-]+"
    r"(?:human[\\s-]+annotation"
    r"|manual[\\s-]+adjudicat(?:ion|or|ors|e|ed|ing)?"
    r"|human[\\s-]+review"
    r"|manual[\\s-]+review)"
    r"(?:[\\s,;/+-]+or[\\s,;/+-]+"
    r"(?:human[\\s-]+annotation"
    r"|manual[\\s-]+adjudicat(?:ion|or|ors|e|ed|ing)?"
    r"|human[\\s-]+review"
    r"|manual[\\s-]+review))+\\b",
    # Handle the complete ambiguous construction first so that
    # "manual review" is removed together with its explicit denial
    # of human adjudication.
    r"\bflagged[\s-]+for[\s-]+manual[\s-]+review"
    r"[\s\S]{0,80}?"
    r"\bflagging[\s-]+only"
    r"[,;:]?[\s-]+not[\s-]+manual[\s-]+adjudicat"
    r"(?:ion|or|ors|e|ed|ing)?\b",

    r"\bflagging[\s-]+only"
    r"[,;:]?[\s-]+not[\s-]+manual[\s-]+adjudicat"
    r"(?:ion|or|ors|e|ed|ing)?\b",

    r"\bnot[\s-]+manual[\s-]+adjudicat"
    r"(?:ion|or|ors|e|ed|ing)?\b",

    r"\bno[\s-]+manual[\s-]+adjudicat"
    r"(?:ion|or|ors|e|ed|ing)?"
    r"(?:[\s-]+is)?"
    r"(?:[\s-]+required)?\b",

    r"\bwithout[\s-]+manual[\s-]+adjudicat"
    r"(?:ion|or|ors|e|ed|ing)?\b",

    r"\brather[\s-]+than[\s-]+manual[\s-]+adjudicat"
    r"(?:ion|or|ors|e|ed|ing)?\b",

    r"\binstead[\s-]+of[\s-]+manual[\s-]+adjudicat"
    r"(?:ion|or|ors|e|ed|ing)?\b",

    r"\bdoes[\s-]+not[\s-]+require"
    r"[\s-]+manual[\s-]+adjudicat"
    r"(?:ion|or|ors|e|ed|ing)?\b",

    r"\bno[\s-]+additional[\s-]+models"
    r"[\s-]+or[\s-]+human[\s-]+annotators?"
    r"(?:[\s-]+are)?"
    r"[\s-]+required\b",

    r"\bno[\s-]+human[\s-]+annotation"
    r"[\s-]+is[\s-]+used\b",

    r"\bno[\s-]+human[\s-]+annotation"
    r"(?:[\s-]+is)?"
    r"(?:[\s-]+required)?\b",

    r"\bwithout[\s-]+human[\s-]+annotation\b",

    r"\bdoes[\s-]+not[\s-]+require"
    r"[\s-]+human[\s-]+annotation\b",

    r"\bdue[\s-]+to[\s-]+no"
    r"[\s-]+human[\s-]+annotation\b",

    r"\bno[\s-]+human[\s-]+review"
    r"(?:[\s-]+is)?"
    r"(?:[\s-]+required)?\b",

    r"\bwithout[\s-]+human[\s-]+review\b",

    r"\bdoes[\s-]+not[\s-]+require"
    r"[\s-]+human[\s-]+review\b",

    r"\bno[\s-]+manual[\s-]+review"
    r"(?:[\s-]+is)?"
    r"(?:[\s-]+required)?\b",

    r"\bwithout[\s-]+manual[\s-]+review\b",

    r"\bdoes[\s-]+not[\s-]+require"
    r"[\s-]+manual[\s-]+review\b",

    r"\bno[\s-]+human[\s-]+adjudicat"
    r"(?:ion|or|ors|e|ed|ing)?"
    r"(?:[\s-]+is)?"
    r"(?:[\s-]+required)?\b",

    r"\bwithout[\s-]+human[\s-]+adjudicat"
    r"(?:ion|or|ors|e|ed|ing)?\b",

    r"\bdoes[\s-]+not[\s-]+require"
    r"[\s-]+human[\s-]+adjudicat"
    r"(?:ion|or|ors|e|ed|ing)?\b",

    r"\bno[\s-]+human[\s-]+scor"
    r"(?:ing|er|ers|e|ed)?"
    r"(?:[\s-]+is)?"
    r"(?:[\s-]+required)?\b",

    r"\bwithout[\s-]+human[\s-]+scor"
    r"(?:ing|er|ers|e|ed)?\b",

    r"\bdoes[\s-]+not[\s-]+require"
    r"[\s-]+human[\s-]+scor"
    r"(?:ing|er|ers|e|ed)?\b",

    r"\bno[\s-]+manual[\s-]+scor"
    r"(?:ing|er|ers|e|ed)?"
    r"(?:[\s-]+is)?"
    r"(?:[\s-]+required)?\b",

    r"\bwithout[\s-]+manual[\s-]+scor"
    r"(?:ing|er|ers|e|ed)?\b",

    r"\bno[\s-]+human[\s-]+evaluat"
    r"(?:ion|or|ors|e|ed|ing)?"
    r"(?:[\s-]+is)?"
    r"(?:[\s-]+required)?\b",

    r"\bwithout[\s-]+human[\s-]+evaluat"
    r"(?:ion|or|ors|e|ed|ing)?\b",

    r"\bdoes[\s-]+not[\s-]+require"
    r"[\s-]+human[\s-]+evaluat"
    r"(?:ion|or|ors|e|ed|ing)?\b",

    r"\bno[\s-]+manual[\s-]+evaluat"
    r"(?:ion|or|ors|e|ed|ing)?"
    r"(?:[\s-]+is)?"
    r"(?:[\s-]+required)?\b",

    r"\bwithout[\s-]+manual[\s-]+evaluat"
    r"(?:ion|or|ors|e|ed|ing)?\b",

    r"\bno[\s-]+external[\s-]+human[\s-]+labor"
    r"(?:[\s-]+is)?"
    r"(?:[\s-]+required)?\b",

    r"\bno[\s-]+external[\s-]+human[\s-]+labour"
    r"(?:[\s-]+is)?"
    r"(?:[\s-]+required)?\b",

    r"\bwithout[\s-]+external[\s-]+human[\s-]+labor\b",

    r"\bwithout[\s-]+external[\s-]+human[\s-]+labour\b",

    r"\bautomated[\s-]+flagging[\s-]+only\b",

    r"\bautomated[\s-]+review[\s-]+only\b",

    r"\bfully[\s-]+automated[\s-]+review\b",

    r"\baudit[\s-]+is[\s-]+"
    r"(?:fully[\s-]+)?automated\b",

    r"\bautomated[\s-]+audit\b",

    r"\bno[\s-]+human[\s-]+in[\s-]+"
    r"(?:the[\s-]+)?loop\b",

    r"\bno[\s-]+human[\s-]?in[\s-]?loop\b",

    r"\bwithout[\s-]+human[\s-]+in[\s-]+"
    r"(?:the[\s-]+)?loop\b",

    r"\bfully[\s-]+autonomous\b",

    r"\bfully[\s-]+automated\b",

    r"\bno[\s-]+external[\s-]+partners?\b",

    r"\bwithout[\s-]+external[\s-]+partners?\b",

    r"\bdoes[\s-]+not[\s-]+require"
    r"[\s-]+external[\s-]+partners?\b",

    r"\bno[\s-]+external[\s-]+validators?\b",

    r"\bwithout[\s-]+external[\s-]+validators?\b",

    r"\bno[\s-]+industry[\s-]+partners?\b",

    r"\bno[\s-]+university[\s-]+partners?\b",
)


PRIVATE_INFRASTRUCTURE_PATTERNS = {
    "private live lab": (
        r"\blive[\s-]?lab\b"
    ),
    "private laboratory": (
        r"\bprivate lab(?:oratory)?\b"
    ),
    "canary environment": (
        r"\bcanary environment\b"
    ),
    "external controlled environment": (
        r"\bexternal controlled environment\b"
    ),
}


GPU_DEPENDENCY_PATTERNS = {
    "CUDA": (
        r"\bcuda\b"
    ),
    "local GPU": (
        r"\blocal[\s-]+gpu\b"
    ),
    "GPU execution": (
        r"\bgpu[\s-]+"
        r"(?:execution|inference|training|compute|runtime)\b"
    ),
    "GPU required": (
        r"\bgpu[\s-]+"
        r"(?:is[\s-]+)?required\b"
    ),
    "GPU dependency": (
        r"\bgpu[\s-]+dependenc(?:y|ies)\b"
    ),
    "LoRA training": (
        r"\blora[\s-]+"
        r"(?:training|fine[\s-]?tuning|adaptation)\b"
    ),
    "local 7B model": (
        r"\b(?:local[\s-]+7b|7b[\s-]+local)"
        r"(?:[\s-]+model)?\b"
    ),
    "local 70B model": (
        r"\b(?:local[\s-]+70b|70b[\s-]+local)"
        r"(?:[\s-]+model)?\b"
    ),
    "80 GB GPU": (
        r"\b80\s*gb[\s-]+gpu\b"
    ),
    "V100": (
        r"\bv100\b"
    ),
    "A100": (
        r"\ba100\b"
    ),
    "H100": (
        r"\bh100\b"
    ),
}


NEGATED_GPU_PATTERNS = (
    r"\bno[\s-]+local[\s-]+gpu"
    r"(?:[\s-]+is)?(?:[\s-]+required)?\b",

    r"\bno[\s-]+"
    r"(?:[a-z0-9]+[\s-]+){0,4}"
    r"or[\s-]+local[\s-]+gpu"
    r"(?:[\s-]+model)?"
    r"(?:[\s-]+"
    r"(?:execution|inference|training|compute|runtime))?"
    r"(?:[\s-]+is)?"
    r"[\s-]+required\b",

    r"\bno[\s-]+gpu"
    r"(?:[\s-]+is)?(?:[\s-]+required)?\b",

    r"\bwithout[\s-]+"
    r"(?:a[\s-]+)?(?:local[\s-]+)?gpu\b",

    r"\bdoes[\s-]+not[\s-]+require"
    r"[\s-]+(?:a[\s-]+)?(?:local[\s-]+)?gpu\b",

    r"\bdo[\s-]+not[\s-]+require"
    r"[\s-]+(?:a[\s-]+)?(?:local[\s-]+)?gpu\b",

    r"\bnot[\s-]+dependent[\s-]+on"
    r"[\s-]+(?:a[\s-]+)?(?:local[\s-]+)?gpu\b",

    r"\bgpu[\s-]+not[\s-]+required\b",

    r"\bgpu[\s-]+is[\s-]+not[\s-]+required\b",

    r"\bcpu[\s-]?only"
    r"(?:[\s-]+execution)?\b",

    r"\bcpu[\s-]+compatible\b",

    r"\blocal_gpu[\"']?\s*"
    r"[:.=]\s*(?:false|0|null|none)\b",

    r"[\"']available[\"']\s*:\s*false"
    r"(?=[^{}]{0,120}"
    r"[\"']?(?:gpu|local_gpu)[\"']?)",
)


KUBERNETES_PATTERNS = {
    "Kubernetes": (
        r"\bkubernetes\b"
    ),
    "Kubernetes job": (
        r"\bk8s\b"
    ),
}

NEGATED_KUBERNETES_PATTERNS = (
    r"\bno[\s-]+kubernetes\b",

    r"\bwithout[\s-]+kubernetes\b",

    r"\bdoes[\s-]+not[\s-]+require"
    r"[\s-]+kubernetes\b",

    r"\bkubernetes[\s-]+not[\s-]+required\b",

    r"\bkubernetes[\s-]+is[\s-]+not[\s-]+required\b",

    r"\bno[\s-]+k8s\b",

    r"\bwithout[\s-]+k8s\b",

    r"\bdoes[\s-]+not[\s-]+require"
    r"[\s-]+k8s\b",
)

DOCKER_PATTERNS = {
    "Docker": (
        r"\bdocker\b"
    ),
    "Docker Compose": (
        r"\bdocker compose\b"
    ),
}


def _serialise(
    value: Any,
) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        default=str,
    ).lower()


def _find_patterns(
    text: str,
    patterns: dict[str, str],
) -> list[str]:
    return [
        name
        for name, pattern in patterns.items()
        if re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )
    ]


def _remove_patterns(
    text: str,
    patterns: tuple[str, ...],
) -> str:
    cleaned = text

    for pattern in patterns:
        cleaned = re.sub(
            pattern,
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )

    return cleaned


def _prepare_human_dependency_scan_text(
    text: str,
) -> str:
    """
    Remove explicit statements denying human scientific labour.

    The remaining text is inspected for positive dependencies such as
    required human annotation, manual review, or adjudication.
    """
    return _remove_patterns(
        text,
        NEGATED_HUMAN_DEPENDENCY_PATTERNS,
    )



def _human_dependency_occurrence_is_negated(
    text: str,
    *,
    start: int,
    end: int,
) -> bool:
    """
    Return True when this particular human-dependency occurrence is
    explicitly denied or contrasted away in its local clause.

    Classification is occurrence-local: negating one dependency must
    not suppress a positive dependency in another clause.
    """
    clause_breaks = ".;:\n"

    left_boundary = max(
        [
            text.rfind(ch, 0, start)
            for ch in clause_breaks
        ]
        + [-1]
    )

    right_candidates = [
        position
        for ch in clause_breaks
        for position in [text.find(ch, end)]
        if position != -1
    ]

    right_boundary = (
        min(right_candidates)
        if right_candidates
        else len(text)
    )

    clause_start = left_boundary + 1
    clause_end = right_boundary

    clause = text[
        clause_start:clause_end
    ]

    relative_start = start - clause_start
    relative_end = end - clause_start

    before = clause[:relative_start]
    dependency_text = clause[
        relative_start:relative_end
    ]
    after = clause[relative_end:]

    before_window = before[-140:]
    after_window = after[:140]

    # --------------------------------------------------------
    # 1. Direct pre-nominal negation.
    #
    # no human annotation
    # without manual review
    # no manual imputation or human annotation
    # --------------------------------------------------------
    if re.search(
        r"\b(?:no|without)\b"
        r"(?:(?!\b(?:but|however|whereas)\b).){0,120}$",
        before_window,
        flags=re.IGNORECASE,
    ):
        return True

    # Direct local denial:
    #   not manual adjudication
    #   not human annotation
    if re.search(
        r"\bnot\s*$",
        before_window,
        flags=re.IGNORECASE,
    ):
        return True

    # --------------------------------------------------------
    # 1b. Passive-purpose denial before dependency.
    #
    # not used for manual adjudication
    # not intended for human annotation
    # not required for manual review
    #
    # Here the dependency noun follows a denied purpose phrase,
    # so the direct "not <dependency>" rule above does not apply.
    # --------------------------------------------------------
    if re.search(
        r"\bnot\s+"
        r"(?:used|intended|required|needed)\s+for\s*$",
        before_window,
        flags=re.IGNORECASE,
    ):
        return True

    # --------------------------------------------------------
    # 2. Contrastive replacement.
    #
    # deterministic resolution instead of manual adjudication
    # verifier resolution rather than manual adjudication
    # --------------------------------------------------------
    if re.search(
        r"\b(?:instead\s+of|rather\s+than)\s*$",
        before_window,
        flags=re.IGNORECASE,
    ):
        return True

    # --------------------------------------------------------
    # 3. Negative action before dependency.
    #
    # does not require human annotation
    # do not perform manual adjudication
    # will not use human reviewers
    # must not rely on manual review
    # --------------------------------------------------------
    if re.search(
        r"\b(?:do|does|did|will|would|shall|should|"
        r"must|can|could|may|might)\s+not\s+"
        r"(?:require|requires|required|"
        r"use|uses|used|"
        r"perform|performs|performed|"
        r"include|includes|included|"
        r"involve|involves|involved|"
        r"need|needs|needed|"
        r"rely\s+on|depend\s+on)\s*$",
        before_window,
        flags=re.IGNORECASE,
    ):
        return True

    if re.search(
        r"\b(?:is|are|was|were)\s+not\s+"
        r"(?:using|requiring|performing|including|"
        r"involving|relying\s+on|depending\s+on)\s*$",
        before_window,
        flags=re.IGNORECASE,
    ):
        return True

    # --------------------------------------------------------
    # 4. Negative predicate after dependency.
    #
    # human annotation is not required
    # human annotation will not be used
    # manual review must not be performed
    # --------------------------------------------------------

    # Ordinary copula:
    #   annotation is not required
    if re.match(
        r"\s*(?:is|are|was|were)\s+not\s+"
        r"(?:required|needed|used|performed|included|"
        r"involved|permitted|necessary)\b",
        after_window,
        flags=re.IGNORECASE,
    ):
        return True

    # Modal:
    #   annotation will not be used
    #   review must not be performed
    if re.match(
        r"\s*(?:will|would|shall|should|must|can|could|"
        r"may|might)\s+not\s+be\s+"
        r"(?:required|needed|used|performed|included|"
        r"involved|permitted|necessary)\b",
        after_window,
        flags=re.IGNORECASE,
    ):
        return True

    # Explicitly negative adjective:
    #   manual review is unnecessary
    if re.match(
        r"\s*(?:is|are|was|were)\s+"
        r"(?:unnecessary|unneeded|prohibited|forbidden)"
        r"\b",
        after_window,
        flags=re.IGNORECASE,
    ):
        return True

    # --------------------------------------------------------
    # 5. Existing "manual review" ambiguity used only as
    # deterministic flagging terminology.
    #
    # flagged for manual review
    # (flagging only, not manual adjudication)
    #
    # In this explicitly qualified construction, "manual review"
    # is not an executable human-scientific dependency.
    # --------------------------------------------------------
    if (
        dependency_text.lower() == "manual review"
        and re.search(
            r"^\s*"
            r"\(?"
            r"(?:(?![.;]).){0,100}"
            r"\bflagging\s+only\b"
            r"(?:(?![.;]).){0,80}"
            r"\bnot\s+manual\s+adjudicat\w*\b",
            after_window,
            flags=re.IGNORECASE,
        )
    ):
        return True

    # --------------------------------------------------------
    # 6. Human assessment mentioned only as an external
    # comparison/reference target, not as an executable dependency.
    #
    # Examples:
    #   surrogate for human adjudication
    #   proxy for human adjudication
    # --------------------------------------------------------
    if re.search(
        r"\b(?:surrogate|proxy)\s+for\s*$",
        before_window,
        flags=re.IGNORECASE,
    ):
        return True

    return False

def _find_positive_human_dependencies(
    text: str,
    patterns: dict[str, str],
) -> list[str]:
    """
    Find human-scientific dependencies that are actually asserted,
    rather than merely mentioned inside an explicit denial.

    Each occurrence is classified independently so a sentence such as
    "No human annotation is used, but manual review is required"
    still correctly reports manual review.
    """
    found: set[str] = set()

    for name, pattern in patterns.items():
        for match in re.finditer(
            pattern,
            text,
            flags=re.IGNORECASE,
        ):
            if not _human_dependency_occurrence_is_negated(
                text,
                start=match.start(),
                end=match.end(),
            ):
                found.add(name)

    return sorted(found)


def _prepare_gpu_scan_text(
    text: str,
) -> str:
    """
    Remove explicit statements that deny a GPU requirement.

    The remaining text is then inspected for positive GPU dependencies.
    This avoids treating phrases such as "no local GPU required" as
    evidence that the design requires a GPU.
    """
    return _remove_patterns(
        text,
        NEGATED_GPU_PATTERNS,
    )

def _prepare_kubernetes_scan_text(
    text: str,
) -> str:
    """
    Remove explicit statements denying a Kubernetes requirement.

    The remaining text is inspected for positive Kubernetes
    orchestration dependencies.
    """
    return _remove_patterns(
        text,
        NEGATED_KUBERNETES_PATTERNS,
    )

def _extract_estimated_calls(
    design: dict[str, Any],
) -> int | None:
    value = design.get(
        "estimated_model_calls"
    )

    if isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        return int(value)

    return None


def validate_design_feasibility(
    *,
    design: dict[str, Any],
    capability_manifest: dict[str, Any],
) -> list[str]:
    text = _serialise(
        design
    )

    human_scan_text = (
        _prepare_human_dependency_scan_text(
            text
        )
    )

    issues: list[str] = []

    if not capability_manifest.get(
        "human_scientific_labour_allowed",
        False,
    ):
        for dependency in _find_positive_human_dependencies(
            text,
            HUMAN_DEPENDENCY_PATTERNS,
        ):
            issues.append(
                "Forbidden human scientific "
                f"dependency: {dependency}"
            )

    if not capability_manifest.get(
        "external_partner_allowed",
        False,
    ):
        external_patterns = {
            key: value
            for key, value
            in HUMAN_DEPENDENCY_PATTERNS.items()
            if (
                "external" in key.lower()
                or "partner" in key.lower()
                or "nda" in key.lower()
            )
        }

        for dependency in _find_positive_human_dependencies(
            text,
            external_patterns,
        ):
            issues.append(
                "Forbidden external dependency: "
                f"{dependency}"
            )

    if not capability_manifest.get(
        "human_annotation_allowed",
        False,
    ):
        annotation_patterns = {
            key: value
            for key, value
            in HUMAN_DEPENDENCY_PATTERNS.items()
            if (
                "rater" in key.lower()
                or "annotation" in key.lower()
                or "adjudication" in key.lower()
            )
        }

        for dependency in _find_positive_human_dependencies(
            text,
            annotation_patterns,
        ):
            issues.append(
                "Human annotation is unavailable: "
                f"{dependency}"
            )

    if not capability_manifest.get(
        "nda_resources_allowed",
        False,
    ):
        if re.search(
            r"\bnda\b",
            text,
            flags=re.IGNORECASE,
        ):
            issues.append(
                "NDA-protected resources are unavailable."
            )

    if not capability_manifest.get(
        "private_live_lab_available",
        False,
    ):
        for dependency in _find_patterns(
            text,
            PRIVATE_INFRASTRUCTURE_PATTERNS,
        ):
            issues.append(
                "Unavailable private infrastructure: "
                f"{dependency}"
            )

    local_gpu = capability_manifest.get(
        "local_gpu",
        {},
    )

    gpu_available = bool(
        local_gpu.get(
            "available",
            False,
        )
    )

    if not gpu_available:
        gpu_scan_text = _prepare_gpu_scan_text(
            text
        )

        for dependency in _find_patterns(
            gpu_scan_text,
            GPU_DEPENDENCY_PATTERNS,
        ):
            issues.append(
                "Unavailable local GPU dependency: "
                f"{dependency}"
            )

    if not capability_manifest.get(
        "kubernetes_available",
        False,
    ):
        kubernetes_scan_text = (
            _prepare_kubernetes_scan_text(
                text
            )
        )

        for dependency in _find_patterns(
            kubernetes_scan_text,
            KUBERNETES_PATTERNS,
        ):
            issues.append(
                "Unavailable orchestration dependency: "
                f"{dependency}"
            )

    if not capability_manifest.get(
        "docker_available",
        False,
    ):
        for dependency in _find_patterns(
            text,
            DOCKER_PATTERNS,
        ):
            issues.append(
                "Unavailable container dependency: "
                f"{dependency}"
            )

    if capability_manifest.get(
        "public_datasets_only",
        False,
    ):
        private_data_patterns = (
            r"\bprivate dataset\b",
            r"\bproprietary dataset\b",
            r"\bcontrolled[- ]access dataset\b",
            r"\bconfidential data\b",
            r"\bpartner data\b",
        )

        if any(
            re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )
            for pattern in private_data_patterns
        ):
            issues.append(
                "Design requires data that are not "
                "clearly public."
            )

    if capability_manifest.get(
        "autonomous_scoring_required",
        False,
    ):
        # Autonomous-scoring feasibility must distinguish an asserted
        # human-scoring dependency from an explicit denial of one.
        #
        # Examples that must PASS:
        #   "human adjudication is prohibited"
        #   "no human scoring is used"
        #   "without manual evaluation"
        #
        # Examples that must FAIL:
        #   "human adjudication resolves ambiguous cases"
        #   "manual scoring is required"
        #
        # Reuse the same clause-aware occurrence classifier used for
        # the other human-scientific dependency gates rather than
        # treating every lexical mention as a positive dependency.
        scoring_patterns = {
            "human scoring": (
                r"\bhuman scor(?:e|ed|er|ers|ing)\b"
            ),
            "manual scoring": (
                r"\bmanual scor(?:e|ed|er|ers|ing)\b"
            ),
            "human evaluation": (
                r"\bhuman evaluat"
                r"(?:e|ed|es|ing|ion|or|ors)\b"
            ),
            "manual evaluation": (
                r"\bmanual evaluat"
                r"(?:e|ed|es|ing|ion|or|ors)\b"
            ),
            "human adjudication": (
                r"\bhuman adjudicat"
                r"(?:ion|or|ors|e|ed|ing)\b"
            ),
        }

        positive_scoring_dependencies = (
            _find_positive_human_dependencies(
                text,
                scoring_patterns,
            )
        )

        if positive_scoring_dependencies:
            issues.append(
                "Design violates autonomous-scoring requirement: "
                + ", ".join(positive_scoring_dependencies)
            )

    maximum_calls = capability_manifest.get(
        "maximum_planned_model_calls"
    )

    estimated_calls = _extract_estimated_calls(
        design
    )

    if (
        isinstance(maximum_calls, int)
        and estimated_calls is not None
        and estimated_calls > maximum_calls
    ):
        issues.append(
            "Estimated model calls exceed "
            "the capability limit: "
            f"{estimated_calls} > {maximum_calls}"
        )

    return sorted(
        set(
            issues
        )
    )


def feasibility_report(
    *,
    design: dict[str, Any],
    capability_manifest: dict[str, Any],
) -> dict[str, Any]:
    issues = validate_design_feasibility(
        design=design,
        capability_manifest=(
            capability_manifest
        ),
    )

    return {
        "status": (
            "passed"
            if not issues
            else "failed"
        ),
        "issue_count": len(
            issues
        ),
        "issues": issues,
    }
