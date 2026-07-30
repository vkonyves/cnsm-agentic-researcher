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
    "manual adjudication": (
        r"\bmanual adjudicat"
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
}

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
    "CUDA": r"\bcuda\b",
    "GPU": r"\bgpu\b",
    "LoRA": r"\blora\b",
    "7B local model": (
        r"\b7b\b"
    ),
    "70B model": (
        r"\b70b\b"
    ),
    "80 GB GPU": (
        r"\b80\s*gb\b"
    ),
    "V100": r"\bv100\b",
    "A100": r"\ba100\b",
    "H100": r"\bh100\b",
}

KUBERNETES_PATTERNS = {
    "Kubernetes": (
        r"\bkubernetes\b"
    ),
    "Kubernetes job": (
        r"\bk8s\b"
    ),
}

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

    issues: list[str] = []

    if not capability_manifest.get(
        "human_scientific_labour_allowed",
        False,
    ):
        for dependency in _find_patterns(
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

        for dependency in _find_patterns(
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

        for dependency in _find_patterns(
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
        for dependency in _find_patterns(
            text,
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
        for dependency in _find_patterns(
            text,
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
        scoring_patterns = (
            r"\bhuman scor",
            r"\bmanual scor",
            r"\bhuman evaluat",
            r"\bmanual evaluat",
            r"\bhuman adjudicat",
        )

        if any(
            re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )
            for pattern in scoring_patterns
        ):
            issues.append(
                "Design violates autonomous-scoring requirement."
            )

    maximum_calls = capability_manifest.get(
        "maximum_planned_model_calls"
    )

    estimated_calls = (
        _extract_estimated_calls(
            design
        )
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
        set(issues)
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
