from __future__ import annotations

from cnsm_agentic.autonomous_research.feasibility import (
    feasibility_report,
    validate_design_feasibility,
)

CAPABILITIES = {
    "human_scientific_labour_allowed": False,
    "external_partner_allowed": False,
    "human_annotation_allowed": False,
    "manual_adjudication_allowed": False,
    "nda_resources_allowed": False,
    "private_live_lab_available": False,
    "public_datasets_only": True,
    "public_literature_only": True,
    "autonomous_scoring_required": True,
    "docker_available": True,
    "kubernetes_available": False,
    "hosted_model_api_available": True,
    "public_internet_retrieval_available": True,
    "local_gpu": {
        "available": False,
        "memory_gb": 0,
    },
    "cpu_execution_available": True,
    "maximum_planned_model_calls": 10_000,
    "maximum_wall_clock_days": 7,
}


def test_rejects_gpu_and_human_dependencies() -> None:
    design = {
        "adapter_family": (
            "LLaMA 70B LoRA local CUDA"
        ),
        "validation": (
            "Two human raters, one third rater, "
            "and an external validator under NDA."
        ),
        "environment": (
            "Kubernetes live-lab canary environment"
        ),
        "estimated_model_calls": 5_040,
    }

    issues = validate_design_feasibility(
        design=design,
        capability_manifest=CAPABILITIES,
    )

    combined = "\n".join(
        issues
    ).lower()

    assert "gpu" in combined
    assert "human" in combined
    assert "external" in combined
    assert "nda" in combined
    assert "kubernetes" in combined
    assert "live lab" in combined


def test_rejects_excessive_model_calls() -> None:
    issues = validate_design_feasibility(
        design={
            "estimated_model_calls": 10_001,
        },
        capability_manifest=CAPABILITIES,
    )

    assert any(
        "10001 > 10000" in issue
        for issue in issues
    )


def test_accepts_bounded_api_study() -> None:
    report = feasibility_report(
        design={
            "adapter_family": (
                "Hosted API benchmark evaluator"
            ),
            "implementation_strategy": (
                "Public benchmark, deterministic "
                "transformations, automatic scoring"
            ),
            "estimated_model_calls": 2_000,
        },
        capability_manifest=CAPABILITIES,
    )

    assert report["status"] == "passed"
    assert report["issues"] == []
    
def test_repaired_hosted_api_plan_passes_after_gpu_plan_fails() -> None:
    rejected_plan = {
        "adapter_family": "Local 7B model running on GPU",
        "implementation_strategy": "CUDA inference",
        "estimated_model_calls": 3_000,
    }

    rejected_issues = validate_design_feasibility(
        design=rejected_plan,
        capability_manifest=CAPABILITIES,
    )

    assert any(
        "gpu" in issue.lower()
        for issue in rejected_issues
    )

    repaired_plan = {
        "adapter_family": "Hosted model API benchmark",
        "implementation_strategy": (
            "Hosted API execution with deterministic "
            "automatic scoring"
        ),
        "estimated_model_calls": 3_000,
    }

    repaired_issues = validate_design_feasibility(
        design=repaired_plan,
        capability_manifest=CAPABILITIES,
    )

    assert repaired_issues == []    
