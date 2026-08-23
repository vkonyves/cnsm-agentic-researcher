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


def test_explicit_no_gpu_requirement_is_accepted() -> None:
    design = {
        "implementation_strategy": (
            "Hosted API execution and CPU-only procedural synthesis. "
            "No local GPU required."
        ),
        "estimated_model_calls": 2_400,
    }

    issues = validate_design_feasibility(
        design=design,
        capability_manifest=CAPABILITIES,
    )

    assert not any(
        "gpu" in issue.lower()
        for issue in issues
    )


def test_coordinated_no_local_gpu_requirement_is_accepted() -> None:
    design = {
        "model_scope": [
            (
                "Hosted instruction-tuned LLM accessed via "
                "hosted-model API."
            ),
            (
                "No private or local-GPU model execution "
                "required; all LLM work uses hosted-model API."
            ),
        ],
        "estimated_model_calls": 5_400,
    }

    issues = validate_design_feasibility(
        design=design,
        capability_manifest=CAPABILITIES,
    )

    assert not any(
        "gpu" in issue.lower()
        for issue in issues
    )


def test_gpu_requirement_is_still_rejected() -> None:
    design = {
        "implementation_strategy": (
            "Run model inference on a local GPU using CUDA."
        ),
        "estimated_model_calls": 100,
    }

    issues = validate_design_feasibility(
        design=design,
        capability_manifest=CAPABILITIES,
    )

    assert any(
        "gpu" in issue.lower()
        for issue in issues
    )


def test_hosted_7b_model_is_not_assumed_local() -> None:
    design = {
        "adapter_family": (
            "Hosted 7B model API"
        ),
        "implementation_strategy": (
            "All inference is performed through a hosted endpoint."
        ),
        "estimated_model_calls": 500,
    }

    issues = validate_design_feasibility(
        design=design,
        capability_manifest=CAPABILITIES,
    )

    assert not any(
        "gpu" in issue.lower()
        for issue in issues
    )


def test_local_7b_model_is_rejected() -> None:
    design = {
        "adapter_family": (
            "Local 7B model"
        ),
        "implementation_strategy": (
            "Execute the model locally."
        ),
        "estimated_model_calls": 500,
    }

    issues = validate_design_feasibility(
        design=design,
        capability_manifest=CAPABILITIES,
    )

    assert any(
        "gpu" in issue.lower()
        for issue in issues
    )


def test_no_human_annotation_required_is_accepted() -> None:
    design = {
        "audit_policy": (
            "All audits are deterministic and automated. "
            "No human annotation required."
        ),
        "estimated_model_calls": 100,
    }

    issues = validate_design_feasibility(
        design=design,
        capability_manifest=CAPABILITIES,
    )

    assert not any(
        "human" in issue.lower()
        or "annotation" in issue.lower()
        for issue in issues
    )


def test_automated_flagging_without_human_labour_is_accepted() -> None:
    design = {
        "contamination_detection": (
            "Items are flagged for automated quarantine. "
            "Automated flagging only; no external human labour required."
        ),
        "estimated_model_calls": 100,
    }

    issues = validate_design_feasibility(
        design=design,
        capability_manifest=CAPABILITIES,
    )

    assert not any(
        "human" in issue.lower()
        or "manual review" in issue.lower()
        for issue in issues
    )


def test_required_manual_review_is_rejected() -> None:
    design = {
        "audit_policy": (
            "Every flagged item must undergo manual review."
        ),
        "estimated_model_calls": 100,
    }

    issues = validate_design_feasibility(
        design=design,
        capability_manifest=CAPABILITIES,
    )

    assert any(
        "manual review" in issue.lower()
        or "human" in issue.lower()
        for issue in issues
    )


def test_required_human_annotation_is_rejected() -> None:
    design = {
        "scoring": (
            "Two human annotators label every generated artifact."
        ),
        "estimated_model_calls": 100,
    }

    issues = validate_design_feasibility(
        design=design,
        capability_manifest=CAPABILITIES,
    )

    assert any(
        "human annotation" in issue.lower()
        or "human scientific" in issue.lower()
        for issue in issues
    )


def test_no_human_adjudication_is_accepted() -> None:
    design = {
        "missingness_plan": (
            "All missingness decisions are deterministic and logged. "
            "No human adjudication will be used."
        ),
        "estimated_model_calls": 100,
    }

    issues = validate_design_feasibility(
        design=design,
        capability_manifest=CAPABILITIES,
    )

    assert (
        "Design violates autonomous-scoring requirement."
        not in issues
    )

    assert not any(
        "human" in issue.lower()
        or "adjudication" in issue.lower()
        for issue in issues
    )


def test_required_human_adjudication_is_rejected() -> None:
    design = {
        "scoring_plan": (
            "Disagreements are resolved by human adjudication."
        ),
        "estimated_model_calls": 100,
    }

    issues = validate_design_feasibility(
        design=design,
        capability_manifest=CAPABILITIES,
    )

    assert any(
        "adjudication" in issue.lower()
        or "autonomous-scoring" in issue.lower()
        or "human scientific" in issue.lower()
        for issue in issues
    )


def test_no_human_evaluation_is_accepted() -> None:
    design = {
        "evaluation_plan": (
            "Metrics are computed automatically. "
            "No human evaluation is required."
        ),
        "estimated_model_calls": 100,
    }

    issues = validate_design_feasibility(
        design=design,
        capability_manifest=CAPABILITIES,
    )

    assert (
        "Design violates autonomous-scoring requirement."
        not in issues
    )

def test_flagging_only_not_manual_adjudication_is_not_human_dependency():
    design = {
        "contamination_plan": {
            "detection_procedures": [
                (
                    "Items with near-perfect accuracy are flagged "
                    "for manual review (flagging only, not manual "
                    "adjudication)."
                )
            ]
        }
    }

    report = feasibility_report(
        design=design,
        capability_manifest={
            "human_annotation_available": False,
        },
    )

    issues = report.get(
        "issues",
        [],
    )

    assert not any(
        "manual review" in issue.lower()
        for issue in issues
    )

    assert not any(
        "manual adjudication" in issue.lower()
        for issue in issues
    )


def test_automated_audit_with_no_human_annotation_is_allowed():
    design = {
        "transformation_validation": {
            "audit_sample_policy": (
                "The audit is automated with no human annotation."
            )
        }
    }

    report = feasibility_report(
        design=design,
        capability_manifest={
            "human_annotation_available": False,
        },
    )

    issues = report.get(
        "issues",
        [],
    )

    assert not any(
        "human annotation"
        in issue.lower()
        for issue in issues
    )


def test_true_manual_review_dependency_remains_forbidden():
    design = {
        "validation_plan": (
            "Human experts perform manual review of every flagged "
            "item before inclusion."
        )
    }

    report = feasibility_report(
        design=design,
        capability_manifest={
            "human_annotation_available": False,
        },
    )

    issues = report.get(
        "issues",
        [],
    )

    assert any(
        "manual review"
        in issue.lower()
        for issue in issues
    )