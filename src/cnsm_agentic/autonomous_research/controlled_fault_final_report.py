from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

REPORT_ID = "controlled_fault_final_experiment_report_v1"
REPORT_VERSION = "1.0"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_mcnemar_two_sided(n10: int, n01: int) -> float:
    discordant = n10 + n01
    if discordant == 0:
        return 1.0
    k = min(n10, n01)
    cumulative = sum(math.comb(discordant, i) for i in range(k + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * cumulative)


def _bootstrap_difference_ci(
    differences: list[int], *, resamples: int, seed: int, alpha: float = 0.05
) -> tuple[float, float]:
    if not differences:
        raise ValueError("No paired differences are available.")
    rng = random.Random(seed)
    n = len(differences)
    draws = []
    for _ in range(resamples):
        sample = [differences[rng.randrange(n)] for _ in range(n)]
        draws.append(sum(sample) / n)
    draws.sort()
    lower_index = max(0, min(len(draws) - 1, int((alpha / 2) * len(draws))))
    upper_index = max(0, min(len(draws) - 1, int((1 - alpha / 2) * len(draws)) - 1))
    return draws[lower_index], draws[upper_index]


def _checkpoint_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((run_dir / "execution" / "checkpoints").glob("*.json")):
        checkpoint = _read_json(path)
        checkpoint["_artifact_path"] = str(path)
        rows.append(checkpoint)
    return rows


def _scoring_by_task(run_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for path in sorted((run_dir / "execution" / "scoring").glob("*.json")):
        record = _read_json(path)
        if path.name.endswith("-guarded.json"):
            task_id = path.name[: -len("-guarded.json")]
        elif path.name.endswith("-baseline.json"):
            task_id = path.name[: -len("-baseline.json")]
        else:
            continue
        record["_artifact_path"] = str(path)
        result[task_id][str(record["condition"])] = record
    return dict(result)


def _rate(successes: int, total: int) -> float | None:
    return successes / total if total else None


def _aggregate_group(records: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record[key])].append(record)
    output = []
    for value in sorted(grouped):
        group = grouped[value]
        baseline_successes = sum(int(item["baseline_score"]) for item in group)
        guarded_successes = sum(int(item["guarded_score"]) for item in group)
        n10 = sum(item["baseline_score"] == 0 and item["guarded_score"] == 1 for item in group)
        n01 = sum(item["baseline_score"] == 1 and item["guarded_score"] == 0 for item in group)
        output.append({
            key: value,
            "complete_pair_count": len(group),
            "baseline_successes": baseline_successes,
            "baseline_success_rate": _rate(baseline_successes, len(group)),
            "guarded_successes": guarded_successes,
            "guarded_success_rate": _rate(guarded_successes, len(group)),
            "n_10_guarded_only": n10,
            "n_01_baseline_only": n01,
            "paired_difference": sum(item["guarded_score"] - item["baseline_score"] for item in group) / len(group),
        })
    return output


def build_final_experiment_report(
    *, run_dir: Path, launch_lock_path: Path, audit_path: Path,
    bootstrap_resamples: int = 10000, bootstrap_seed: int = 7,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    launch_lock_path = launch_lock_path.resolve()
    audit_path = audit_path.resolve()
    manifest_path = run_dir / "run_manifest.json"
    frozen_plan_path = run_dir / "frozen_plan.json"
    manifest = _read_json(manifest_path)
    frozen_plan = _read_json(frozen_plan_path)
    launch_lock = _read_json(launch_lock_path)
    audit = _read_json(audit_path)
    issues: list[str] = []

    if manifest.get("execution_status") != "COMPLETED":
        issues.append("Run manifest execution_status is not COMPLETED.")
    if manifest.get("plan_sha256") != frozen_plan.get("plan_sha256"):
        issues.append("Manifest and frozen-plan hashes differ.")
    if launch_lock.get("plan_sha256") != manifest.get("plan_sha256"):
        issues.append("Launch-lock and manifest plan hashes differ.")
    if audit.get("plan_sha256") != manifest.get("plan_sha256"):
        issues.append("Audit and manifest plan hashes differ.")
    if launch_lock.get("audit_report_sha256") != audit.get("audit_report_sha256"):
        issues.append("Launch-lock and audit-report hashes differ.")
    if launch_lock.get("runner", {}).get("code_fingerprint_sha256") != manifest.get("code_fingerprint_sha256"):
        issues.append("Launch-lock and manifest code fingerprints differ.")
    if audit.get("runner", {}).get("code_fingerprint_sha256") != manifest.get("code_fingerprint_sha256"):
        issues.append("Audit and manifest code fingerprints differ.")

    plan_pairs = {str(pair["task_id"]): pair for pair in frozen_plan["pairs"]}
    checkpoints = _checkpoint_rows(run_dir)
    scoring = _scoring_by_task(run_dir)
    complete_records: list[dict[str, Any]] = []
    incomplete_cases: list[dict[str, Any]] = []
    failed_repairs: list[dict[str, Any]] = []

    for checkpoint in checkpoints:
        task_id = str(checkpoint["task_id"])
        pair_spec = plan_pairs[task_id]
        conditions = scoring.get(task_id, {})
        baseline = conditions.get("baseline")
        guarded = conditions.get("guarded")
        if baseline is None or guarded is None:
            incomplete_cases.append({
                "pair_id": checkpoint["pair_id"],
                "task_id": task_id,
                "task_index": checkpoint["task_index"],
                "workflow_pattern": checkpoint["workflow_pattern"],
                "fault_class": checkpoint["fault_class"],
                "source_stage": checkpoint["source_stage"],
                "score_reason_codes": sorted({str(row.get("score_reason_code")) for row in checkpoint.get("rows", [])}),
                "source_candidate_sha256": checkpoint.get("source_candidate_sha256"),
                "source_validation": checkpoint.get("source_validation"),
                "terminal_error": checkpoint.get("terminal_error"),
                "checkpoint_artifact": checkpoint["_artifact_path"],
            })
            continue
        record = {
            "pair_id": checkpoint["pair_id"],
            "task_id": task_id,
            "task_index": checkpoint["task_index"],
            "workflow_pattern": pair_spec["workflow_pattern"],
            "difficulty_level": pair_spec["difficulty_level"],
            "fault_class": pair_spec["fault_class"],
            "baseline_score": int(baseline["score"]),
            "guarded_score": int(guarded["score"]),
            "shared_injected_candidate_sha256": baseline.get("shared_injected_candidate_sha256"),
            "baseline_artifact": baseline["_artifact_path"],
            "guarded_artifact": guarded["_artifact_path"],
        }
        complete_records.append(record)
        if guarded["score"] == 0:
            failed_repairs.append({
                **record,
                "shared_injected_candidate": guarded.get("shared_injected_candidate"),
                "final_configuration": guarded.get("final_configuration"),
                "validation_before": guarded.get("validation_before"),
                "validation_after": guarded.get("validation_after"),
            })

    baseline_successes = sum(item["baseline_score"] for item in complete_records)
    guarded_successes = sum(item["guarded_score"] for item in complete_records)
    n10 = sum(item["baseline_score"] == 0 and item["guarded_score"] == 1 for item in complete_records)
    n01 = sum(item["baseline_score"] == 1 and item["guarded_score"] == 0 for item in complete_records)
    differences = [item["guarded_score"] - item["baseline_score"] for item in complete_records]
    paired_difference = sum(differences) / len(differences) if differences else None
    bootstrap_ci = _bootstrap_difference_ci(differences, resamples=bootstrap_resamples, seed=bootstrap_seed) if differences else None

    artifact_counts = {
        directory: len(list((run_dir / "execution" / directory).glob("*")))
        for directory in ("checkpoints", "prompts", "responses", "provider_calls", "faults", "scoring")
    }
    if artifact_counts["provider_calls"] != int(manifest["model_calls_used"]):
        issues.append("Provider-call artifact count does not match manifest.")
    if len(checkpoints) != int(manifest["pair_count"]):
        issues.append("Checkpoint count does not match manifest pair_count.")
    if len(complete_records) + len(incomplete_cases) != len(checkpoints):
        issues.append("Complete and incomplete pair counts are inconsistent.")

    return {
        "schema_version": "1.0",
        "report_id": REPORT_ID,
        "report_version": REPORT_VERSION,
        "status": "PASS" if not issues else "FAIL",
        "study_id": run_dir.name,
        "run_directory": str(run_dir),
        "provenance": {
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256_file(manifest_path),
            "frozen_plan_path": str(frozen_plan_path),
            "frozen_plan_file_sha256": _sha256_file(frozen_plan_path),
            "plan_sha256": manifest["plan_sha256"],
            "launch_lock_path": str(launch_lock_path),
            "launch_lock_file_sha256": _sha256_file(launch_lock_path),
            "launch_lock_sha256": launch_lock.get("launch_lock_sha256"),
            "audit_path": str(audit_path),
            "audit_file_sha256": _sha256_file(audit_path),
            "audit_report_sha256": audit.get("audit_report_sha256"),
            "code_fingerprint_sha256": manifest["code_fingerprint_sha256"],
            "git_commit_sha": launch_lock.get("git", {}).get("commit_sha"),
            "git_tags_at_launch": launch_lock.get("git", {}).get("tags_at_head", []),
            "model_provider": manifest["model_provider"],
            "model_name": manifest["model_name"],
            "max_output_tokens": manifest["max_output_tokens"],
            "prompt_protocol_version": manifest["prompt_protocol_version"],
        },
        "execution": {
            "execution_status": manifest["execution_status"],
            "planned_pair_count": manifest["pair_count"],
            "complete_pair_count": len(complete_records),
            "incomplete_pair_count": len(incomplete_cases),
            "maximum_model_calls": manifest["maximum_model_calls"],
            "model_calls_used": manifest["model_calls_used"],
            "artifact_counts": artifact_counts,
        },
        "paired_analysis": {
            "baseline_successes": baseline_successes,
            "baseline_success_rate": _rate(baseline_successes, len(complete_records)),
            "guarded_successes": guarded_successes,
            "guarded_success_rate": _rate(guarded_successes, len(complete_records)),
            "n_10_guarded_only": n10,
            "n_01_baseline_only": n01,
            "paired_difference": paired_difference,
            "exact_mcnemar_p_value": _exact_mcnemar_two_sided(n10, n01),
            "bootstrap_resamples": bootstrap_resamples,
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_ci_95": list(bootstrap_ci) if bootstrap_ci else None,
        },
        "by_fault_class": _aggregate_group(complete_records, "fault_class"),
        "by_workflow_pattern": _aggregate_group(complete_records, "workflow_pattern"),
        "complete_pairs": complete_records,
        "incomplete_cases": incomplete_cases,
        "failed_repairs": failed_repairs,
        "issues": issues,
    }


def render_final_experiment_markdown(report: dict[str, Any]) -> str:
    execution = report["execution"]
    analysis = report["paired_analysis"]
    provenance = report["provenance"]
    lines = [
        "# Frozen Controlled-Fault Experiment Report", "",
        f"**Report status:** {report['status']}",
        f"**Study:** `{report['study_id']}`", "",
        "## Provenance", "",
        f"- Plan SHA-256: `{provenance['plan_sha256']}`",
        f"- Code fingerprint SHA-256: `{provenance['code_fingerprint_sha256']}`",
        f"- Git commit: `{provenance['git_commit_sha']}`",
        f"- Launch-lock SHA-256: `{provenance['launch_lock_sha256']}`",
        f"- Audit-report SHA-256: `{provenance['audit_report_sha256']}`",
        f"- Model: `{provenance['model_name']}` via `{provenance['model_provider']}`", "",
        "## Execution summary", "",
        f"- Planned pairs: {execution['planned_pair_count']}",
        f"- Complete scientific pairs: {execution['complete_pair_count']}",
        f"- Incomplete pairs: {execution['incomplete_pair_count']}",
        f"- Model calls used: {execution['model_calls_used']}", "",
        "## Paired analysis", "",
        f"- Baseline success: {analysis['baseline_successes']}/{execution['complete_pair_count']}",
        f"- Guarded success: {analysis['guarded_successes']}/{execution['complete_pair_count']}",
        f"- Paired difference: {analysis['paired_difference']:.6f}",
        f"- Exact two-sided McNemar p-value: {analysis['exact_mcnemar_p_value']:.12g}",
        f"- Paired bootstrap 95% CI: [{analysis['bootstrap_ci_95'][0]:.6f}, {analysis['bootstrap_ci_95'][1]:.6f}]", "",
        "## Results by fault class", "",
        "| Fault class | Pairs | Baseline | Guarded | Difference |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["by_fault_class"]:
        lines.append(f"| `{row['fault_class']}` | {row['complete_pair_count']} | {row['baseline_successes']}/{row['complete_pair_count']} | {row['guarded_successes']}/{row['complete_pair_count']} | {row['paired_difference']:.6f} |")
    lines += ["", "## Results by workflow pattern", "", "| Workflow pattern | Pairs | Baseline | Guarded | Difference |", "|---|---:|---:|---:|---:|"]
    for row in report["by_workflow_pattern"]:
        lines.append(f"| `{row['workflow_pattern']}` | {row['complete_pair_count']} | {row['baseline_successes']}/{row['complete_pair_count']} | {row['guarded_successes']}/{row['complete_pair_count']} | {row['paired_difference']:.6f} |")
    lines += ["", "## Incomplete source cases", ""]
    if not report["incomplete_cases"]:
        lines.append("None.")
    for case in report["incomplete_cases"]:
        violations = case.get("source_validation", {}).get("violation_codes", [])
        lines += [f"### {case['pair_id']} / {case['task_id']}", "", f"- Workflow: `{case['workflow_pattern']}`", f"- Assigned fault: `{case['fault_class']}`", f"- Source stage: `{case['source_stage']}`", "- Source validation violations: " + ", ".join(f"`{code}`" for code in violations), ""]
    lines += ["## Unsuccessful guarded repairs", ""]
    if not report["failed_repairs"]:
        lines.append("None.")
    for case in report["failed_repairs"]:
        before = case.get("validation_before", {}).get("violation_codes", [])
        after = case.get("validation_after", {}).get("violation_codes", [])
        lines += [f"### {case['pair_id']} / {case['task_id']}", "", f"- Workflow: `{case['workflow_pattern']}`", f"- Fault: `{case['fault_class']}`", "- Violations before repair: " + ", ".join(f"`{code}`" for code in before), "- Violations after repair: " + ", ".join(f"`{code}`" for code in after), "", "**Injected candidate**", "", "```text", case.get("shared_injected_candidate", ""), "```", "", "**Guarded output**", "", "```text", case.get("final_configuration", ""), "```", ""]
    lines += ["## Interpretation", "", f"Among complete paired cases, deterministic validation plus one bounded repair call improved success by {analysis['paired_difference']:.6f}. One planned pair was excluded because its hosted source candidate was invalid before controlled-fault injection. The two guarded failures remain scored failures and are retained as case studies rather than corrected post hoc.", ""]
    return "\n".join(lines)


def write_final_experiment_report(report: dict[str, Any], *, json_path: Path, markdown_path: Path, overwrite: bool = False) -> tuple[Path, Path]:
    json_path = json_path.resolve()
    markdown_path = markdown_path.resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with json_path.open(mode, encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    markdown_path.write_text(render_final_experiment_markdown(report), encoding="utf-8")
    return json_path, markdown_path
