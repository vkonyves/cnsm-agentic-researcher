from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Protocol


COMPLETED_STATUS = "COMPLETED"

PAIRED_BINARY_ANALYSIS_FAMILY = "paired_binary_analysis_v1"
COMPATIBLE_EXECUTION_ADAPTER_FAMILIES = (
    "synthetic_paired_llm_benchmark_v1",
    "hosted_netops_gvr_v1",
)
COMPATIBLE_RESULT_SCHEMA_ID = "paired_binary_episode_v1"
COMPATIBLE_RESULT_SCHEMA_VERSIONS = ("1.0",)

REQUIRED_COMPLETED_ANALYSIS_FIELDS = (
    "status",
    "schema_version",
    "analysis_executor",
    "study_id",
    "input_results_sha256",
    "confirmatory_results",
    "secondary_results",
    "missingness_summary",
    "exclusions",
    "deviations_from_preregistration",
    "results_path",
    "artifact_hashes",
    "warnings",
)


class AnalysisExecutor(Protocol):
    family: str
    aliases: tuple[str, ...]

    def supports(
        self,
        *,
        analysis_plan: dict[str, Any],
        execution_manifest: dict[str, Any],
    ) -> bool:
        ...

    def execute(
        self,
        *,
        analysis_plan: dict[str, Any],
        preregistration: dict[str, Any],
        execution_manifest: dict[str, Any],
        run_dir: Path,
    ) -> dict[str, Any]:
        ...


_ANALYSIS_EXECUTORS: list[
    AnalysisExecutor
] = []


def normalise_analysis_family(
    value: Any,
) -> str:
    text = str(
        value or ""
    ).strip().lower()

    characters: list[str] = []
    previous_was_separator = False

    for character in text:
        if character.isalnum():
            characters.append(
                character
            )
            previous_was_separator = False
        else:
            if (
                characters
                and not previous_was_separator
            ):
                characters.append(
                    "_"
                )
                previous_was_separator = True

    return "".join(
        characters
    ).strip("_")


def analysis_family_matches(
    value: Any,
    *,
    family: str,
    aliases: tuple[str, ...] = (),
) -> bool:
    requested = normalise_analysis_family(
        value
    )

    if not requested:
        return False

    supported = {
        normalise_analysis_family(
            identifier
        )
        for identifier in (
            family,
            *aliases,
        )
        if normalise_analysis_family(
            identifier
        )
    }

    return requested in supported


def _registered_identifiers(
    executor: AnalysisExecutor,
) -> set[str]:
    return {
        normalise_analysis_family(
            identifier
        )
        for identifier in (
            executor.family,
            *getattr(
                executor,
                "aliases",
                (),
            ),
        )
        if normalise_analysis_family(
            identifier
        )
    }


def register_analysis_executor(
    executor: AnalysisExecutor,
) -> None:
    new_identifiers = (
        _registered_identifiers(
            executor
        )
    )

    if not new_identifiers:
        raise ValueError(
            "Analysis executor must expose a "
            "non-empty family identifier."
        )

    for registered in _ANALYSIS_EXECUTORS:
        overlap = (
            new_identifiers
            & _registered_identifiers(
                registered
            )
        )

        if overlap:
            raise ValueError(
                "Analysis executor identifier "
                "already registered: "
                + ", ".join(
                    sorted(
                        overlap
                    )
                )
            )

    _ANALYSIS_EXECUTORS.append(
        executor
    )


def clear_registered_analysis_executors() -> None:
    """
    Clear the executor registry.

    Intended for isolated tests and controlled development setup.
    """
    _ANALYSIS_EXECUTORS.clear()


def registered_analysis_families() -> list[str]:
    return sorted(
        executor.family
        for executor
        in _ANALYSIS_EXECUTORS
    )


def resolve_analysis_executor(
    *,
    analysis_plan: dict[str, Any],
    execution_manifest: dict[str, Any],
) -> AnalysisExecutor | None:
    return next(
        (
            executor
            for executor
            in _ANALYSIS_EXECUTORS
            if executor.supports(
                analysis_plan=analysis_plan,
                execution_manifest=(
                    execution_manifest
                ),
            )
        ),
        None,
    )


def _sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def _safe_artifact_path(
    *,
    run_dir: Path,
    relative_path: str,
) -> Path | None:
    candidate = Path(
        relative_path
    )

    if candidate.is_absolute():
        return None

    if any(
        part == ".."
        for part in candidate.parts
    ):
        return None

    resolved = (
        run_dir
        / candidate
    ).resolve()

    run_root = run_dir.resolve()

    try:
        resolved.relative_to(
            run_root
        )
    except ValueError:
        return None

    return resolved


def validate_analysis_results(
    results: dict[str, Any],
    *,
    run_dir: Path,
    execution_manifest: dict[str, Any],
) -> list[str]:
    """
    Validate that analysis results are complete, inspectable,
    and tied to the actual execution artifacts.
    """
    issues: list[str] = []

    if not isinstance(
        results,
        dict,
    ):
        return [
            "Analysis results are not a dictionary."
        ]

    status = results.get(
        "status"
    )

    if status != COMPLETED_STATUS:
        issues.append(
            "Analysis status is not COMPLETED."
        )

    if status == COMPLETED_STATUS:
        for field in (
            REQUIRED_COMPLETED_ANALYSIS_FIELDS
        ):
            if field not in results:
                issues.append(
                    "Completed analysis results "
                    f"lack required field: {field}"
                )

    for field in (
        "confirmatory_results",
        "secondary_results",
        "exclusions",
        "deviations_from_preregistration",
        "warnings",
    ):
        value = results.get(
            field
        )

        if (
            value is not None
            and not isinstance(
                value,
                list,
            )
        ):
            issues.append(
                f"{field} must be a list."
            )

    missingness_summary = results.get(
        "missingness_summary"
    )

    if (
        missingness_summary is not None
        and not isinstance(
            missingness_summary,
            dict,
        )
    ):
        issues.append(
            "missingness_summary must be "
            "a dictionary."
        )

    confirmatory_results = results.get(
        "confirmatory_results"
    )

    if (
        status == COMPLETED_STATUS
        and isinstance(
            confirmatory_results,
            list,
        )
        and not confirmatory_results
    ):
        issues.append(
            "Completed analysis contains no "
            "confirmatory results."
        )

    execution_results_path = (
        execution_manifest.get(
            "results_path"
        )
    )

    execution_hashes = (
        execution_manifest.get(
            "artifact_hashes"
        )
    )

    expected_input_hash: str | None = None

    if (
        isinstance(
            execution_results_path,
            str,
        )
        and isinstance(
            execution_hashes,
            dict,
        )
    ):
        expected = execution_hashes.get(
            execution_results_path
        )

        if isinstance(
            expected,
            str,
        ):
            expected_input_hash = (
                expected.strip().lower()
            )

    supplied_input_hash = results.get(
        "input_results_sha256"
    )

    if (
        expected_input_hash
        and isinstance(
            supplied_input_hash,
            str,
        )
        and supplied_input_hash.strip().lower()
        != expected_input_hash
    ):
        issues.append(
            "Analysis input hash does not match "
            "the execution results hash."
        )

    if (
        expected_input_hash
        and not isinstance(
            supplied_input_hash,
            str,
        )
    ):
        issues.append(
            "Analysis input-results hash is missing."
        )

    results_path_value = results.get(
        "results_path"
    )

    resolved_results_path: Path | None = None

    if (
        not isinstance(
            results_path_value,
            str,
        )
        or not results_path_value.strip()
    ):
        issues.append(
            "results_path must be a "
            "non-empty relative path."
        )
    else:
        resolved_results_path = (
            _safe_artifact_path(
                run_dir=run_dir,
                relative_path=(
                    results_path_value
                ),
            )
        )

        if resolved_results_path is None:
            issues.append(
                "Analysis results path is unsafe: "
                f"{results_path_value}"
            )
        elif not resolved_results_path.is_file():
            issues.append(
                "Analysis results artifact does "
                f"not exist: {results_path_value}"
            )

    artifact_hashes = results.get(
        "artifact_hashes"
    )

    if (
        not isinstance(
            artifact_hashes,
            dict,
        )
        or not artifact_hashes
    ):
        issues.append(
            "Analysis artifact hashes are missing."
        )
    elif isinstance(artifact_hashes, dict):
        # A JSON document cannot contain its own final SHA-256 digest without
        # an impossible self-reference. Validate all supporting artifacts;
        # validate results.json through existence, required fields, and the
        # execution input-results hash linkage above.
        for relative_path, expected_hash in artifact_hashes.items():
            if not isinstance(relative_path, str) or not relative_path.strip():
                issues.append("Analysis artifact hash path is invalid.")
                continue

            artifact_path = _safe_artifact_path(
                run_dir=run_dir,
                relative_path=relative_path,
            )

            if artifact_path is None:
                issues.append(
                    "Analysis artifact path is unsafe: "
                    f"{relative_path}"
                )
                continue

            if not artifact_path.is_file():
                issues.append(
                    "Analysis artifact does not exist: "
                    f"{relative_path}"
                )
                continue

            if (
                not isinstance(expected_hash, str)
                or not expected_hash.strip()
            ):
                issues.append(
                    "Analysis artifact hash is missing for: "
                    f"{relative_path}"
                )
                continue

            if (
                _sha256_file(artifact_path).lower()
                != expected_hash.strip().lower()
            ):
                issues.append(
                    "Analysis artifact hash does not match: "
                    f"{relative_path}"
                )

    return sorted(
        set(
            issues
        )
    )


def paired_binary_analysis_compatibility_issues(
    *,
    analysis_plan: dict[str, Any],
    execution_manifest: dict[str, Any],
) -> list[str]:
    """Return explicit incompatibilities for paired binary analysis."""
    issues: list[str] = []
    if not analysis_family_matches(
        analysis_plan.get("analysis_executor"),
        family=PAIRED_BINARY_ANALYSIS_FAMILY,
        aliases=("paired-binary-analysis-v1",),
    ):
        issues.append("Analysis family is incompatible.")

    if execution_manifest.get("status") != COMPLETED_STATUS:
        issues.append("Execution manifest is not completed.")
    if execution_manifest.get("adapter_family") not in (
        COMPATIBLE_EXECUTION_ADAPTER_FAMILIES
    ):
        issues.append("Execution adapter family is incompatible.")
    if execution_manifest.get("result_schema_id") != (
        COMPATIBLE_RESULT_SCHEMA_ID
    ):
        issues.append("Execution result schema is incompatible.")
    if execution_manifest.get("result_schema_version") not in (
        COMPATIBLE_RESULT_SCHEMA_VERSIONS
    ):
        issues.append("Execution result schema version is incompatible.")

    if analysis_plan.get("study_id") != execution_manifest.get("study_id"):
        issues.append("Analysis and execution study IDs do not agree.")
    if analysis_plan.get("estimand") != (
        "paired_success_rate_difference_guarded_minus_baseline"
    ):
        issues.append("Analysis plan requests an incompatible estimand.")
    if analysis_plan.get("failed_call_treatment") not in {
        "complete_pair_primary",
        "complete_pair_primary_with_failure_as_zero_sensitivity",
    }:
        issues.append("Failed-call treatment is not explicitly supported.")

    for field in ("results_path", "result_schema_path", "artifact_hashes"):
        if not execution_manifest.get(field):
            issues.append(f"Execution manifest lacks required field: {field}")

    if execution_manifest.get("execution_mode") == "final_autonomous_run":
        if execution_manifest.get("human_scientific_intervention_after_launch") is not False:
            issues.append(
                "Final execution does not attest absence of human "
                "scientific intervention after launch."
            )
        if not execution_manifest.get("master_prompt_sha256"):
            issues.append("Final execution lacks sealed master-prompt provenance.")

    return sorted(set(issues))



def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _exact_mcnemar_two_sided(guarded_only: int, baseline_only: int) -> float:
    discordant = guarded_only + baseline_only
    if discordant == 0:
        return 1.0
    smaller = min(guarded_only, baseline_only)
    lower_tail = sum(
        math.comb(discordant, k) for k in range(smaller + 1)
    ) / (2 ** discordant)
    return min(1.0, 2.0 * lower_tail)


def _paired_bootstrap_interval(
    differences: list[int],
    *,
    seed: int,
    resamples: int,
    confidence_level: float,
) -> tuple[float, float]:
    if not differences:
        raise ValueError("Cannot bootstrap an empty paired sample.")
    rng = random.Random(seed)
    n = len(differences)
    estimates = []
    for _ in range(resamples):
        estimates.append(
            sum(differences[rng.randrange(n)] for _ in range(n)) / n
        )
    estimates.sort()
    alpha = 1.0 - confidence_level
    lower_index = max(0, int((alpha / 2.0) * resamples))
    upper_index = min(
        resamples - 1,
        int((1.0 - alpha / 2.0) * resamples) - 1,
    )
    return estimates[lower_index], estimates[upper_index]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_svg(
    path: Path,
    *,
    baseline_rate: float,
    guarded_rate: float,
    estimate: float,
    lower: float,
    upper: float,
) -> None:
    def x(value: float) -> float:
        return 70 + max(0.0, min(1.0, value)) * 400

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="560" height="260" viewBox="0 0 560 260">
<rect width="560" height="260" fill="white"/>
<text x="20" y="28" font-family="sans-serif" font-size="18">Paired binary development rehearsal</text>
<line x1="70" y1="210" x2="470" y2="210" stroke="black"/>
<rect x="{x(0):.2f}" y="65" width="{400*baseline_rate:.2f}" height="35" fill="#999"/>
<rect x="{x(0):.2f}" y="120" width="{400*guarded_rate:.2f}" height="35" fill="#555"/>
<text x="10" y="88" font-family="sans-serif" font-size="13">Baseline</text>
<text x="10" y="143" font-family="sans-serif" font-size="13">Guarded</text>
<text x="70" y="235" font-family="sans-serif" font-size="12">0</text>
<text x="455" y="235" font-family="sans-serif" font-size="12">1</text>
<text x="70" y="185" font-family="sans-serif" font-size="13">Difference={estimate:.3f}; 95% CI [{lower:.3f}, {upper:.3f}]</text>
</svg>\n"""
    path.write_text(svg, encoding="utf-8")


class PairedBinaryAnalysisExecutor:
    family = PAIRED_BINARY_ANALYSIS_FAMILY
    aliases = ("paired-binary-analysis-v1",)

    def supports(
        self,
        *,
        analysis_plan: dict[str, Any],
        execution_manifest: dict[str, Any],
    ) -> bool:
        return not paired_binary_analysis_compatibility_issues(
            analysis_plan=analysis_plan,
            execution_manifest=execution_manifest,
        )

    def execute(
        self,
        *,
        analysis_plan: dict[str, Any],
        preregistration: dict[str, Any],
        execution_manifest: dict[str, Any],
        run_dir: Path,
    ) -> dict[str, Any]:
        issues = paired_binary_analysis_compatibility_issues(
            analysis_plan=analysis_plan,
            execution_manifest=execution_manifest,
        )
        if issues:
            raise ValueError("Unsupported paired analysis: " + "; ".join(issues))

        results_relative = str(execution_manifest["results_path"])
        results_path = _safe_artifact_path(
            run_dir=run_dir,
            relative_path=results_relative,
        )
        if results_path is None or not results_path.is_file():
            raise FileNotFoundError(results_relative)
        expected_hash = execution_manifest["artifact_hashes"].get(
            results_relative
        )
        actual_hash = _sha256_file(results_path)
        if not isinstance(expected_hash, str) or actual_hash != expected_hash:
            raise ValueError("Execution results hash does not match manifest.")

        rows = [
            json.loads(line)
            for line in results_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        pairs: dict[str, dict[str, dict[str, Any]]] = {}
        exclusions: list[dict[str, Any]] = []
        for row in rows:
            pair_id = str(row.get("pair_id"))
            condition = str(row.get("condition"))
            bucket = pairs.setdefault(pair_id, {})
            if condition in bucket:
                raise ValueError(
                    "Duplicate condition row for "
                    f"{pair_id}/{condition}."
                )
            bucket[condition] = row

        complete: list[tuple[int, int]] = []
        missing_counts = {
            "complete_pairs": 0,
            "baseline_only_observed_pairs": 0,
            "guarded_only_observed_pairs": 0,
            "both_missing_pairs": 0,
            "failed_baseline_episodes": 0,
            "failed_guarded_episodes": 0,
            "unscorable_baseline_episodes": 0,
            "unscorable_guarded_episodes": 0,
        }
        contamination_counts: dict[tuple[str, str], int] = {}

        for pair_id, conditions in sorted(pairs.items()):
            baseline = conditions.get("baseline")
            guarded = conditions.get("guarded")
            baseline_score = baseline.get("score") if baseline else None
            guarded_score = guarded.get("score") if guarded else None
            baseline_observed = baseline_score in (0, 1) and not isinstance(
                baseline_score, bool
            )
            guarded_observed = guarded_score in (0, 1) and not isinstance(
                guarded_score, bool
            )
            baseline_failed = bool(
                baseline and baseline.get("call_status") == "FAILED"
            )
            guarded_failed = bool(
                guarded and guarded.get("call_status") == "FAILED"
            )
            if baseline_failed:
                missing_counts["failed_baseline_episodes"] += 1
            if guarded_failed:
                missing_counts["failed_guarded_episodes"] += 1
            if baseline and not baseline_observed and not baseline_failed:
                missing_counts["unscorable_baseline_episodes"] += 1
            if guarded and not guarded_observed and not guarded_failed:
                missing_counts["unscorable_guarded_episodes"] += 1
            for row in (baseline, guarded):
                if not row:
                    continue
                for flag in row.get("contamination_flags", []):
                    key = (str(flag), str(row["condition"]))
                    contamination_counts[key] = contamination_counts.get(key, 0) + 1

            if baseline_observed and guarded_observed:
                complete.append((int(guarded_score), int(baseline_score)))
                missing_counts["complete_pairs"] += 1
            elif baseline_observed:
                missing_counts["baseline_only_observed_pairs"] += 1
            elif guarded_observed:
                missing_counts["guarded_only_observed_pairs"] += 1
            else:
                missing_counts["both_missing_pairs"] += 1

        if not complete:
            raise ValueError("Paired analysis has no complete pairs.")

        n_11 = sum(g == 1 and b == 1 for g, b in complete)
        n_10 = sum(g == 1 and b == 0 for g, b in complete)
        n_01 = sum(g == 0 and b == 1 for g, b in complete)
        n_00 = sum(g == 0 and b == 0 for g, b in complete)
        n = len(complete)
        guarded_successes = n_11 + n_10
        baseline_successes = n_11 + n_01
        guarded_rate = guarded_successes / n
        baseline_rate = baseline_successes / n
        estimate = guarded_rate - baseline_rate
        p_value = _exact_mcnemar_two_sided(n_10, n_01)

        seed = int(analysis_plan.get("bootstrap_seed", 1729))
        resamples = int(analysis_plan.get("bootstrap_resamples", 10_000))
        confidence_level = float(analysis_plan.get("confidence_level", 0.95))
        lower, upper = _paired_bootstrap_interval(
            [g - b for g, b in complete],
            seed=seed,
            resamples=resamples,
            confidence_level=confidence_level,
        )

        analysis_dir = run_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        contingency_path = analysis_dir / "paired_contingency_table.csv"
        _write_csv(
            contingency_path,
            ["guarded", "baseline", "count"],
            [
                {"guarded": 1, "baseline": 1, "count": n_11},
                {"guarded": 1, "baseline": 0, "count": n_10},
                {"guarded": 0, "baseline": 1, "count": n_01},
                {"guarded": 0, "baseline": 0, "count": n_00},
            ],
        )
        condition_path = analysis_dir / "condition_summary.csv"
        _write_csv(
            condition_path,
            ["condition", "successes", "failures", "observed", "success_rate"],
            [
                {
                    "condition": "baseline",
                    "successes": baseline_successes,
                    "failures": n - baseline_successes,
                    "observed": n,
                    "success_rate": baseline_rate,
                },
                {
                    "condition": "guarded",
                    "successes": guarded_successes,
                    "failures": n - guarded_successes,
                    "observed": n,
                    "success_rate": guarded_rate,
                },
            ],
        )
        missing_path = analysis_dir / "missingness_summary.csv"
        _write_csv(
            missing_path,
            ["metric", "count"],
            [{"metric": key, "count": value} for key, value in missing_counts.items()],
        )
        contamination_path = analysis_dir / "contamination_summary.csv"
        _write_csv(
            contamination_path,
            ["flag", "condition", "episode_count"],
            [
                {"flag": flag, "condition": condition, "episode_count": count}
                for (flag, condition), count in sorted(contamination_counts.items())
            ],
        )
        figure_path = analysis_dir / "paired_difference_figure.svg"
        _write_summary_svg(
            figure_path,
            baseline_rate=baseline_rate,
            guarded_rate=guarded_rate,
            estimate=estimate,
            lower=lower,
            upper=upper,
        )
        log_path = analysis_dir / "analysis_log.jsonl"
        log_path.write_text(
            json.dumps({
                "event_type": "paired_analysis_completed",
                "study_id": execution_manifest["study_id"],
                "complete_pair_count": n,
                "bootstrap_seed": seed,
                "bootstrap_resamples": resamples,
            }, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        confirmatory = {
            "hypothesis_id": str(analysis_plan.get("hypothesis_id", "H1")),
            "estimand": "paired_success_rate_difference_guarded_minus_baseline",
            "complete_pair_count": n,
            "baseline_success_count": baseline_successes,
            "guarded_success_count": guarded_successes,
            "baseline_success_rate": baseline_rate,
            "guarded_success_rate": guarded_rate,
            "estimate": estimate,
            "n_11": n_11,
            "n_10": n_10,
            "n_01": n_01,
            "n_00": n_00,
            "discordant_ordering": "n_10=guarded_only;n_01=baseline_only",
            "test": "exact_mcnemar_two_sided",
            "p_value": p_value,
            "confidence_interval_method": "paired_bootstrap_percentile",
            "confidence_level": confidence_level,
            "confidence_interval_lower": lower,
            "confidence_interval_upper": upper,
            "bootstrap_seed": seed,
            "bootstrap_resamples": resamples,
        }
        warnings = []
        if execution_manifest.get("execution_mode") == "development_rehearsal":
            warnings.append("DEVELOPMENT_REHEARSAL_NOT_SCIENTIFIC_EVIDENCE")
        if n < 20:
            warnings.append("SMALL_COMPLETE_PAIR_COUNT_BOOTSTRAP_INTERVAL_UNSTABLE")

        results_payload = {
            "status": COMPLETED_STATUS,
            "schema_version": "1.0",
            "analysis_executor": self.family,
            "study_id": execution_manifest["study_id"],
            "input_results_sha256": actual_hash,
            "confirmatory_results": [confirmatory],
            "secondary_results": [],
            "missingness_summary": missing_counts,
            "exclusions": exclusions,
            "deviations_from_preregistration": [],
            "warnings": warnings,
            "execution_mode": execution_manifest.get("execution_mode"),
        }
        results_path_out = analysis_dir / "results.json"
        supporting_artifacts = [
            contingency_path,
            condition_path,
            missing_path,
            contamination_path,
            figure_path,
            log_path,
        ]
        artifact_hashes = {
            path.relative_to(run_dir).as_posix(): _sha256_file(path)
            for path in supporting_artifacts
        }
        results = dict(results_payload)
        results["results_path"] = results_path_out.relative_to(run_dir).as_posix()
        results["artifact_hashes"] = artifact_hashes
        results_path_out.write_text(
            json.dumps(results, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return results


def register_builtin_analysis_executors() -> None:
    """Register built-in analysis executors explicitly and idempotently."""
    if PAIRED_BINARY_ANALYSIS_FAMILY not in registered_analysis_families():
        register_analysis_executor(PairedBinaryAnalysisExecutor())
