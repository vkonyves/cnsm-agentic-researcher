from pathlib import Path

from cnsm_agentic.autonomous_research.final_pipeline import (
    audit_manuscript_artifact_references,
)
from cnsm_agentic.autonomous_research.final_schemas import (
    ManuscriptPackage,
    ManuscriptSections,
)


def manuscript_with_results(results: str):
    return ManuscriptPackage(
        title="Test",
        abstract="Test",
        sections=ManuscriptSections(
            introduction="",
            related_work="",
            methodology="",
            results=results,
            discussion="",
            conclusion="",
        ),
        figure_captions=[],
        table_captions=[],
        cited_record_ids=[],
        disclosure_statement="",
        limitations=[],
    )


def test_valid_archived_path_and_hash_pass(tmp_path):
    artifact = tmp_path / "analysis" / "results.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        '{"ok": true}',
        encoding="utf-8",
    )

    import hashlib
    digest = hashlib.sha256(
        artifact.read_bytes()
    ).hexdigest()

    manuscript = manuscript_with_results(
        "The result is archived as analysis/results.json "
        f"(SHA-256 = {digest})."
    )

    audit = audit_manuscript_artifact_references(
        manuscript=manuscript,
        run_dir=tmp_path,
    )

    assert audit["passed"] is True
    assert audit["issues"] == []


def test_missing_claimed_archived_artifact_fails(tmp_path):
    manuscript = manuscript_with_results(
        "The extracted slice is archived as "
        "analysis/repair_invocations_slice.json."
    )

    audit = audit_manuscript_artifact_references(
        manuscript=manuscript,
        run_dir=tmp_path,
    )

    assert audit["passed"] is False
    assert any(
        "does not exist" in issue
        for issue in audit["issues"]
    )


def test_wrong_hash_fails(tmp_path):
    artifact = (
        tmp_path
        / "execution"
        / "provider_calls"
        / "task-000008-guarded-repair.json"
    )
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        '{"real": true}',
        encoding="utf-8",
    )

    manuscript = manuscript_with_results(
        "execution/provider_calls/"
        "task-000008-guarded-repair.json "
        "SHA-256 = "
        + ("a" * 64)
    )

    audit = audit_manuscript_artifact_references(
        manuscript=manuscript,
        run_dir=tmp_path,
    )

    assert audit["passed"] is False
    assert any(
        "mismatch" in issue.lower()
        for issue in audit["issues"]
    )


def test_fabricated_provider_call_path_fails(tmp_path):
    real = (
        tmp_path
        / "execution"
        / "provider_calls"
        / "task-000008-guarded-repair.json"
    )
    real.parent.mkdir(parents=True)
    real.write_text(
        '{"real": true}',
        encoding="utf-8",
    )

    manuscript = manuscript_with_results(
        "repair_provider_trace_path: "
        "execution/provider_calls/task-000041-repair.json, "
        "repair_provider_trace_sha256: "
        + ("b" * 64)
    )

    audit = audit_manuscript_artifact_references(
        manuscript=manuscript,
        run_dir=tmp_path,
    )

    assert audit["passed"] is False
    assert any(
        "does not exist" in issue
        or "missing" in issue
        for issue in audit["issues"]
    )


def test_glob_archived_claim_requires_at_least_one_match(
    tmp_path,
):
    manuscript = manuscript_with_results(
        "Validator traces are archived under "
        "execution/scoring/*.json."
    )

    audit = audit_manuscript_artifact_references(
        manuscript=manuscript,
        run_dir=tmp_path,
    )

    assert audit["passed"] is False

    scoring = tmp_path / "execution" / "scoring"
    scoring.mkdir(parents=True)
    (scoring / "task-1.json").write_text(
        "{}",
        encoding="utf-8",
    )

    audit = audit_manuscript_artifact_references(
        manuscript=manuscript,
        run_dir=tmp_path,
    )

    assert audit["passed"] is True


def test_unrelated_nearby_hash_is_not_paired_with_path(
    tmp_path,
):
    manifest = tmp_path / "execution" / "execution_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("manifest", encoding="utf-8")

    other = tmp_path / "analysis" / "missingness_summary.csv"
    other.parent.mkdir(parents=True)
    other.write_text("missingness", encoding="utf-8")

    import hashlib

    other_sha = hashlib.sha256(
        other.read_bytes()
    ).hexdigest()

    manuscript = manuscript_with_results(
        "The execution manifest is archived at "
        "execution/execution_manifest.json. "
        "Missingness is summarized separately in "
        "analysis/missingness_summary.csv "
        f"(SHA-256 = {other_sha})."
    )

    audit = audit_manuscript_artifact_references(
        manuscript=manuscript,
        run_dir=tmp_path,
    )

    assert audit["passed"] is True


def test_reproduction_output_path_need_not_preexist(
    tmp_path,
):
    source = tmp_path / "execution" / "raw_results.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text("{}\n", encoding="utf-8")

    manuscript = manuscript_with_results(
        "Run the extraction command on "
        "execution/raw_results.jsonl and write the output to "
        "analysis/pairs_slice.ndjson."
    )

    audit = audit_manuscript_artifact_references(
        manuscript=manuscript,
        run_dir=tmp_path,
    )

    assert audit["passed"] is True


def test_all_structured_provider_records_are_checked(
    tmp_path,
):
    manuscript = manuscript_with_results(
        'repair_provider_trace_path: '
        '"execution/provider_calls/task-000041-repair.json", '
        'repair_provider_trace_sha256: "'
        + ("a" * 64)
        + '"\n\n'
        'repair_provider_trace_path: '
        '"execution/provider_calls/task-000127-repair.json", '
        'repair_provider_trace_sha256: "'
        + ("b" * 64)
        + '"\n\n'
        'repair_provider_trace_path: '
        '"execution/provider_calls/task-000159-repair.json", '
        'repair_provider_trace_sha256: "'
        + ("c" * 64)
        + '"'
    )

    audit = audit_manuscript_artifact_references(
        manuscript=manuscript,
        run_dir=tmp_path,
    )

    assert audit["passed"] is False

    text = "\n".join(audit["issues"])

    assert "task-000041-repair.json" in text
    assert "task-000127-repair.json" in text
    assert "task-000159-repair.json" in text


def test_input_results_hash_is_not_file_hash_claim(
    tmp_path,
):
    results = tmp_path / "analysis" / "results.json"
    results.parent.mkdir(parents=True)
    results.write_text(
        '{"input_results_sha256": "placeholder"}',
        encoding="utf-8",
    )

    raw = tmp_path / "execution" / "raw_results.jsonl"
    raw.parent.mkdir(parents=True)
    raw.write_text(
        "{}\n",
        encoding="utf-8",
    )

    import hashlib

    raw_sha = hashlib.sha256(
        raw.read_bytes()
    ).hexdigest()

    manuscript = manuscript_with_results(
        "Analysis executor inputs: "
        "analysis/results.json "
        f"(input results SHA-256 = {raw_sha})."
    )

    audit = audit_manuscript_artifact_references(
        manuscript=manuscript,
        run_dir=tmp_path,
    )

    assert audit["passed"] is True
