import json

from cnsm_agentic.autonomous_research.final_pipeline import (
    _compact_execution_manifest_for_manuscript,
    _compact_manuscript_evidence_bundle,
    _compact_verified_records_for_manuscript,
    _manuscript_revision_context,
)


def encoded_size(value):
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def test_execution_manifest_drops_bulk_hash_inventory():
    manifest = {
        "status": "COMPLETED",
        "completed_episode_count": 240,
        "model_calls_used": 120,
        "artifact_hashes": {
            f"execution/scoring/task-{i:06d}.json":
                "a" * 64
            for i in range(2000)
        },
    }

    compact = (
        _compact_execution_manifest_for_manuscript(
            manifest
        )
    )

    assert "artifact_hashes" not in compact
    assert compact["artifact_hash_count"] == 2000
    assert encoded_size(compact) < encoded_size(manifest)


def test_compact_records_omit_abstracts():
    records = [
        {
            "record_id": "R1",
            "title": "Paper",
            "abstract": "x" * 10000,
            "publication_year": 2026,
            "doi": "10.1/example",
            "url": "https://example.test",
            "source_api": "crossref",
            "authors": ["A"],
        }
    ]

    compact = (
        _compact_verified_records_for_manuscript(
            records
        )
    )

    assert compact[0]["title"] == "Paper"
    assert "abstract" not in compact[0]


def test_evidence_bundle_lists_are_bounded():
    bundle = {
        "execution_summary": {"completed": 240},
        "artifact_examples": [
            {"id": i}
            for i in range(100)
        ],
        "representative_tasks": [
            {"id": i}
            for i in range(100)
        ],
    }

    compact = _compact_manuscript_evidence_bundle(
        bundle
    )

    assert len(compact["artifact_examples"]) == 12
    assert len(compact["representative_tasks"]) == 8
    assert compact["artifact_examples_total_count"] == 100
    assert compact["representative_tasks_total_count"] == 100


def test_combined_revision_context_is_bounded():
    records = [
        {
            "record_id": f"R{i}",
            "title": f"Paper {i}",
            "abstract": "x" * 5000,
        }
        for i in range(80)
    ]

    manifest = {
        "status": "COMPLETED",
        "artifact_hashes": {
            f"path/{i}": "a" * 64
            for i in range(5000)
        },
    }

    bundle = {
        "artifact_examples": [
            {"payload": "x" * 1000}
            for _ in range(100)
        ],
        "representative_tasks": [
            {"payload": "x" * 1000}
            for _ in range(100)
        ],
    }

    full_size = encoded_size(
        {
            "records": records,
            "manifest": manifest,
            "bundle": bundle,
        }
    )

    compact = _manuscript_revision_context(
        records=records,
        execution_manifest=manifest,
        manuscript_evidence_bundle=bundle,
    )

    assert encoded_size(compact) < full_size * 0.25


def test_all_reviser_calls_use_compact_context():
    import re
    from pathlib import Path

    text = Path(
        "src/cnsm_agentic/autonomous_research/final_pipeline.py"
    ).read_text(encoding="utf-8")

    pattern = re.compile(
        r"await run_agent\(\s*"
        r"MANUSCRIPT_REVISER,\s*"
        r"\{"
        r".*?"
        r"expected_type=ManuscriptPackage,",
        flags=re.DOTALL,
    )

    call_blocks = [
        match.group(0)
        for match in pattern.finditer(text)
    ]

    assert len(call_blocks) == 5

    for block in call_blocks:
        assert (
            '"verified_records"' in block
        )
        assert (
            '"execution_manifest"' in block
        )
        assert (
            '"manuscript_evidence_bundle"' in block
        )

        # All three potentially large fields must come from the
        # deterministic bounded manuscript revision context.
        assert block.count(
            "manuscript_revision_context"
        ) >= 3

        # None of the five reviser calls may inject the original
        # unbounded sources directly.
        assert (
            '"verified_records": records'
            not in block
        )
        assert not re.search(
            r'"execution_manifest"\s*:\s*'
            r'\(\s*execution_manifest\s*\)',
            block,
        )
        assert not re.search(
            r'"manuscript_evidence_bundle"\s*:\s*'
            r'\(\s*manuscript_evidence_bundle\s*\)',
            block,
        )

