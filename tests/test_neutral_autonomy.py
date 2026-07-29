from pathlib import Path


def test_no_candidate_specific_scientific_branches() -> None:
    paths = list(Path("src/cnsm_agentic/autonomous_research").rglob("*.py")) + [Path("scripts/run_autonomous_discovery.py")]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    for token in ['candidate_id == "C2"', 'candidate_id == "C4"', 'supports selected C2', 'selected_candidate_id"] != "C2"']:
        assert token not in text


def test_selection_schema_is_unconstrained() -> None:
    from cnsm_agentic.autonomous_research.schemas import SelectionDecision
    field = SelectionDecision.model_fields["selected_candidate_id"]
    assert field.is_required()
