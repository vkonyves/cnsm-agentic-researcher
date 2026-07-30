from pathlib import Path
import pytest
from cnsm_agentic.autonomous_research.final_guardrails import assert_fresh_run_dir,assert_no_development_inputs

def test_rejects_development_path():
    with pytest.raises(ValueError):assert_fresh_run_dir(Path('studies/development/autonomous_discovery_v06'),development_rehearsal=False)
def test_rejects_development_payload():
    with pytest.raises(ValueError):assert_no_development_inputs({'x':'studies/development/evidence_repair_v07'})
def test_empty_dir_allowed(tmp_path):
    p=tmp_path/'final';p.mkdir();assert_fresh_run_dir(p,development_rehearsal=False)
