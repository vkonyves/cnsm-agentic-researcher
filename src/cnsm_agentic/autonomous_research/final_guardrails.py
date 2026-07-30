from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any

MARKERS=("studies/development/","autonomous_discovery_v06","evidence_repair_v07")

def sha256_file(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def assert_fresh_run_dir(run_dir:Path,*,development_rehearsal:bool)->None:
    value=run_dir.as_posix().lower()
    if not development_rehearsal and any(m.lower() in value for m in MARKERS):
        raise ValueError(f"Final run cannot use development path: {run_dir}")
    if run_dir.exists():
        unexpected=[p.name for p in run_dir.iterdir() if p.name!='provenance']
        if unexpected:
            raise ValueError(f"Run directory is not fresh: {sorted(unexpected)}")

def assert_no_development_inputs(payload:Any)->None:
    value=json.dumps(payload,ensure_ascii=False,default=str).lower()
    for marker in MARKERS:
        if marker.lower() in value:
            raise ValueError(f"Development scientific input forbidden: {marker}")

def validate_intervention(entry:dict[str,Any],policy:dict[str,Any])->None:
    if entry.get('category') not in set(policy.get('allowed',[])):
        raise ValueError('Intervention category is not allowed')
    if not str(entry.get('description','')).strip():
        raise ValueError('Intervention description is required')
