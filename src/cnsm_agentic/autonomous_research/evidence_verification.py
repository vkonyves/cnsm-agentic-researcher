from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class EvidenceVerificationResult:
    total_records:int; referenced_record_count:int
    missing_record_ids:list[str]; metadata_incomplete_record_ids:list[str]
    metadata_only_record_ids:list[str]; duplicate_dois:list[str]
    quality_score:float; critical_issues:list[str]; warnings:list[str]
    def to_dict(self)->dict[str,Any]: return self.__dict__.copy()

def _doi(v:str|None)->str|None:
    return v.strip().lower().removeprefix('https://doi.org/') if v else None

def collect_referenced_ids(*,synthesis:dict,candidates:dict,decision:dict)->set[str]:
    out:set[str]=set()
    for section in ('established_findings','unresolved_questions','candidate_gaps'):
        for claim in synthesis.get(section,[]): out.update(claim.get('evidence_record_ids',[]))
    for c in candidates.get('candidates',[]):
        out.update(c.get('novelty_evidence_ids',[])); out.update(c.get('feasibility_evidence_ids',[]))
    out.update(decision.get('evidence_record_ids',[])); return out

def verify_evidence(*,records:list[dict],synthesis:dict,candidates:dict,decision:dict)->EvidenceVerificationResult:
    by_id={str(r['record_id']):r for r in records}; refs=collect_referenced_ids(synthesis=synthesis,candidates=candidates,decision=decision)
    missing=sorted(refs-set(by_id)); incomplete=[]; metadata_only=[]; abstracts=0
    for rid in sorted(refs&set(by_id)):
        r=by_id[rid]
        if not(str(r.get('title','')).strip() and r.get('publication_year') is not None and (r.get('doi') or r.get('url'))): incomplete.append(rid)
        if r.get('abstract'): abstracts+=1
        else: metadata_only.append(rid)
    counts=Counter(d for d in (_doi(r.get('doi')) for r in records) if d)
    dup=sorted(d for d,n in counts.items() if n>1)
    critical=[]; warnings=[]
    if missing: critical.append('Evidence IDs do not resolve to retrieved records.')
    if incomplete: critical.append('Referenced records lack complete bibliographic identity.')
    if refs and abstracts==0: critical.append('No referenced record contains an abstract.')
    if metadata_only: warnings.append('Some support is metadata-only and needs deeper verification in the final run.')
    if dup: warnings.append('Duplicate DOI records remain.')
    den=max(len(refs),1); score=round(.4*(len(refs)-len(missing))/den+.35*(len(refs)-len(incomplete))/den+.25*abstracts/den,6)
    return EvidenceVerificationResult(len(records),len(refs),missing,incomplete,metadata_only,dup,score,critical,warnings)
