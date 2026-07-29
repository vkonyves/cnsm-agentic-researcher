#!/usr/bin/env python3
from __future__ import annotations
import argparse,asyncio,hashlib,json
from pathlib import Path
from typing import Any
from cnsm_agentic.autonomous_research.design_repair import DESIGN_REPAIR_AGENT,READINESS_JUDGE,run_agent_with_retry
from cnsm_agentic.autonomous_research.evidence_verification import verify_evidence
from cnsm_agentic.autonomous_research.repair_schemas import RepairedStudyDesign,RepairReadinessReport

def read(p:Path)->Any:
    if not p.is_file(): raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding='utf-8'))
def write(p:Path,v:Any)->None:
    p.parent.mkdir(parents=True,exist_ok=True)
    if hasattr(v,'model_dump'): v=v.model_dump()
    p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
async def run(a):
    prog=read(a.programme); d=a.discovery_run
    paths={'records':d/'literature/records.json','synthesis':d/'literature/evidence_synthesis.json','candidates':d/'selection/candidates.json','reviews':d/'selection/critic_reviews.json','decision':d/'selection/decision.json','validation':d/'selection/candidate_validation.json'}
    data={k:read(v) for k,v in paths.items()}
    if data['validation'].get('candidate_validation_status')!='passed': raise ValueError('v0.6 validation failed')
    sid=data['decision']['selected_candidate_id']; by_id={c['candidate_id']:c for c in data['candidates']['candidates']}
    if sid not in by_id: raise ValueError('unknown selected candidate')
    ev=verify_evidence(records=data['records'],synthesis=data['synthesis'],candidates=data['candidates'],decision=data['decision'])
    a.output_dir.mkdir(parents=True,exist_ok=True); write(a.output_dir/'evidence_verification.json',ev.to_dict())
    payload={'programme':prog,'selected_candidate':by_id[sid],'selection_decision':data['decision'],'critic_reviews':data['reviews'],'evidence_synthesis':data['synthesis'],'evidence_verification':ev.to_dict(),'allowed_evidence_record_ids':sorted(r['record_id'] for r in data['records'])}
    repaired=await run_agent_with_retry(DESIGN_REPAIR_AGENT,payload,expected_type=RepairedStudyDesign,stage_name='Autonomous design repair')
    if repaired.selected_candidate_id!=sid: raise ValueError('candidate mismatch')
    allowed={r['record_id'] for r in data['records']}; bad=sorted(set(repaired.evidence_record_ids)-allowed)
    if bad: raise ValueError(f'unknown evidence IDs: {bad}')
    write(a.output_dir/'repaired_design.json',repaired)
    readiness=await run_agent_with_retry(READINESS_JUDGE,{'programme':prog,'selected_candidate_id':sid,'evidence_verification':ev.to_dict(),'repaired_design':repaired.model_dump()},expected_type=RepairReadinessReport,stage_name='Readiness judgement')
    ready=not ev.critical_issues and repaired.preregistration_fields_complete and not repaired.unresolved_critical_issues
    readiness.next_state='FRAMEWORK_VALIDATED_FOR_FINAL_RUN' if ready else 'DESIGN_REPAIR_REQUIRED'
    write(a.output_dir/'readiness_report.json',readiness)
    write(a.output_dir/'provenance.json',{'schema_version':'0.7.0','programme_sha256':sha(a.programme),'source_hashes':{k:sha(v) for k,v in paths.items()},'selected_candidate_id':sid,'evidence_quality_score':ev.quality_score,'next_state':readiness.next_state,'development_rehearsal':True})
    write(a.output_dir/'state.json',{'state':readiness.next_state,'selected_candidate_id':sid,'development_rehearsal':True})
    print('Evidence verification and design repair complete'); print('Selected candidate:',sid); print('Evidence quality score:',ev.quality_score); print('Next state:',readiness.next_state)
def main():
    p=argparse.ArgumentParser(); p.add_argument('--programme',type=Path,required=True); p.add_argument('--discovery-run',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--model',default='gpt-5-mini'); a=p.parse_args(); DESIGN_REPAIR_AGENT.model=a.model; READINESS_JUDGE.model=a.model; asyncio.run(run(a))
if __name__=='__main__':main()
