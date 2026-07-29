from cnsm_agentic.autonomous_research.evidence_verification import verify_evidence

def test_missing_is_critical():
 r=verify_evidence(records=[{'record_id':'r1','title':'P','abstract':'A','publication_year':2025,'doi':'10/x','url':None}],synthesis={'established_findings':[{'evidence_record_ids':['r1','missing']}]},candidates={'candidates':[]},decision={'evidence_record_ids':[]})
 assert r.missing_record_ids==['missing'] and r.critical_issues

def test_complete_scores_one():
 r=verify_evidence(records=[{'record_id':'r1','title':'P','abstract':'A','publication_year':2025,'doi':'10/x','url':None}],synthesis={'established_findings':[{'evidence_record_ids':['r1']}]},candidates={'candidates':[]},decision={'evidence_record_ids':[]})
 assert r.quality_score==1.0
