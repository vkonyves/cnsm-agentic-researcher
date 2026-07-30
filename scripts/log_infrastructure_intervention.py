#!/usr/bin/env python3
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
from cnsm_agentic.autonomous_research.final_guardrails import validate_intervention
p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--category',required=True);p.add_argument('--description',required=True);a=p.parse_args();prov=a.run_dir/'provenance';policy=json.loads((prov/'intervention_policy.json').read_text());entry={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'category':a.category,'description':a.description};validate_intervention(entry,policy);open(prov/'intervention_log.jsonl','a').write(json.dumps(entry)+chr(10));print('Intervention logged')
