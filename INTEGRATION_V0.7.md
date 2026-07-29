# Integration

```bash
tar -xzf v0.7-evidence-verification-and-design-repair.tar.gz
rsync -a v0.7-evidence-verification-and-design-repair/ ./
python -m compileall -q src scripts
pytest tests/test_evidence_verification.py tests/test_design_repair.py
python scripts/run_evidence_verification_and_repair.py \
  --programme configs/research_programmes/llm_netops_invariance.json \
  --discovery-run studies/development/autonomous_discovery_v06 \
  --output-dir studies/development/evidence_repair_v07 \
  --model gpt-5-mini
```
