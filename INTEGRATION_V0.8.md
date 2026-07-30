# Integration

```bash
tar -xzf v0.8-final-autonomous-run-bootstrap.tar.gz
rsync -a v0.8-final-autonomous-run-bootstrap/ ./
python -m compileall -q src scripts
pytest tests/test_final_guardrails.py tests/test_final_schemas.py
```

Dry rehearsal:

```bash
python scripts/freeze_final_run.py \
  --master-prompt configs/final_run/master_prompt.template.txt \
  --output-dir studies/development/final_bootstrap_rehearsal/provenance \
  --development-rehearsal

python scripts/run_final_autonomous_research.py \
  --master-prompt configs/final_run/master_prompt.template.txt \
  --run-dir studies/development/final_bootstrap_rehearsal \
  --model gpt-5-mini \
  --development-rehearsal
```
