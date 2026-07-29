# Integration

```bash
tar -xzf v0.6-autonomous-discovery-bootstrap.tar.gz
rsync -a v0.6-autonomous-discovery-bootstrap/ ./
python -m compileall -q src scripts
pytest tests/test_neutral_autonomy.py
python -m pip install openai-agents pydantic
```

Development run:

```bash
python scripts/run_autonomous_discovery.py \
  --programme configs/research_programmes/llm_netops_invariance.json \
  --run-dir studies/development/autonomous_discovery_v06 \
  --model gpt-5-mini
```

This is still a development validation run, not the final autonomous paper run.
