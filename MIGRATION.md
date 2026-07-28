# Integration

From the clean `cnsm-agentic-researcher` repository:

```bash
tar -xzf v0.4-autonomous-study-design.tar.gz
rsync -a v0.4-autonomous-study-design/ ./
python -m compileall -q src scripts
pytest
python scripts/run_study_design.py \
  --programme configs/research_programmes/llm_netops_invariance.json \
  --output studies/generated/llm_netops_invariance
```

Suggested commit and tag:

```bash
git add src/cnsm_agentic/study_design configs/research_programmes \
        scripts/run_study_design.py tests/test_study_design.py \
        README_V0.4.md MIGRATION.md VERSION

git commit -m "Add autonomous study-design architecture"
git tag -a v0.4-autonomous-study-design \
  -m "Scientific-method layer with design tournament and execution DAG"
git push origin main
git push origin v0.4-autonomous-study-design
```

The next step is to replace the deterministic candidate generator and critic
with SDK-backed agents that emit the same validated schemas.
