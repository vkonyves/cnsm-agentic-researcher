# Integration

From the repository root:

```bash
tar -xzf v0.4.1-finalist-resolution.tar.gz

rsync -a   v0.4.1-finalist-resolution/   ./
```

Apply the small configuration changes described in `PATCHES_REQUIRED.md`.

Then run:

```bash
python -m compileall -q src scripts
pytest tests/test_finalist_resolution.py
```

Regenerate the tied tournament:

```bash
rm -rf studies/generated/llm_netops_invariance

python scripts/run_study_design.py   --programme configs/research_programmes/llm_netops_invariance.json   --output studies/generated/llm_netops_invariance
```

Resolve the finalists:

```bash
python scripts/resolve_study_finalists.py   --programme configs/research_programmes/llm_netops_invariance.json   --study-dir studies/generated/llm_netops_invariance
```

Expected result:

```text
Selected candidate: C2
Planned follow-up: C4
Research state: DESIGN_SELECTED
```
