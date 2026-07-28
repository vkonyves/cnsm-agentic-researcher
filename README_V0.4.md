# v0.4 Autonomous Study Design

This additive implementation stage introduces the scientific-method layer for
`cnsm-agentic-researcher`.

It transforms a high-level research mandate into multiple falsifiable study
candidates, structured hypotheses, adversarial reviews, a scored design
tournament, a selected preregistration-ready plan, an experiment DAG, and an
initial claim-evidence ledger.

It deliberately does not write the paper. Writing should be enabled only after
experiments, falsification checks, and claim freezing.

## Example

```bash
python scripts/run_study_design.py \
  --programme configs/research_programmes/llm_netops_invariance.json \
  --output studies/generated/llm_netops_invariance
```
