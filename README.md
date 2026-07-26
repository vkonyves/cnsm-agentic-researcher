# CNSM Agentic Researcher — Starter

A deliberately small first implementation for testing the OpenAI Agents SDK,
structured outputs, configuration, run folders, and topic adapters.

It does **not** yet perform the final CNSM research. It creates a machine-readable
pilot research plan for one of two experiment families:

- `llm_benchmark`: 6G-Bench / TSNBench / OpsEval-style LLM evaluation
- `tabular_ml`: O-RAN KPI, 5G attack detection, or similar tabular datasets

## 1. Activate the environment

```bash
conda activate cnsm-agentic
cd cnsm-agentic-starter
```

## 2. Install

```bash
python -m pip install --upgrade pip
pip install -e .
```

## 3. Configure

```bash
cp .env.example .env
```

Add your real `OPENAI_API_KEY` to `.env`. Do not commit `.env`.

The default model is `gpt-5-nano`, appropriate for this cheap smoke test.
Later, use stronger models for scientific design and review.

## 4. Check API access

```bash
python scripts/check_models.py
```

## 5. Run the toy pilot

LLM benchmark family:

```bash
python -m cnsm_agentic.cli plan --config configs/pilot_llm.yaml
```

Tabular network-data family:

```bash
python -m cnsm_agentic.cli plan --config configs/pilot_tabular.yaml
```

Outputs are written under `runs/<run_id>/`.

## 6. Run tests

```bash
pytest -q
```

## Design boundary

The orchestrator and provenance layer are shared. Each scientific family supplies
an adapter:

```text
Research workflow
  -> topic/data discovery
  -> question formulation
  -> protocol
  -> implementation
  -> execution
  -> analysis
  -> review
  -> paper

Experiment adapter
  -> llm_benchmark
  -> tabular_ml
```

This lets the final system investigate either CNSM topic without rewriting the
whole architecture.
