# Confirmatory permutation experiment

This directory contains compact provenance and aggregate results for the
preregistered CNSM 2026 confirmatory MCQA permutation experiment.

## Design

- Benchmark: 6G-Bench
- Model: gpt-5-nano
- Questions: 300
- Tasks: 30
- Questions per task: 10
- Permutation seed: 20260729
- Task-cluster bootstrap repetitions: 10,000
- Bootstrap seed: 20260730

## Conditions

- confirmatory-original-n300
- confirmatory-repeat-n300
- confirmatory-permuted-n300

## Primary result

- Repeat semantic disagreement: 36/300 = 12.00%
- Permutation semantic disagreement: 71/300 = 23.67%
- Absolute excess disagreement: 11.67 percentage points
- Relative disagreement ratio: 1.97
- Permutation-only disagreements: 52
- Repeat-only disagreements: 17
- Exact two-sided McNemar p-value: 2.93e-05
- Task-cluster bootstrap 95% interval for excess:
  6.67–17.00 percentage points

## Robustness

Leave-one-task-out excess disagreement remained positive for all 30 task
omissions, ranging from 10.34 to 12.76 percentage points.

Full prediction JSONL files and complete run directories are intentionally
excluded from Git.
