# Confirmatory analysis report

## Confirmatory dataset

- Questions: 300
- Tasks: 30
- Questions per task: 10

## Mapping validation

- Identity permutations found: 0
- Gold-answer mappings: verified for all records
- Repeat gold answers: verified against original records
- Semantic mapping method: normalized option-text matching

## Condition accuracy

| Condition | Correct | Accuracy |
|---|---:|---:|
| Original | 207 | 69.00% |
| Unchanged repeat | 219 | 73.00% |
| Permuted | 217 | 72.33% |

## Primary semantic-disagreement analysis

- Original vs repeat disagreement: 36/300 = 12.00%
- Original vs permutation disagreement: 71/300 = 23.67%
- Absolute excess disagreement: 0.1167 (11.67 percentage points)
- Relative disagreement ratio: 1.9722

## Paired instability-control counts

- Both agree: 212
- Both disagree: 19
- Permutation only disagrees: 52
- Repeat only disagrees: 17

## Exact McNemar test

- Discordant pair counts: b=52, c=17
- Exact two-sided p-value: 2.93043217455e-05

## Task-cluster bootstrap

- Repetitions: 10000
- Seed: 20260730
- 95% percentile interval, repeat disagreement rate: [0.0900, 0.1533]
- 95% percentile interval, permutation disagreement rate: [0.1833, 0.2900]
- 95% percentile interval, absolute excess disagreement: [0.0667, 0.1700]
- 95% percentile interval, relative disagreement ratio: [1.5122, 2.6667]

## Leave-one-task-out robustness analysis

This analysis was conducted post hoc as a robustness check.

- Tasks omitted in turn: 30
- Minimum excess disagreement: 10.34 percentage points
- Minimum obtained when omitting: T25
- Maximum excess disagreement: 12.76 percentage points
- Maximum obtained when omitting: T2
- Positive excess after every task omission: yes

The confirmatory effect was therefore not attributable to any single 6G-Bench task.

## Correctness transitions

### Original to unchanged repeat

- correct to correct: 198
- correct to incorrect: 9
- incorrect to correct: 21
- incorrect to incorrect: 72

### Original to permutation

- correct to correct: 182
- correct to incorrect: 25
- incorrect to correct: 35
- incorrect to incorrect: 58

## Interpretation boundary

The preregistered primary comparison concerns semantic answer disagreement relative to the unchanged-repeat instability control. Aggregate accuracy differences are secondary and do not by themselves measure permutation sensitivity.
