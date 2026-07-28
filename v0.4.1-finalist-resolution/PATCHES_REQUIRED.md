# Required small edits

## 1. `ResearchProgramme` in `models.py`

Add this field after `tie_band`:

```python
finalist_resolution_weights: dict[str, float] = field(
    default_factory=lambda: {
        "measurement_validity": 0.25,
        "confound_control": 0.20,
        "statistical_identifiability": 0.20,
        "intervention_executability": 0.15,
        "scientific_sequence_value": 0.10,
        "compute_feasibility": 0.10,
    }
)
```

Ensure the import contains:

```python
from dataclasses import asdict, dataclass, field
```

Add to `ResearchProgramme.validate()`:

```python
if (
    abs(
        sum(self.finalist_resolution_weights.values())
        - 1.0
    )
    > 1e-9
):
    raise ValueError(
        "finalist_resolution_weights must sum to 1.0"
    )
```

## 2. Programme JSON

Add:

```json
"finalist_resolution_weights": {
  "measurement_validity": 0.25,
  "confound_control": 0.20,
  "statistical_identifiability": 0.20,
  "intervention_executability": 0.15,
  "scientific_sequence_value": 0.10,
  "compute_feasibility": 0.10
}
```
