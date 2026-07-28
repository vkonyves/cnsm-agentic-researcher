# Required edit

In `src/cnsm_agentic/study_design/state_machine.py`, add:

```python
PREREGISTRATION_FROZEN = "PREREGISTRATION_FROZEN"
```

Permit:

```text
DESIGN_SELECTED -> PREREGISTRATION_FROZEN
```

No programme JSON changes are required.
