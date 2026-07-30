# v0.8 TODOs

1. Integrate and run compile/tests.
2. Run a development rehearsal with a deliberately non-final prompt.
3. Implement or register candidate-family experiment adapters before the real freeze.
4. Confirm the orchestrator stops at `AUTONOMOUS_EXECUTION_ADAPTER_REQUIRED` rather than inventing results.
5. Add IEEE LaTeX rendering and PDF page-count/reference checks.
6. Commit/tag `v0.8-final-autonomous-run-bootstrap`.
7. Only then write and hash the actual master prompt and launch a fresh final run.

After final launch, only logged infrastructure interventions are allowed.
