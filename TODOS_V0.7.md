# v0.7 TODOs

1. Keep v0.6 and its development run unchanged.
2. Integrate, compile, and run both tests.
3. Run the v0.7 development rehearsal.
4. Check only structural properties: JSON validity, evidence-ID resolution,
   no placeholders, internally consistent readiness state.
5. Do not manually change candidate, hypotheses, sample size, budget, model,
   analysis, or conclusions.
6. After a successful rehearsal, commit and tag:

```bash
git add src/cnsm_agentic/autonomous_research \
  scripts/run_evidence_verification_and_repair.py \
  tests/test_evidence_verification.py tests/test_design_repair.py \
  README_V0.7.md INTEGRATION_V0.7.md TODOS_V0.7.md VERSION
git commit -m "Add evidence verification and autonomous design repair"
git tag -a v0.7-autonomous-research-framework \
  -m "Validate evidence and repair autonomous study designs"
git push origin main
git push origin v0.7-autonomous-research-framework
```

Do not normally commit `studies/development/evidence_repair_v07/`.
After this tag, stop modifying scientific architecture and freeze the final
framework commit plus Research Director master prompt.
