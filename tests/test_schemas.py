from cnsm_agentic.schemas import CandidateExperiment, DiscoveryReport, MetricSpec, ResearchPlan, VerifiedResource

def test_discovery_report_accepts_unresolved_resource() -> None:
    report = DiscoveryReport(
        scope_summary="test",
        searched_resources=["Example"],
        resources=[
            VerifiedResource(
                name="Example",
                status="not_verified",
                repository_url=None,
                dataset_urls=[],
                dataset_location_notes=(
                    "No concrete dataset location was verified."
                ),
                task_types=[],
                cpu_feasibility="unknown",
                verified_claims=[],
                unresolved_questions=[
                    "No repository found."
                ],
                evidence=[],
            )
        ],
        rejected_or_unresolved_leads=[],
        discovery_limitations=["Smoke test"],
        next_verification_actions=[
            "Search primary sources"
        ],
    )

    assert report.resources[0].status == "not_verified"
    assert report.resources[0].dataset_urls == []

def test_research_plan_requires_operational_metric() -> None:
    metric=MetricSpec(name='accuracy', definition='Correct predictions divided by all predictions.', required_reference='Gold labels', automatically_computable=True)
    candidate=CandidateExperiment(title='Example', target_resources=['ExampleBench'], research_question='Does A improve accuracy?', hypothesis='A improves accuracy.', verified_facts_used=[], unverified_assumptions=['Gold labels may exist.'], required_verification_actions=['Verify labels.'], dataset_requirements=['Machine-readable tasks'], baselines=['Direct prompt'], metrics=[metric], cpu_feasibility='API based', estimated_api_call_pattern='One call per item', key_risks=['Dataset unavailable'])
    plan=ResearchPlan(experiment_family='llm_benchmark', scope_summary='test', candidates=[candidate, candidate.model_copy(update={'title':'Example 2'})], provisional_recommendation='Example', recommendation_reason='provisional', recommendation_dependencies=['Verify labels'], next_machine_actions=['Verify dataset'])
    assert len(plan.candidates)==2
