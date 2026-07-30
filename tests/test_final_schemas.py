import pytest
from cnsm_agentic.autonomous_research.final_schemas import ExperimentPlan,PreregistrationDocument

def test_zero_calls_rejected():
    with pytest.raises(ValueError):ExperimentPlan(study_id='s',adapter_family='x',implementation_strategy='i',public_resources=['r'],model_and_version_plan=['m'],task_manifest_strategy='t',transformation_manifest_strategy='x',execution_batches=['b'],randomisation_plan='r',caching_plan='c',failure_recovery_plan='f',result_schema='s',estimated_model_calls=0,estimated_compute_notes='n')
def test_unresolved_prereg_rejected():
    with pytest.raises(ValueError):PreregistrationDocument(study_id='s',title='t',research_question='q',confirmatory_hypotheses=['h'],exploratory_questions=[],evidence_record_ids=['r'],benchmark_scope=['b'],model_scope=['m'],transformation_scope=['t'],primary_estimand='e',secondary_estimands=[],sampling_plan='s',power_and_precision_plan='p',exclusion_rules=[],missingness_plan='m',analysis_plan='a',multiplicity_plan='m',contamination_plan='c',stopping_rule='s',planned_outputs=['o'],unresolved_critical_issues=['x'])
