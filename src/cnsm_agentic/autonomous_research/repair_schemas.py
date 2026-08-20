from __future__ import annotations
from pydantic import BaseModel,Field,field_validator

class ContaminationRiskPlan(BaseModel):
    benchmark_names:list[str]; risk_factors:list[str]; detection_procedures:list[str]
    item_flagging_rules:list[str]; analysis_treatment:str; residual_uncertainty:list[str]
class BudgetScenario(BaseModel):
    scenario_id:str; description:str; planned_model_calls:int=Field(gt=0)
    models:list[str]; transformations:list[str]; discovery_items:int=Field(ge=0)
    confirmatory_items:int=Field(gt=0); task_cluster_count:int=Field(gt=0); feasibility_rationale:str
class PowerPlanningBrief(BaseModel):
    primary_estimand:str; target_effect:str; clustering_unit:str; calculation_method:str
    assumptions:list[str]; recommended_scenario_id:str; minimum_detectable_effect_notes:str; sensitivity_analyses:list[str]
class TransformationValidationPlan(BaseModel):
    transformation_families:list[str]; semantic_equivalence_checks:list[str]
    automatic_rejection_rules:list[str]; audit_sample_policy:str; mapping_validation:str; residual_risks:list[str]
class RepairedStudyDesign(BaseModel):
    selected_candidate_id:str; title:str; research_question:str; confirmatory_hypotheses:list[str]
    exploratory_questions:list[str]; benchmark_scope:list[str]; model_scope:list[str]
    transformation_scope:list[str]; primary_estimand:str; secondary_estimands:list[str]
    sampling_plan:str; analysis_plan:str; multiplicity_plan:str; missingness_plan:str
    contamination_plan:ContaminationRiskPlan; budget_scenarios:list[BudgetScenario]
    power_plan:PowerPlanningBrief; transformation_validation:TransformationValidationPlan
    preregistration_fields_complete:bool; unresolved_critical_issues:list[str]
    remaining_noncritical_uncertainties:list[str]; evidence_record_ids:list[str]
    @field_validator('title','research_question','primary_estimand','sampling_plan','analysis_plan','multiplicity_plan','missingness_plan')
    @classmethod
    def no_placeholders(cls,v:str)->str:
        if v.strip().upper() in {'','TBD','TODO','UNKNOWN','INVALID','PLACEHOLDER'}: raise ValueError('placeholder')
        return v.strip()
class RepairReadinessReport(BaseModel):
    readiness_status:str; selected_candidate_id:str; evidence_quality_score:float=Field(ge=0,le=1)
    passed_gates:list[str]; failed_gates:list[str]; warnings:list[str]; next_state:str
