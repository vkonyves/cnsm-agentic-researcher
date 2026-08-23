from __future__ import annotations
from pydantic import BaseModel, Field, field_validator, model_validator
BAD={'','TBD','TODO','UNKNOWN','INVALID','PLACEHOLDER'}

def clean(v:str)->str:
    v=v.strip()
    if v.upper() in BAD: raise ValueError('Placeholder forbidden')
    return v

class FrozenRunManifest(BaseModel):
    schema_version: str
    framework_commit: str
    framework_tag: str | None = None
    framework_dirty: bool
    master_prompt_sha256: str
    intervention_policy_sha256: str
    capability_manifest_sha256: str
    created_at_utc: str
    development_rehearsal: bool
    # v1 autonomous-paper freeze extensions. Optional for backward
    # compatibility with older manifests; required by the v1 launcher.
    paper_run_constraints_sha256: str | None = None
    model: str | None = None
    run_type: str | None = None

    @field_validator(
        "schema_version",
        "framework_commit",
        "master_prompt_sha256",
        "intervention_policy_sha256",
        "capability_manifest_sha256",
        "created_at_utc",
    )
    @classmethod
    def no_placeholders(
        cls,
        value: str,
    ) -> str:
        return clean(value)


class PreregistrationExecutionContract(BaseModel):
    adapter_family: str
    execution_mode: str
    design: str
    conditions: list[str]
    model_provider: str
    model_names: list[str]
    task_count: int = Field(gt=0)
    planned_episode_count: int = Field(gt=0)
    maximum_model_calls: int = Field(gt=0)

@model_validator(mode="after")
def structurally_complete(self):
    if not self.conditions:
        raise ValueError(
            "At least one executable condition is required."
        )

    if not self.model_names:
        raise ValueError(
            "At least one executable model is required."
        )

    return self


class PreregistrationDocument(BaseModel):
    study_id: str
    title: str
    research_question: str
    confirmatory_hypotheses: list[str]
    exploratory_questions: list[str]
    evidence_record_ids: list[str]
    benchmark_scope: list[str]
    model_scope: list[str]
    transformation_scope: list[str]
    execution_contract: PreregistrationExecutionContract
    primary_estimand: str
    secondary_estimands: list[str]
    sampling_plan: str
    power_and_precision_plan: str
    exclusion_rules: list[str]
    missingness_plan: str
    analysis_plan: str
    multiplicity_plan: str
    contamination_plan: str
    stopping_rule: str
    planned_outputs: list[str]
    unresolved_critical_issues: list[str] = Field(
        default_factory=list
    )

    @field_validator(
        'study_id',
        'title',
        'research_question',
        'primary_estimand',
        'sampling_plan',
        'power_and_precision_plan',
        'missingness_plan',
        'analysis_plan',
        'multiplicity_plan',
        'contamination_plan',
        'stopping_rule',
    )
    @classmethod
    def no_placeholders(cls, v: str) -> str:
        return clean(v)

    @model_validator(mode='after')
    def sealable(self):
        if not self.confirmatory_hypotheses:
            raise ValueError(
                'Confirmatory hypothesis required'
            )
        if self.unresolved_critical_issues:
            raise ValueError(
                'Critical issues prevent sealing'
            )
        return self

class ExperimentTransformations(BaseModel):
    baseline: str
    guarded: str


class ExperimentPlan(BaseModel):
    study_id: str
    adapter_family: str

    # Machine-readable execution contract consumed by registered adapters.
    execution_mode: str
    design: str
    conditions: list[str]
    task_families: list[str]
    transformations: ExperimentTransformations

    result_schema_id: str
    result_schema_version: str

    model_provider: str
    model_name: str
    model_version: str

    deterministic_automated_scoring: bool
    requires_human_scientific_labour: bool

    task_count: int = Field(gt=0)
    task_indices: list[int]

    estimated_model_calls: int = Field(gt=0)
    maximum_model_calls: int = Field(gt=0)

    reasoning_effort: str
    maximum_attempts_per_call: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)

    # Human-readable scientific and provenance description.
    implementation_strategy: str
    public_resources: list[str]
    model_and_version_plan: list[str]
    task_manifest_strategy: str
    transformation_manifest_strategy: str
    execution_batches: list[str]
    randomisation_plan: str
    caching_plan: str
    failure_recovery_plan: str
    result_schema: str
    estimated_compute_notes: str

class AnalysisPlan(BaseModel):
    study_id: str
    analysis_executor: str
    estimand: str
    primary_analysis: str
    secondary_analyses: list[str]
    sensitivity_analyses: list[str]
    uncertainty_quantification: str
    multiplicity_implementation: str
    contamination_analysis: str
    failed_call_treatment: str
    table_specifications: list[str]
    figure_specifications: list[str]

class ManuscriptSections(BaseModel):
    introduction: str
    related_work: str
    methodology: str
    results: str
    discussion: str
    conclusion: str


class ManuscriptPackage(BaseModel):
    title: str
    abstract: str
    sections: ManuscriptSections
    figure_captions: list[str]
    table_captions: list[str]
    cited_record_ids: list[str]
    disclosure_statement: str
    limitations: list[str]

class PeerReviewReport(BaseModel):
    summary:str; novelty_score:int=Field(ge=1,le=5)
    technical_depth_score:int=Field(ge=1,le=5); soundness_score:int=Field(ge=1,le=5)
    clarity_score:int=Field(ge=1,le=5); critical_issues:list[str]
    required_revisions:list[str]; optional_revisions:list[str]; accept_for_finalisation:bool

class FinalReadinessReport(BaseModel):
    ready:bool; passed_gates:list[str]; failed_gates:list[str]
    warnings:list[str]; final_state:str
