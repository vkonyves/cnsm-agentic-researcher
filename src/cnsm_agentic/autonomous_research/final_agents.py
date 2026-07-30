from agents import Agent
from .final_schemas import PreregistrationDocument,ExperimentPlan,AnalysisPlan,ManuscriptPackage,PeerReviewReport,FinalReadinessReport
PREREGISTRATION_AGENT=Agent(name='Autonomous Preregistration Author',model='gpt-5-mini',output_type=PreregistrationDocument,instructions='Create a complete sealable preregistration from the autonomously selected and repaired study. Do not leave critical issues.')
EXPERIMENT_PLANNER=Agent(name='Autonomous Experiment Planner',model='gpt-5-mini',output_type=ExperimentPlan,instructions='Create an executable plan and select an installed adapter family. Do not change the preregistered science.')
ANALYSIS_PLANNER=Agent(name='Autonomous Analysis Planner',model='gpt-5-mini',output_type=AnalysisPlan,instructions='Create the preregistration-preserving analysis implementation plan.')
MANUSCRIPT_AUTHOR=Agent(name='Autonomous Manuscript Author',model='gpt-5-mini',output_type=ManuscriptPackage,instructions='Write only from verified evidence and completed results. Never invent data or references.')
PEER_REVIEWER=Agent(name='Autonomous AI Peer Reviewer',model='gpt-5-mini',output_type=PeerReviewReport,instructions='Review novelty, depth, soundness and clarity; reject unsupported claims.')
MANUSCRIPT_REVISER=Agent(name='Autonomous Manuscript Reviser',model='gpt-5-mini',output_type=ManuscriptPackage,instructions='Revise using review while preserving evidence, preregistration and real results.')
FINAL_JUDGE=Agent(name='Autonomous Final Readiness Judge',model='gpt-5-mini',output_type=FinalReadinessReport,instructions='Require completed execution, analysis, references, review, revision, IEEE source, disclosure and PDF checks.')
