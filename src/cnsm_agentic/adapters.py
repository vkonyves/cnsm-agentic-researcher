from __future__ import annotations

from abc import ABC, abstractmethod

from cnsm_agentic.config import PilotConfig


class ExperimentAdapter(ABC):
    @abstractmethod
    def planning_guidance(self, config: PilotConfig) -> str:
        raise NotImplementedError


class LLMBenchmarkAdapter(ExperimentAdapter):
    def planning_guidance(self, config: PilotConfig) -> str:
        return """
Design candidate experiments for an LLM-for-NetOps benchmark. Prefer CPU-light
evaluation through APIs. Consider direct prompting, structured prompting,
self-consistency, retrieval, tool/verifier guidance, calibration, abstention,
cost and latency. Do not assume any named dataset is actually downloadable:
verification is a required later action. Avoid claiming novelty at this stage.
"""


class TabularMLAdapter(ExperimentAdapter):
    def planning_guidance(self, config: PilotConfig) -> str:
        return """
Design candidate experiments for a CPU-only tabular network-management dataset.
Consider classification, regression, anomaly detection, leakage-safe splits,
class imbalance, interpretable baselines, calibration, uncertainty, and
operationally meaningful error costs. Do not assume any named dataset is
accessible or licensed: verification is a required later action.
"""


def get_adapter(name: str) -> ExperimentAdapter:
    if name == "llm_benchmark":
        return LLMBenchmarkAdapter()
    if name == "tabular_ml":
        return TabularMLAdapter()
    raise ValueError(f"Unsupported experiment family: {name}")
