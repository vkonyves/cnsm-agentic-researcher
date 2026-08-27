from types import SimpleNamespace

from cnsm_agentic.autonomous_research.final_pipeline import (
    canonicalize_preregistration_analysis_contract,
)


class FakeExecutionContract:
    def __init__(self, adapter_family: str):
        self.adapter_family = adapter_family


class FakePreregistration:
    def __init__(
        self,
        *,
        adapter_family: str,
        primary_estimand_id: str,
        primary_estimand: str,
    ):
        self.execution_contract = FakeExecutionContract(
            adapter_family
        )
        self.primary_estimand_id = primary_estimand_id
        self.primary_estimand = primary_estimand


def test_unique_compatible_analysis_contract_canonicalizes_id():
    prereg = FakePreregistration(
        adapter_family="hosted_netops_gvr_v1",
        primary_estimand_id="paraphrased_wrong_id",
        primary_estimand=(
            "Guarded minus baseline paired success-rate difference"
        ),
    )

    contracts = {
        "paired_binary_analysis_v1": {
            "analysis_executor": (
                "paired_binary_analysis_v1"
            ),
            "estimand": (
                "paired_success_rate_difference_"
                "guarded_minus_baseline"
            ),
            "compatible_execution_adapter_families": [
                "hosted_netops_gvr_v1"
            ],
        }
    }

    result = (
        canonicalize_preregistration_analysis_contract(
            prereg,
            analysis_contracts=contracts,
        )
    )

    assert result.primary_estimand_id == (
        "paired_success_rate_difference_"
        "guarded_minus_baseline"
    )

    # Scientific prose is untouched.
    assert result.primary_estimand == (
        "Guarded minus baseline paired success-rate difference"
    )


def test_multiple_compatible_contracts_do_not_choose_for_agent():
    prereg = FakePreregistration(
        adapter_family="hosted_netops_gvr_v1",
        primary_estimand_id="agent_selected_id",
        primary_estimand="Scientific estimand prose",
    )

    contracts = {
        "analysis_a": {
            "estimand": "estimand_a",
            "compatible_execution_adapter_families": [
                "hosted_netops_gvr_v1"
            ],
        },
        "analysis_b": {
            "estimand": "estimand_b",
            "compatible_execution_adapter_families": [
                "hosted_netops_gvr_v1"
            ],
        },
    }

    result = (
        canonicalize_preregistration_analysis_contract(
            prereg,
            analysis_contracts=contracts,
        )
    )

    assert result.primary_estimand_id == (
        "agent_selected_id"
    )


def test_no_compatible_contract_leaves_id_unchanged():
    prereg = FakePreregistration(
        adapter_family="hosted_netops_gvr_v1",
        primary_estimand_id="agent_selected_id",
        primary_estimand="Scientific estimand prose",
    )

    contracts = {
        "other_analysis": {
            "estimand": "other_estimand",
            "compatible_execution_adapter_families": [
                "some_other_adapter"
            ],
        }
    }

    result = (
        canonicalize_preregistration_analysis_contract(
            prereg,
            analysis_contracts=contracts,
        )
    )

    assert result.primary_estimand_id == (
        "agent_selected_id"
    )


def test_empty_estimand_identifier_is_not_canonicalized():
    prereg = FakePreregistration(
        adapter_family="hosted_netops_gvr_v1",
        primary_estimand_id="agent_selected_id",
        primary_estimand="Scientific estimand prose",
    )

    contracts = {
        "analysis_a": {
            "estimand": "",
            "compatible_execution_adapter_families": [
                "hosted_netops_gvr_v1"
            ],
        }
    }

    result = (
        canonicalize_preregistration_analysis_contract(
            prereg,
            analysis_contracts=contracts,
        )
    )

    assert result.primary_estimand_id == (
        "agent_selected_id"
    )
