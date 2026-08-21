from pathlib import Path

from cnsm_agentic.autonomous_research.hosted_netops_adapter import (
    HostedNetOpsGVRAdapter,
)


def test_hosted_adapter_has_no_hardcoded_pilot_execution_mode():
    source = Path(
        "src/cnsm_agentic/autonomous_research/"
        "hosted_netops_adapter.py"
    ).read_text()

    assert (
        '"execution_mode": "scientific_pilot"'
        not in source
    )

    assert (
        '"execution_mode": str(plan["execution_mode"])'
        in source
    )


def test_hosted_adapter_no_longer_has_hidden_task_cap():
    assert (
        HostedNetOpsGVRAdapter.maximum_task_count
        is None
    )
