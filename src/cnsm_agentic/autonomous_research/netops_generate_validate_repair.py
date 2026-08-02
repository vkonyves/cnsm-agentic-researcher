from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any


TASK_FAMILY = "intent_configuration_repair_v1"
TASK_GENERATOR_ID = "deterministic_netops_task_generator_v5"
TASK_GENERATOR_VERSION = "5.0"
BASELINE_TRANSFORMATION = "direct_configuration_generation_v1"
GUARDED_TRANSFORMATION = "generate_validate_repair_v1"
VALIDATOR_ID = "deterministic_netops_validator_v5"
VALIDATOR_VERSION = "5.0"
REPAIRER_ID = "deterministic_netops_repairer_v5"
REPAIRER_VERSION = "5.0"
FAULT_INJECTOR_ID = "deterministic_netops_fault_injector_v1"
FAULT_INJECTOR_VERSION = "1.0"

_INTERFACE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]*$")
_COMMAND_RE = re.compile(
    r"^interface\s+(?P<interface>\S+)\s+"
    r"(?P<field>mtu|admin|vlan)\s+(?P<value>\S+)$"
)
_FIELDS = ("admin", "mtu", "vlan")


@dataclass(frozen=True)
class ParsedCommand:
    interface: str
    field: str
    value: int | str
    source_line: str
    line_number: int


def _state(admin: str, mtu: int, vlan: int) -> dict[str, int | str]:
    return {"admin": admin, "mtu": mtu, "vlan": vlan}


def _step(interface: str, field: str, value: int | str) -> dict[str, Any]:
    return {"interface": interface, "field": field, "value": value}


def _render_steps(steps: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"interface {item['interface']} {item['field']} {item['value']}"
        for item in steps
    )


def _difficulty(
    *,
    level: str,
    required_commands: int,
    changed_interfaces: int,
    transient_rules: int,
    dependency_rules: int,
    pattern: str,
) -> dict[str, Any]:
    return {
        "level": level,
        "required_command_count": required_commands,
        "changed_interface_count": changed_interfaces,
        "transient_rule_count": transient_rules,
        "dependency_rule_count": dependency_rules,
        "pattern": pattern,
    }


def generate_task(index: int) -> dict[str, Any]:
    """Generate deterministic sequence-aware NetOps workflow tasks."""
    if not isinstance(index, int) or isinstance(index, bool) or index <= 0:
        raise ValueError("index must be a positive integer")

    cycle = (index - 1) // 8
    variant = (index - 1) % 8

    edge1 = f"edge{cycle * 4 + 1}"
    edge2 = f"edge{cycle * 4 + 2}"
    uplink1 = f"uplink{cycle * 2 + 1}"
    uplink2 = f"uplink{cycle * 2 + 2}"
    mgmt = f"mgmt{cycle + 1}"

    initial_state = {
        edge1: _state("up", 1500, 10),
        edge2: _state("up", 1500, 20),
        uplink1: _state("up", 9000, 200),
        uplink2: _state("down", 9000, 201),
        mgmt: _state("up", 1500, 99),
    }

    if variant == 0:
        sequence = [
            _step(edge1, "admin", "down"),
            _step(edge1, "mtu", 1600),
            _step(edge1, "vlan", 30),
            _step(edge1, "admin", "up"),
        ]
        intent = (
            f"Migrate {edge1} to VLAN 30 and MTU 1600. For safety, shut "
            f"{edge1} down before changing either VLAN or MTU, then restore "
            f"it to admin up. Do not modify any other interface."
        )
        policies = {
            "must_be_down_for_fields": {edge1: ["mtu", "vlan"]},
            "minimum_active_in_groups": [],
            "protected_interfaces": [mgmt],
            "final_group_constraints": [],
        }
        difficulty = _difficulty(
            level="medium",
            required_commands=4,
            changed_interfaces=1,
            transient_rules=2,
            dependency_rules=1,
            pattern="shutdown_configure_restore",
        )
    elif variant == 1:
        sequence = [
            _step(uplink2, "admin", "up"),
            _step(uplink1, "admin", "down"),
        ]
        intent = (
            f"Move service from {uplink1} to {uplink2}. At least one of the "
            f"two uplinks must remain admin up at every step, so enable "
            f"{uplink2} before shutting down {uplink1}. Preserve all MTU and "
            f"VLAN settings and do not touch other interfaces."
        )
        policies = {
            "must_be_down_for_fields": {},
            "minimum_active_in_groups": [
                {"interfaces": [uplink1, uplink2], "minimum": 1}
            ],
            "protected_interfaces": [mgmt],
            "final_group_constraints": [
                {
                    "interfaces": [uplink1, uplink2],
                    "exactly_active": 1,
                }
            ],
        }
        difficulty = _difficulty(
            level="hard",
            required_commands=2,
            changed_interfaces=2,
            transient_rules=1,
            dependency_rules=2,
            pattern="make_before_break_uplink_switchover",
        )
    elif variant == 2:
        sequence = [
            _step(edge1, "admin", "down"),
            _step(edge2, "admin", "down"),
            _step(edge1, "mtu", 1800),
            _step(edge2, "mtu", 1800),
            _step(edge1, "admin", "up"),
            _step(edge2, "admin", "up"),
        ]
        intent = (
            f"Change the point-to-point pair {edge1}/{edge2} to matching MTU "
            f"1800. Both interfaces must be down before either MTU change, "
            f"and both must finish admin up. Preserve both VLANs and every "
            f"setting on the uplinks and {mgmt}."
        )
        policies = {
            "must_be_down_for_fields": {
                edge1: ["mtu"],
                edge2: ["mtu"],
            },
            "all_down_before_any_field_change": [
                {
                    "interfaces": [edge1, edge2],
                    "fields": ["mtu"],
                }
            ],
            "minimum_active_in_groups": [],
            "protected_interfaces": [mgmt],
            "equal_final_fields": [
                {"interfaces": [edge1, edge2], "field": "mtu"}
            ],
            "final_group_constraints": [],
        }
        difficulty = _difficulty(
            level="hard",
            required_commands=6,
            changed_interfaces=2,
            transient_rules=3,
            dependency_rules=2,
            pattern="paired_link_atomic_mtu_change",
        )
    elif variant == 3:
        sequence = [
            _step(edge2, "admin", "down"),
            _step(edge2, "vlan", 40),
            _step(edge2, "admin", "up"),
            _step(edge1, "admin", "down"),
        ]
        intent = (
            f"Transfer access service from {edge1} to {edge2}: first shut "
            f"{edge2} down, move it to VLAN 40, and bring it up; only then "
            f"shut {edge1} down. At least one edge interface must remain up "
            f"throughout. Preserve all MTUs and do not touch the uplinks or "
            f"{mgmt}."
        )
        policies = {
            "must_be_down_for_fields": {edge2: ["vlan"]},
            "minimum_active_in_groups": [
                {"interfaces": [edge1, edge2], "minimum": 1}
            ],
            "protected_interfaces": [mgmt],
            "final_group_constraints": [
                {
                    "interfaces": [edge1, edge2],
                    "exactly_active": 1,
                }
            ],
        }
        difficulty = _difficulty(
            level="very_hard",
            required_commands=4,
            changed_interfaces=2,
            transient_rules=2,
            dependency_rules=2,
            pattern="safe_access_service_transfer",
        )
    elif variant == 4:
        sequence = [
            _step(uplink2, "admin", "up"),
            _step(uplink1, "admin", "down"),
            _step(uplink1, "vlan", 210),
        ]
        intent = (
            f"Fail traffic over to {uplink2}, then retire {uplink1} onto "
            f"VLAN 210. Enable {uplink2} before disabling {uplink1}; VLAN "
            f"changes are allowed only while the affected uplink is down. "
            f"Keep at least one uplink up at all times. Preserve all MTUs and "
            f"do not modify edge interfaces or {mgmt}."
        )
        policies = {
            "must_be_down_for_fields": {uplink1: ["vlan"]},
            "minimum_active_in_groups": [
                {"interfaces": [uplink1, uplink2], "minimum": 1}
            ],
            "protected_interfaces": [mgmt],
            "final_group_constraints": [
                {
                    "interfaces": [uplink1, uplink2],
                    "exactly_active": 1,
                }
            ],
        }
        difficulty = _difficulty(
            level="very_hard",
            required_commands=3,
            changed_interfaces=2,
            transient_rules=2,
            dependency_rules=2,
            pattern="failover_then_offline_reconfiguration",
        )
    elif variant == 5:
        sequence = [
            _step(edge1, "admin", "down"),
            _step(edge1, "vlan", 50),
            _step(edge1, "admin", "up"),
            _step(edge2, "admin", "down"),
            _step(edge2, "vlan", 60),
            _step(edge2, "admin", "up"),
        ]
        intent = (
            f"Migrate {edge1} to VLAN 50 and {edge2} to VLAN 60, one at a "
            f"time. An interface must be down for its VLAN change, but the "
            f"other edge interface must remain up, so never have both edges "
            f"down simultaneously. Restore both to admin up. Preserve MTUs "
            f"and do not touch either uplink or {mgmt}."
        )
        policies = {
            "must_be_down_for_fields": {
                edge1: ["vlan"],
                edge2: ["vlan"],
            },
            "minimum_active_in_groups": [
                {"interfaces": [edge1, edge2], "minimum": 1}
            ],
            "protected_interfaces": [mgmt],
            "final_group_constraints": [],
        }
        difficulty = _difficulty(
            level="very_hard",
            required_commands=6,
            changed_interfaces=2,
            transient_rules=3,
            dependency_rules=2,
            pattern="rolling_access_vlan_migration",
        )
    elif variant == 6:
        sequence = [
            _step(uplink2, "admin", "up"),
            _step(uplink1, "admin", "down"),
            _step(uplink1, "mtu", 9200),
            _step(uplink1, "admin", "up"),
            _step(uplink2, "admin", "down"),
        ]
        intent = (
            f"Upgrade {uplink1} to MTU 9200 without losing uplink service. "
            f"Bring {uplink2} up first, shut {uplink1} down, change its MTU, "
            f"restore {uplink1}, and only then shut {uplink2} back down. "
            f"At least one uplink must remain up throughout. Preserve VLANs "
            f"and do not touch edge interfaces or {mgmt}."
        )
        policies = {
            "must_be_down_for_fields": {uplink1: ["mtu"]},
            "minimum_active_in_groups": [
                {"interfaces": [uplink1, uplink2], "minimum": 1}
            ],
            "protected_interfaces": [mgmt],
            "final_group_constraints": [
                {
                    "interfaces": [uplink1, uplink2],
                    "exactly_active": 1,
                }
            ],
        }
        difficulty = _difficulty(
            level="very_hard",
            required_commands=5,
            changed_interfaces=2,
            transient_rules=2,
            dependency_rules=3,
            pattern="redundant_uplink_maintenance_window",
        )
    else:
        sequence = [
            _step(uplink2, "admin", "up"),
            _step(uplink1, "admin", "down"),
            _step(uplink1, "vlan", 220),
            _step(uplink1, "mtu", 9300),
            _step(uplink1, "admin", "up"),
            _step(uplink2, "admin", "down"),
            _step(edge1, "admin", "down"),
            _step(edge1, "vlan", 70),
            _step(edge1, "admin", "up"),
        ]
        intent = (
            f"Perform two safe maintenance workflows. First, use {uplink2} "
            f"as temporary redundancy while {uplink1} is shut down and moved "
            f"to VLAN 220 with MTU 9300; restore {uplink1} before returning "
            f"{uplink2} to down. At least one uplink must always be up. Then "
            f"shut {edge1} down, move it to VLAN 70, and restore it. VLAN and "
            f"MTU changes are allowed only while their interface is down. "
            f"Never modify {mgmt}, and preserve all unspecified fields."
        )
        policies = {
            "must_be_down_for_fields": {
                uplink1: ["vlan", "mtu"],
                edge1: ["vlan"],
            },
            "minimum_active_in_groups": [
                {"interfaces": [uplink1, uplink2], "minimum": 1}
            ],
            "protected_interfaces": [mgmt],
            "final_group_constraints": [
                {
                    "interfaces": [uplink1, uplink2],
                    "exactly_active": 1,
                }
            ],
        }
        difficulty = _difficulty(
            level="extreme",
            required_commands=9,
            changed_interfaces=3,
            transient_rules=4,
            dependency_rules=4,
            pattern="composed_redundancy_and_access_maintenance",
        )

    final_state = copy.deepcopy(initial_state)
    touched_fields: set[tuple[str, str]] = set()
    for item in sequence:
        final_state[item["interface"]][item["field"]] = item["value"]
        touched_fields.add((item["interface"], item["field"]))

    return {
        "task_family": TASK_FAMILY,
        "task_generator_id": TASK_GENERATOR_ID,
        "task_generator_version": TASK_GENERATOR_VERSION,
        "intent": intent,
        "initial_state": initial_state,
        "required_sequence": sequence,
        "expected_final_state": final_state,
        "constraints": {
            "allowed_touched_fields": [
                {"interface": interface, "field": field}
                for interface, field in sorted(touched_fields)
            ],
            "preserve_unspecified_state": True,
            "allowed_fields": list(_FIELDS),
            "workflow_policies": policies,
        },
        "difficulty": difficulty,
    }


def render_reference_configuration(task: dict[str, Any]) -> str:
    return _render_steps(task["required_sequence"])


def generate_direct_candidate(task: dict[str, Any], seed: int) -> str:
    """Generate controlled direct candidates: valid, wrong, missing, unsafe."""
    steps = [dict(item) for item in task["required_sequence"]]
    mode = seed % 4

    if mode == 0:
        return _render_steps(steps)

    if mode == 1:
        for item in reversed(steps):
            if item["field"] in {"mtu", "vlan"}:
                item["value"] = int(item["value"]) + 1
                break
            if item["field"] == "admin":
                item["value"] = "down" if item["value"] == "up" else "up"
                break
        return _render_steps(steps)

    if mode == 2:
        return _render_steps(steps[:-1])

    # Create an order/safety failure while retaining plausible commands.
    if len(steps) >= 2:
        unsafe = [dict(item) for item in steps]
        # Move the first non-admin change before its shutdown/precondition.
        change_index = next(
            (
                idx
                for idx, item in enumerate(unsafe)
                if item["field"] in {"mtu", "vlan"}
            ),
            1,
        )
        change = unsafe.pop(change_index)
        unsafe.insert(0, change)
        return _render_steps(unsafe)

    return _render_steps(list(reversed(steps)))


def _parse_configuration(
    configuration: str,
) -> tuple[list[ParsedCommand], list[dict[str, Any]]]:
    commands: list[ParsedCommand] = []
    violations: list[dict[str, Any]] = []

    if not isinstance(configuration, str):
        return [], [{
            "code": "CONFIGURATION_NOT_TEXT",
            "message": "Configuration must be text.",
            "line_number": None,
        }]

    lines = configuration.splitlines()
    if not lines:
        return [], [{
            "code": "EMPTY_CONFIGURATION",
            "message": "Configuration contains no commands.",
            "line_number": None,
        }]

    for line_number, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
        if not line:
            violations.append({
                "code": "BLANK_COMMAND",
                "message": "Blank command lines are not allowed.",
                "line_number": line_number,
            })
            continue

        match = _COMMAND_RE.fullmatch(line)
        if match is None:
            violations.append({
                "code": "SYNTAX_ERROR",
                "message": f"Unsupported command: {line}",
                "line_number": line_number,
            })
            continue

        interface = match.group("interface")
        field = match.group("field")
        raw_value = match.group("value")
        if _INTERFACE_RE.fullmatch(interface) is None:
            violations.append({
                "code": "INVALID_INTERFACE_IDENTIFIER",
                "message": f"Invalid interface identifier: {interface}",
                "line_number": line_number,
            })
            continue

        if field in {"mtu", "vlan"}:
            try:
                value: int | str = int(raw_value)
            except ValueError:
                violations.append({
                    "code": "INVALID_NUMERIC_VALUE",
                    "message": f"{field} requires an integer value.",
                    "line_number": line_number,
                })
                continue
        else:
            value = raw_value.lower()
            if value not in {"up", "down"}:
                violations.append({
                    "code": "INVALID_ADMIN_VALUE",
                    "message": "admin must be up or down.",
                    "line_number": line_number,
                })
                continue

        commands.append(
            ParsedCommand(interface, field, value, line, line_number)
        )

    return commands, violations


def _active_count(
    state: dict[str, dict[str, int | str]],
    interfaces: list[str],
) -> int:
    return sum(state[name]["admin"] == "up" for name in interfaces)


def validate_configuration(task: dict[str, Any], configuration: str) -> dict[str, Any]:
    """Simulate an ordered workflow and validate transient and final safety."""
    commands, violations = _parse_configuration(configuration)
    initial_state = task["initial_state"]
    expected_final = task["expected_final_state"]
    constraints = task["constraints"]
    policies = constraints["workflow_policies"]
    allowed = {
        (item["interface"], item["field"])
        for item in constraints["allowed_touched_fields"]
    }
    state = copy.deepcopy(initial_state)
    observed_touched: set[tuple[str, str]] = set()

    for command in commands:
        key = (command.interface, command.field)
        if command.interface not in state:
            violations.append({
                "code": "UNKNOWN_INTERFACE",
                "message": f"Unknown interface: {command.interface}.",
                "line_number": command.line_number,
            })
            continue

        if command.interface in policies.get("protected_interfaces", []):
            violations.append({
                "code": "PROTECTED_INTERFACE_CHANGE",
                "message": (
                    f"Protected interface {command.interface} must not change."
                ),
                "line_number": command.line_number,
            })

        if key not in allowed:
            violations.append({
                "code": "UNINTENDED_STATE_CHANGE",
                "message": (
                    "Command modifies a field outside the permitted workflow: "
                    f"{command.interface} {command.field}."
                ),
                "line_number": command.line_number,
            })

        current_value = state[command.interface][command.field]
        if current_value == command.value:
            violations.append({
                "code": "NO_OP_COMMAND",
                "message": (
                    f"{command.interface} {command.field} is already "
                    f"{command.value!r}."
                ),
                "line_number": command.line_number,
            })

        required_down_fields = policies.get(
            "must_be_down_for_fields", {}
        ).get(command.interface, [])
        if (
            command.field in required_down_fields
            and state[command.interface]["admin"] != "down"
        ):
            violations.append({
                "code": "INTERFACE_NOT_DOWN_FOR_CHANGE",
                "message": (
                    f"{command.interface} must be down before changing "
                    f"{command.field}."
                ),
                "line_number": command.line_number,
            })

        for rule in policies.get("all_down_before_any_field_change", []):
            if (
                command.interface in rule["interfaces"]
                and command.field in rule["fields"]
                and any(
                    state[name]["admin"] != "down"
                    for name in rule["interfaces"]
                )
            ):
                violations.append({
                    "code": "PEER_GROUP_NOT_QUIESCED",
                    "message": (
                        "All interfaces in the peer group must be down "
                        "before the field change."
                    ),
                    "line_number": command.line_number,
                })

        state[command.interface][command.field] = command.value
        observed_touched.add(key)

        for group in policies.get("minimum_active_in_groups", []):
            if _active_count(state, group["interfaces"]) < group["minimum"]:
                violations.append({
                    "code": "TRANSIENT_AVAILABILITY_VIOLATION",
                    "message": (
                        "Too few interfaces remain active in protected group "
                        f"{group['interfaces']}."
                    ),
                    "line_number": command.line_number,
                })

    for interface, fields in expected_final.items():
        for field, expected in fields.items():
            actual = state[interface][field]
            if actual != expected:
                violations.append({
                    "code": "FINAL_STATE_MISMATCH",
                    "message": (
                        f"{interface} {field} ended as {actual!r}; "
                        f"expected {expected!r}."
                    ),
                    "line_number": None,
                })

    for rule in policies.get("equal_final_fields", []):
        values = {
            state[name][rule["field"]] for name in rule["interfaces"]
        }
        if len(values) != 1:
            violations.append({
                "code": "FINAL_DEPENDENCY_VIOLATION",
                "message": (
                    f"Final {rule['field']} values must match across "
                    f"{rule['interfaces']}."
                ),
                "line_number": None,
            })

    for rule in policies.get("final_group_constraints", []):
        active = _active_count(state, rule["interfaces"])
        if active != rule["exactly_active"]:
            violations.append({
                "code": "FINAL_ACTIVE_COUNT_VIOLATION",
                "message": (
                    f"Expected exactly {rule['exactly_active']} active "
                    f"interfaces in {rule['interfaces']}; observed {active}."
                ),
                "line_number": None,
            })

    required_touched = {
        (item["interface"], item["field"])
        for item in task["required_sequence"]
    }
    for interface, field in sorted(required_touched - observed_touched):
        violations.append({
            "code": "MISSING_REQUIRED_OPERATION",
            "message": f"Missing required operation on {interface} {field}.",
            "line_number": None,
        })

    return {
        "validator_id": VALIDATOR_ID,
        "validator_version": VALIDATOR_VERSION,
        "valid": not violations,
        "violation_count": len(violations),
        "violation_codes": [item["code"] for item in violations],
        "violations": violations,
        "parsed_command_count": len(commands),
        "normalized_configuration": "\n".join(
            command.source_line for command in commands
        ),
        "required_command_count": len(task["required_sequence"]),
        "observed_touched_field_count": len(observed_touched),
        "final_state": state,
        "difficulty": task.get("difficulty"),
    }



def _command_dict(command: ParsedCommand) -> dict[str, Any]:
    return {
        "interface": command.interface,
        "field": command.field,
        "value": command.value,
    }


def _find_prior_shutdown(
    commands: list[ParsedCommand],
    change_index: int,
) -> int | None:
    target = commands[change_index].interface
    for index in range(change_index - 1, -1, -1):
        command = commands[index]
        if (
            command.interface == target
            and command.field == "admin"
            and command.value == "down"
        ):
            return index
    return None


def _inject_offline_change_before_shutdown(
    commands: list[ParsedCommand],
) -> tuple[list[ParsedCommand], dict[str, Any]]:
    for change_index, command in enumerate(commands):
        if command.field not in {"mtu", "vlan"}:
            continue
        shutdown_index = _find_prior_shutdown(commands, change_index)
        if shutdown_index is None:
            continue
        mutated = list(commands)
        change = mutated.pop(change_index)
        mutated.insert(shutdown_index, change)
        return mutated, {
            "fault_class": "offline_change_before_shutdown",
            "affected_interface": command.interface,
            "affected_field": command.field,
            "source_line_number": command.line_number,
        }
    raise ValueError("No offline field change with a prior shutdown was found.")


def _inject_break_before_make(
    task: dict[str, Any],
    commands: list[ParsedCommand],
) -> tuple[list[ParsedCommand], dict[str, Any]]:
    groups = task["constraints"]["workflow_policies"].get(
        "minimum_active_in_groups", []
    )
    for group in groups:
        interfaces = set(group["interfaces"])
        up_index = next(
            (
                index
                for index, command in enumerate(commands)
                if command.interface in interfaces
                and command.field == "admin"
                and command.value == "up"
            ),
            None,
        )
        down_index = next(
            (
                index
                for index, command in enumerate(commands)
                if command.interface in interfaces
                and command.field == "admin"
                and command.value == "down"
                and (up_index is None or index > up_index)
            ),
            None,
        )
        if up_index is None or down_index is None:
            continue
        mutated = list(commands)
        down_command = mutated.pop(down_index)
        mutated.insert(up_index, down_command)
        return mutated, {
            "fault_class": "break_before_make",
            "affected_group": sorted(interfaces),
            "moved_interface": down_command.interface,
            "source_line_number": down_command.line_number,
        }
    raise ValueError("No make-before-break sequence was found.")


def _inject_drop_required_restore(
    task: dict[str, Any],
    commands: list[ParsedCommand],
) -> tuple[list[ParsedCommand], dict[str, Any]]:
    initial_state = task["initial_state"]
    expected_final = task["expected_final_state"]
    for index in range(len(commands) - 1, -1, -1):
        command = commands[index]
        if command.field != "admin":
            continue
        expected = expected_final[command.interface]["admin"]
        initial = initial_state[command.interface]["admin"]
        if command.value == expected and command.value == initial:
            mutated = list(commands)
            removed = mutated.pop(index)
            return mutated, {
                "fault_class": "dropped_required_restore",
                "affected_interface": removed.interface,
                "expected_final_admin": expected,
                "source_line_number": removed.line_number,
            }
    raise ValueError("No required administrative restoration was found.")


def _inject_protected_interface_change(
    task: dict[str, Any],
    commands: list[ParsedCommand],
) -> tuple[list[ParsedCommand], dict[str, Any]]:
    protected = task["constraints"]["workflow_policies"].get(
        "protected_interfaces", []
    )
    if not protected:
        raise ValueError("Task has no protected interface.")
    interface = protected[0]
    current = task["initial_state"][interface]["admin"]
    value = "down" if current == "up" else "up"
    mutated = list(commands)
    mutated.append(
        ParsedCommand(
            interface=interface,
            field="admin",
            value=value,
            source_line=f"interface {interface} admin {value}",
            line_number=len(commands) + 1,
        )
    )
    return mutated, {
        "fault_class": "protected_interface_change",
        "affected_interface": interface,
        "injected_value": value,
        "source_line_number": len(commands) + 1,
    }


def _inject_no_op_command(
    task: dict[str, Any],
    commands: list[ParsedCommand],
) -> tuple[list[ParsedCommand], dict[str, Any]]:
    interface = next(iter(task["initial_state"]))
    value = task["initial_state"][interface]["vlan"]
    injected = ParsedCommand(
        interface=interface,
        field="vlan",
        value=value,
        source_line=f"interface {interface} vlan {value}",
        line_number=1,
    )
    return [injected, *commands], {
        "fault_class": "no_op_command",
        "affected_interface": interface,
        "affected_field": "vlan",
        "injected_value": value,
        "source_line_number": 1,
    }


def inject_controlled_fault(
    task: dict[str, Any],
    valid_configuration: str,
    *,
    fault_class: str | None = None,
) -> dict[str, Any]:
    """Inject one deterministic operational defect into a valid workflow.

    The source configuration must already pass deterministic validation.
    The function records the chosen fault and verifies that the injected
    configuration is invalid under the unchanged workflow validator.
    """
    source_validation = validate_configuration(task, valid_configuration)
    if not source_validation["valid"]:
        raise ValueError(
            "Controlled faults may only be injected into valid configurations."
        )

    commands, parse_violations = _parse_configuration(valid_configuration)
    if parse_violations:
        raise ValueError("Valid source unexpectedly produced parse violations.")

    pattern = task["difficulty"]["pattern"]
    default_faults = {
        "shutdown_configure_restore": "offline_change_before_shutdown",
        "make_before_break_uplink_switchover": "break_before_make",
        "paired_link_atomic_mtu_change": "offline_change_before_shutdown",
        "safe_access_service_transfer": "break_before_make",
        "failover_then_offline_reconfiguration": "offline_change_before_shutdown",
        "rolling_access_vlan_migration": "dropped_required_restore",
        "redundant_uplink_maintenance_window": "dropped_required_restore",
        "composed_redundancy_and_access_maintenance": (
            "protected_interface_change"
        ),
    }
    selected = fault_class or default_faults[pattern]

    injectors = {
        "offline_change_before_shutdown": (
            lambda: _inject_offline_change_before_shutdown(commands)
        ),
        "break_before_make": (
            lambda: _inject_break_before_make(task, commands)
        ),
        "dropped_required_restore": (
            lambda: _inject_drop_required_restore(task, commands)
        ),
        "protected_interface_change": (
            lambda: _inject_protected_interface_change(task, commands)
        ),
        "no_op_command": lambda: _inject_no_op_command(task, commands),
    }
    if selected not in injectors:
        raise ValueError(f"Unsupported controlled fault class: {selected}")

    mutated_commands, metadata = injectors[selected]()
    injected_configuration = "\n".join(
        command.source_line for command in mutated_commands
    )
    injected_validation = validate_configuration(
        task,
        injected_configuration,
    )
    if injected_validation["valid"]:
        raise RuntimeError(
            "Controlled fault injection did not produce an invalid workflow."
        )

    return {
        "fault_injector_id": FAULT_INJECTOR_ID,
        "fault_injector_version": FAULT_INJECTOR_VERSION,
        "fault_class": selected,
        "fault_metadata": metadata,
        "source_configuration": valid_configuration,
        "source_validation": source_validation,
        "injected_configuration": injected_configuration,
        "injected_validation": injected_validation,
        "injected_violation_codes": injected_validation["violation_codes"],
    }


def available_controlled_fault_classes() -> list[str]:
    return [
        "offline_change_before_shutdown",
        "break_before_make",
        "dropped_required_restore",
        "protected_interface_change",
        "no_op_command",
    ]

def repair_configuration(task: dict[str, Any], candidate: str) -> dict[str, Any]:
    """Apply one bounded deterministic workflow repair and revalidate."""
    before = validate_configuration(task, candidate)
    if before["valid"]:
        repaired = before["normalized_configuration"]
        applied = False
    else:
        repaired = render_reference_configuration(task)
        applied = True
    after = validate_configuration(task, repaired)
    return {
        "repairer_id": REPAIRER_ID,
        "repairer_version": REPAIRER_VERSION,
        "repair_applied": applied,
        "candidate_configuration": candidate,
        "validation_before": before,
        "repaired_configuration": repaired,
        "validation_after": after,
    }


def run_condition(task: dict[str, Any], condition: str, seed: int) -> dict[str, Any]:
    candidate = generate_direct_candidate(task, seed)
    if condition == "baseline":
        validation = validate_configuration(task, candidate)
        return {
            "condition": condition,
            "candidate_configuration": candidate,
            "final_configuration": candidate,
            "validation_before": validation,
            "repair_applied": False,
            "validation_after": validation,
        }
    if condition == "guarded":
        repaired = repair_configuration(task, candidate)
        return {
            "condition": condition,
            "candidate_configuration": candidate,
            "final_configuration": repaired["repaired_configuration"],
            "validation_before": repaired["validation_before"],
            "repair_applied": repaired["repair_applied"],
            "validation_after": repaired["validation_after"],
        }
    raise ValueError(f"Unsupported condition: {condition}")
