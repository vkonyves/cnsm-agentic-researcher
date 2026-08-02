from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


TASK_FAMILY = "intent_configuration_repair_v1"
TASK_GENERATOR_ID = "deterministic_netops_task_generator_v3"
TASK_GENERATOR_VERSION = "3.0"
BASELINE_TRANSFORMATION = "direct_configuration_generation_v1"
GUARDED_TRANSFORMATION = "generate_validate_repair_v1"
VALIDATOR_ID = "deterministic_netops_validator_v3"
VALIDATOR_VERSION = "3.0"
REPAIRER_ID = "deterministic_netops_repairer_v3"
REPAIRER_VERSION = "3.0"

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


def _state(admin: str, mtu: int, vlan: int) -> dict[str, int | str]:
    return {"admin": admin, "mtu": mtu, "vlan": vlan}


def _difficulty(
    *,
    level: str,
    required_assignments: int,
    changed_interfaces: int,
    preserved_interfaces: int,
    preservation_clauses: int,
    distractor_count: int,
    pattern: str,
) -> dict[str, Any]:
    return {
        "level": level,
        "required_assignment_count": required_assignments,
        "changed_interface_count": changed_interfaces,
        "preserved_interface_count": preserved_interfaces,
        "preservation_clause_count": preservation_clauses,
        "distractor_count": distractor_count,
        "pattern": pattern,
    }


def generate_task(index: int) -> dict[str, Any]:
    """Generate a deterministic state-aware NetOps intent task."""
    if not isinstance(index, int) or isinstance(index, bool) or index <= 0:
        raise ValueError("index must be a positive integer")

    cycle = (index - 1) // 8
    variant = (index - 1) % 8

    a = f"edge{cycle * 4 + 1}"
    b = f"edge{cycle * 4 + 2}"
    c = f"uplink{cycle + 1}"
    d = f"mgmt{cycle + 1}"

    initial_state = {
        a: _state("down", 1500, 1),
        b: _state("up", 1500, 20 + cycle),
        c: _state("up", 9000, 200 + cycle),
        d: _state("up", 1500, 99),
    }

    if variant == 0:
        required_changes = [
            {"interface": a, "field": "admin", "value": "up"},
            {"interface": a, "field": "mtu", "value": 1600 + 100 * (cycle % 2)},
            {"interface": a, "field": "vlan", "value": 10 + cycle},
        ]
        intent = (
            f"Bring {a} up, set its MTU to {required_changes[1]['value']}, "
            f"and move it to access VLAN {required_changes[2]['value']}. "
            f"Leave {b}, {c}, and {d} unchanged."
        )
        difficulty = _difficulty(
            level="medium",
            required_assignments=3,
            changed_interfaces=1,
            preserved_interfaces=3,
            preservation_clauses=1,
            distractor_count=3,
            pattern="single_interface_full_reconfiguration",
        )
    elif variant == 1:
        required_changes = [
            {"interface": a, "field": "admin", "value": "up"},
            {"interface": a, "field": "vlan", "value": 30 + cycle},
            {"interface": b, "field": "mtu", "value": 1700},
        ]
        intent = (
            f"Enable {a} and assign it to access VLAN {30 + cycle}; "
            f"change only the MTU of {b} to 1700. Preserve every other "
            f"setting, including everything on {c} and {d}."
        )
        difficulty = _difficulty(
            level="hard",
            required_assignments=3,
            changed_interfaces=2,
            preserved_interfaces=2,
            preservation_clauses=2,
            distractor_count=4,
            pattern="two_interface_partial_update",
        )
    elif variant == 2:
        required_changes = [
            {"interface": b, "field": "admin", "value": "down"},
            {"interface": b, "field": "vlan", "value": 40 + cycle},
        ]
        intent = (
            f"Administratively shut down {b} and prepare access VLAN "
            f"{40 + cycle} on it. Its MTU must remain 1500. Keep {a}, "
            f"{c}, and {d} completely unchanged."
        )
        difficulty = _difficulty(
            level="hard",
            required_assignments=2,
            changed_interfaces=1,
            preserved_interfaces=3,
            preservation_clauses=2,
            distractor_count=4,
            pattern="target_partial_update_with_explicit_preservation",
        )
    elif variant == 3:
        required_changes = [
            {"interface": a, "field": "mtu", "value": 1800},
            {"interface": b, "field": "admin", "value": "down"},
            {"interface": b, "field": "vlan", "value": 50 + cycle},
        ]
        intent = (
            f"Set {a} MTU to 1800. On {b}, shut the interface down and "
            f"change its access VLAN to {50 + cycle}. Do not alter {a}'s "
            f"admin or VLAN, {b}'s MTU, or anything on {c} or {d}."
        )
        difficulty = _difficulty(
            level="hard",
            required_assignments=3,
            changed_interfaces=2,
            preserved_interfaces=2,
            preservation_clauses=4,
            distractor_count=5,
            pattern="cross_interface_mixed_fields",
        )
    elif variant == 4:
        required_changes = [
            {"interface": a, "field": "admin", "value": "up"},
            {"interface": b, "field": "admin", "value": "down"},
            {"interface": c, "field": "mtu", "value": 9200},
        ]
        intent = (
            f"Bring {a} up, shut {b} down, and change only {c}'s MTU "
            f"from 9000 to 9200. Preserve all VLANs, preserve the MTUs of "
            f"{a}, {b}, and {d}, and leave {d} otherwise untouched."
        )
        difficulty = _difficulty(
            level="very_hard",
            required_assignments=3,
            changed_interfaces=3,
            preserved_interfaces=1,
            preservation_clauses=3,
            distractor_count=6,
            pattern="three_interface_sparse_update",
        )
    elif variant == 5:
        required_changes = [
            {"interface": a, "field": "vlan", "value": 60 + cycle},
            {"interface": c, "field": "admin", "value": "down"},
            {"interface": c, "field": "vlan", "value": 210 + cycle},
        ]
        intent = (
            f"Move {a} to access VLAN {60 + cycle} without enabling it "
            f"or changing its MTU. On {c}, shut the interface down and "
            f"set VLAN {210 + cycle}, but keep its jumbo MTU at 9000. "
            f"Leave {b} and {d} unchanged."
        )
        difficulty = _difficulty(
            level="very_hard",
            required_assignments=3,
            changed_interfaces=2,
            preserved_interfaces=2,
            preservation_clauses=4,
            distractor_count=6,
            pattern="negative_instruction_and_jumbo_preservation",
        )
    elif variant == 6:
        required_changes = [
            {"interface": a, "field": "admin", "value": "up"},
            {"interface": a, "field": "mtu", "value": 1900},
            {"interface": b, "field": "vlan", "value": 70 + cycle},
            {"interface": c, "field": "admin", "value": "down"},
        ]
        intent = (
            f"Enable {a} and set its MTU to 1900 while keeping its VLAN "
            f"at 1. Change only {b}'s VLAN to {70 + cycle}. Shut {c} "
            f"down without changing its MTU or VLAN. Do not touch {d}."
        )
        difficulty = _difficulty(
            level="very_hard",
            required_assignments=4,
            changed_interfaces=3,
            preserved_interfaces=1,
            preservation_clauses=4,
            distractor_count=7,
            pattern="three_interface_four_assignment_update",
        )
    else:
        required_changes = [
            {"interface": a, "field": "mtu", "value": 2000},
            {"interface": b, "field": "admin", "value": "down"},
            {"interface": b, "field": "mtu", "value": 1600},
            {"interface": c, "field": "vlan", "value": 220 + cycle},
        ]
        intent = (
            f"Set only {a}'s MTU to 2000. On {b}, shut it down and set "
            f"its MTU to 1600 while preserving VLAN {20 + cycle}. Change "
            f"only {c}'s VLAN to {220 + cycle}, preserving admin up and "
            f"MTU 9000. Keep every setting on {d} unchanged."
        )
        difficulty = _difficulty(
            level="very_hard",
            required_assignments=4,
            changed_interfaces=3,
            preserved_interfaces=1,
            preservation_clauses=5,
            distractor_count=8,
            pattern="multi_interface_selective_field_preservation",
        )

    return {
        "task_family": TASK_FAMILY,
        "task_generator_id": TASK_GENERATOR_ID,
        "task_generator_version": TASK_GENERATOR_VERSION,
        "intent": intent,
        "initial_state": initial_state,
        "required_changes": required_changes,
        "constraints": {
            "allowed_assignments": [
                {
                    "interface": item["interface"],
                    "field": item["field"],
                }
                for item in required_changes
            ],
            "preserve_unspecified_state": True,
            "allowed_fields": list(_FIELDS),
        },
        "difficulty": difficulty,
    }


def render_reference_configuration(task: dict[str, Any]) -> str:
    return "\n".join(
        f"interface {item['interface']} {item['field']} {item['value']}"
        for item in task["required_changes"]
    )


def generate_direct_candidate(task: dict[str, Any], seed: int) -> str:
    """Generate a deterministic candidate with controlled realistic failures."""
    required = [dict(item) for item in task["required_changes"]]
    mode = seed % 4

    if mode == 0:
        return render_reference_configuration(task)

    if mode == 1:
        item = required[0]
        if item["field"] == "admin":
            item["value"] = "down" if item["value"] == "up" else "up"
        else:
            item["value"] = int(item["value"]) - 1
        return "\n".join(
            f"interface {change['interface']} {change['field']} {change['value']}"
            for change in required
        )

    if mode == 2:
        return "\n".join(
            f"interface {item['interface']} {item['field']} {item['value']}"
            for item in required[:-1]
        )

    distractor = next(
        interface
        for interface in task["initial_state"]
        if interface not in {
            item["interface"] for item in required
        }
    )
    return "\n".join(
        [
            render_reference_configuration(task),
            f"interface {distractor} admin down",
        ]
    )


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
        violations.append({
            "code": "EMPTY_CONFIGURATION",
            "message": "Configuration contains no commands.",
            "line_number": None,
        })
        return commands, violations

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

        value: int | str
        if field in {"mtu", "vlan"}:
            try:
                value = int(raw_value)
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

        commands.append(ParsedCommand(interface, field, value, line))

    return commands, violations


def validate_configuration(task: dict[str, Any], configuration: str) -> dict[str, Any]:
    """Validate syntax, required changes, and preservation of unspecified state."""
    commands, violations = _parse_configuration(configuration)
    initial_state = task["initial_state"]
    required = {
        (item["interface"], item["field"]): item["value"]
        for item in task["required_changes"]
    }

    observed: dict[tuple[str, str], int | str] = {}
    duplicate_keys: set[tuple[str, str]] = set()

    for command in commands:
        key = (command.interface, command.field)
        if command.interface not in initial_state:
            violations.append({
                "code": "UNKNOWN_INTERFACE",
                "message": f"Unknown interface: {command.interface}.",
                "line_number": None,
            })
        elif key not in required:
            violations.append({
                "code": "UNINTENDED_STATE_CHANGE",
                "message": (
                    "Command modifies state that the intent requires to be "
                    f"preserved: {command.interface} {command.field}."
                ),
                "line_number": None,
            })
        if key in observed:
            duplicate_keys.add(key)
        observed[key] = command.value

    for interface, field in sorted(duplicate_keys):
        violations.append({
            "code": "DUPLICATE_ASSIGNMENT",
            "message": f"Duplicate assignment for {interface} {field}.",
            "line_number": None,
        })

    for (interface, field), expected in required.items():
        key = (interface, field)
        if key not in observed:
            violations.append({
                "code": "MISSING_REQUIRED_SETTING",
                "message": f"Missing required setting: {interface} {field}.",
                "line_number": None,
            })
        elif observed[key] != expected:
            violations.append({
                "code": "INTENT_CONSTRAINT_VIOLATION",
                "message": (
                    f"{interface} {field} is {observed[key]!r}; "
                    f"expected {expected!r}."
                ),
                "line_number": None,
            })

    codes = [violation["code"] for violation in violations]
    return {
        "validator_id": VALIDATOR_ID,
        "validator_version": VALIDATOR_VERSION,
        "valid": not violations,
        "violation_count": len(violations),
        "violation_codes": codes,
        "violations": violations,
        "parsed_command_count": len(commands),
        "normalized_configuration": "\n".join(
            command.source_line for command in commands
        ),
        "required_assignment_count": len(required),
        "observed_assignment_count": len(observed),
        "difficulty": task.get("difficulty"),
    }


def repair_configuration(task: dict[str, Any], candidate: str) -> dict[str, Any]:
    """Apply one bounded deterministic repair and revalidate it."""
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
    """Run direct generation or one bounded generate-validate-repair workflow."""
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
