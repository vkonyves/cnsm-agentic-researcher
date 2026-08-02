from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


TASK_FAMILY = "intent_configuration_repair_v1"
BASELINE_TRANSFORMATION = "direct_configuration_generation_v1"
GUARDED_TRANSFORMATION = "generate_validate_repair_v1"
VALIDATOR_ID = "deterministic_netops_validator_v1"
VALIDATOR_VERSION = "1.0"
REPAIRER_ID = "deterministic_netops_repairer_v1"
REPAIRER_VERSION = "1.0"

_INTERFACE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]*$")
_COMMAND_RE = re.compile(
    r"^interface\s+(?P<interface>\S+)\s+"
    r"(?P<field>mtu|admin|vlan)\s+(?P<value>\S+)$"
)


@dataclass(frozen=True)
class ParsedCommand:
    interface: str
    field: str
    value: int | str
    source_line: str


def generate_task(index: int) -> dict[str, Any]:
    """Generate one deterministic, self-contained NetOps intent task."""
    if not isinstance(index, int) or isinstance(index, bool) or index <= 0:
        raise ValueError("index must be a positive integer")

    interface = f"eth{index}"
    required_mtu = 1500 + 100 * ((index - 1) % 3)
    required_vlan = 10 + ((index - 1) % 4)
    required_admin = "up"

    return {
        "task_family": TASK_FAMILY,
        "interface": interface,
        "intent": (
            f"Configure {interface} administratively {required_admin}, "
            f"with MTU {required_mtu} and access VLAN {required_vlan}."
        ),
        "initial_state": {
            "interface": interface,
            "admin": "down",
            "mtu": 1500,
            "vlan": 1,
        },
        "constraints": {
            "required_admin": required_admin,
            "required_mtu": required_mtu,
            "required_vlan": required_vlan,
            "allowed_interfaces": [interface],
            "allowed_fields": ["admin", "mtu", "vlan"],
        },
    }


def render_reference_configuration(task: dict[str, Any]) -> str:
    constraints = task["constraints"]
    interface = task["interface"]
    return "\n".join(
        [
            f"interface {interface} admin {constraints['required_admin']}",
            f"interface {interface} mtu {constraints['required_mtu']}",
            f"interface {interface} vlan {constraints['required_vlan']}",
        ]
    )


def generate_direct_candidate(task: dict[str, Any], seed: int) -> str:
    """Generate a deterministic direct candidate with controlled failure modes."""
    reference = render_reference_configuration(task).splitlines()
    mode = seed % 4

    if mode == 0:
        return "\n".join(reference)
    if mode == 1:
        # Wrong MTU while all syntax remains valid.
        constraints = task["constraints"]
        interface = task["interface"]
        return "\n".join(
            [
                f"interface {interface} admin {constraints['required_admin']}",
                f"interface {interface} mtu {constraints['required_mtu'] - 100}",
                f"interface {interface} vlan {constraints['required_vlan']}",
            ]
        )
    if mode == 2:
        # Missing VLAN command.
        return "\n".join(reference[:2])

    # Correct target plus an unintended change on another interface.
    return "\n".join(
        [
            *reference,
            "interface management0 admin down",
        ]
    )


def _parse_configuration(configuration: str) -> tuple[list[ParsedCommand], list[dict[str, Any]]]:
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
    """Validate syntax, intent satisfaction, and unintended changes."""
    commands, violations = _parse_configuration(configuration)
    constraints = task["constraints"]
    allowed_interfaces = set(constraints["allowed_interfaces"])
    allowed_fields = set(constraints["allowed_fields"])

    observed: dict[tuple[str, str], int | str] = {}
    duplicate_keys: set[tuple[str, str]] = set()

    for command in commands:
        key = (command.interface, command.field)
        if command.interface not in allowed_interfaces:
            violations.append({
                "code": "UNINTENDED_INTERFACE_CHANGE",
                "message": (
                    "Command modifies an interface outside the task scope: "
                    f"{command.interface}"
                ),
                "line_number": None,
            })
        if command.field not in allowed_fields:
            violations.append({
                "code": "UNINTENDED_FIELD_CHANGE",
                "message": f"Field is outside the task scope: {command.field}",
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

    interface = task["interface"]
    required = {
        "admin": constraints["required_admin"],
        "mtu": constraints["required_mtu"],
        "vlan": constraints["required_vlan"],
    }
    for field, expected in required.items():
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
