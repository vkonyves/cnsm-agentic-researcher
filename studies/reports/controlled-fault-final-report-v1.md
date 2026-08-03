# Frozen Controlled-Fault Experiment Report

**Report status:** PASS
**Study:** `controlled-fault-frozen-plan-v1`

## Provenance

- Plan SHA-256: `b10c0fa2834e1e08a81731892b21ef827a162eae9f594bcb2354f537d652e521`
- Code fingerprint SHA-256: `9d46a914977d7dfb9bcb3406d72781a46b11285a45c5a8c59937718865808d2a`
- Git commit: `4b93ba8d6bea0ff691a1999f77f620b13e73c1d0`
- Launch-lock SHA-256: `934ce831485688685a183a92a1f66e69d3f585a9c9d8d48fa0663d514ea1cc06`
- Audit-report SHA-256: `c2726bc1028ac415b14f67273eacfc2e0b6fd087016b9594466a1d656adef545`
- Model: `gpt-5-mini` via `openai_responses`

## Execution summary

- Planned pairs: 40
- Complete scientific pairs: 39
- Incomplete pairs: 1
- Model calls used: 79

## Paired analysis

- Baseline success: 0/39
- Guarded success: 37/39
- Paired difference: 0.948718
- Exact two-sided McNemar p-value: 1.45519152284e-11
- Paired bootstrap 95% CI: [0.871795, 1.000000]

## Results by fault class

| Fault class | Pairs | Baseline | Guarded | Difference |
|---|---:|---:|---:|---:|
| `break_before_make` | 8 | 0/8 | 7/8 | 0.875000 |
| `dropped_required_restore` | 8 | 0/8 | 7/8 | 0.875000 |
| `no_op_command` | 7 | 0/7 | 7/7 | 1.000000 |
| `offline_change_before_shutdown` | 8 | 0/8 | 8/8 | 1.000000 |
| `protected_interface_change` | 8 | 0/8 | 8/8 | 1.000000 |

## Results by workflow pattern

| Workflow pattern | Pairs | Baseline | Guarded | Difference |
|---|---:|---:|---:|---:|
| `composed_redundancy_and_access_maintenance` | 5 | 0/5 | 5/5 | 1.000000 |
| `failover_then_offline_reconfiguration` | 5 | 0/5 | 4/5 | 0.800000 |
| `make_before_break_uplink_switchover` | 4 | 0/4 | 4/4 | 1.000000 |
| `paired_link_atomic_mtu_change` | 5 | 0/5 | 5/5 | 1.000000 |
| `redundant_uplink_maintenance_window` | 5 | 0/5 | 5/5 | 1.000000 |
| `rolling_access_vlan_migration` | 5 | 0/5 | 5/5 | 1.000000 |
| `safe_access_service_transfer` | 5 | 0/5 | 4/5 | 0.800000 |
| `shutdown_configure_restore` | 5 | 0/5 | 5/5 | 1.000000 |

## Incomplete source cases

### pair-000010 / task-000010

- Workflow: `make_before_break_uplink_switchover`
- Assigned fault: `no_op_command`
- Source stage: `INVALID`
- Source validation violations: `UNINTENDED_STATE_CHANGE`, `FINAL_STATE_MISMATCH`

## Unsuccessful guarded repairs

### pair-000013 / task-000013

- Workflow: `failover_then_offline_reconfiguration`
- Fault: `break_before_make`
- Violations before repair: `TRANSIENT_AVAILABILITY_VIOLATION`
- Violations after repair: `INTERFACE_NOT_DOWN_FOR_CHANGE`

**Injected candidate**

```text
interface uplink3 admin down
interface uplink4 admin up
interface uplink3 vlan 210
```

**Guarded output**

```text
interface uplink4 admin up
interface uplink3 vlan 210
interface uplink3 admin down
```

### pair-000028 / task-000028

- Workflow: `safe_access_service_transfer`
- Fault: `dropped_required_restore`
- Violations before repair: `TRANSIENT_AVAILABILITY_VIOLATION`, `FINAL_STATE_MISMATCH`, `FINAL_ACTIVE_COUNT_VIOLATION`
- Violations after repair: `NO_OP_COMMAND`

**Injected candidate**

```text
interface edge14 admin down
interface edge14 vlan 40
interface edge13 admin down
```

**Guarded output**

```text
interface edge14 admin down
interface edge13 admin up
interface edge14 vlan 40
interface edge14 admin up
interface edge13 admin down
```

## Interpretation

Among complete paired cases, deterministic validation plus one bounded repair call improved success by 0.948718. One planned pair was excluded because its hosted source candidate was invalid before controlled-fault injection. The two guarded failures remain scored failures and are retained as case studies rather than corrected post hoc.
