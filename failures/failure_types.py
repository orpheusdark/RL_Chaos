"""Failure types and definitions for ChaosOps-RC."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class FailureType:
    """Definition of a failure type."""

    name: str  # "latency_spike", "config_corruption", etc.
    description: str
    observable_signals: List[str]  # What agent sees
    valid_fix_paths: List[str]  # What can fix it
    base_health_loss: float  # How much health is lost
    propagation_factor: float  # How much dependents are affected
    duration_steps: int  # How long failure lasts if not fixed


# Standard failure types
LATENCY_SPIKE = FailureType(
    name="latency_spike",
    description="Network congestion or high load causing latency spike",
    observable_signals=[
        "high latency detected",
        "p99 latency > 5s",
        "increased error rate",
        "timeout errors in logs",
    ],
    valid_fix_paths=[
        "reduce_load",
        "allocate_resources",
        "drain_requests",
        "restart_service",
    ],
    base_health_loss=0.3,
    propagation_factor=0.15,
    duration_steps=5,
)

DEPENDENCY_FAILURE = FailureType(
    name="dependency_failure",
    description="Downstream service is unavailable or failing",
    observable_signals=[
        "connection refused",
        "dependency failure detected",
        "upstream service unavailable",
    ],
    valid_fix_paths=[
        "restart_dependency",
        "promote_replica",
        "allocate_resources",
    ],
    base_health_loss=0.4,
    propagation_factor=0.25,
    duration_steps=3,
)

CONFIG_CORRUPTION = FailureType(
    name="config_corruption",
    description="Configuration mismatch or corruption",
    observable_signals=[
        "config validation error",
        "invalid configuration",
        "config mismatch detected",
    ],
    valid_fix_paths=[
        "rollback_service",
        "patch_config",
    ],
    base_health_loss=0.35,
    propagation_factor=0.1,
    duration_steps=2,
)

VERSION_DRIFT = FailureType(
    name="version_drift",
    description="Service version mismatch with dependencies",
    observable_signals=[
        "API contract violation",
        "version mismatch detected",
        "schema incompatibility error",
    ],
    valid_fix_paths=[
        "rollback_service",
        "restart_service",
    ],
    base_health_loss=0.25,
    propagation_factor=0.2,
    duration_steps=2,
)

RESOURCE_EXHAUSTION = FailureType(
    name="resource_exhaustion",
    description="CPU/memory limits hit, OOMKilled, or throttled",
    observable_signals=[
        "OOMKilled",
        "throttled",
        "resource limit exceeded",
        "high memory usage",
    ],
    valid_fix_paths=[
        "allocate_resources",
        "drain_requests",
        "restart_service",
    ],
    base_health_loss=0.45,
    propagation_factor=0.3,
    duration_steps=4,
)

CASCADING_FAILURE = FailureType(
    name="cascading_failure",
    description="Multiple services failing in sequence due to propagation",
    observable_signals=[
        "cascading failure detected",
        "multiple services unhealthy",
        "systemic failure pattern",
    ],
    valid_fix_paths=[
        "stabilize_root_cause",
        "restart_critical_service",
    ],
    base_health_loss=0.5,
    propagation_factor=0.4,
    duration_steps=6,
)

# Registry of all failure types
ALL_FAILURE_TYPES = {
    "latency_spike": LATENCY_SPIKE,
    "dependency_failure": DEPENDENCY_FAILURE,
    "config_corruption": CONFIG_CORRUPTION,
    "version_drift": VERSION_DRIFT,
    "resource_exhaustion": RESOURCE_EXHAUSTION,
    "cascading_failure": CASCADING_FAILURE,
}


def get_failure_type(failure_name: str) -> Optional[FailureType]:
    """Get a failure type by name."""
    return ALL_FAILURE_TYPES.get(failure_name)


def list_failure_types() -> List[str]:
    """List all available failure types."""
    return list(ALL_FAILURE_TYPES.keys())
