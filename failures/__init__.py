"""Failure injection for ChaosOps-RC."""

from .failure_types import (
    FailureType,
    LATENCY_SPIKE,
    DEPENDENCY_FAILURE,
    CONFIG_CORRUPTION,
    VERSION_DRIFT,
    RESOURCE_EXHAUSTION,
    CASCADING_FAILURE,
    ALL_FAILURE_TYPES,
    get_failure_type,
    list_failure_types,
)
from .injector import FailureInjector

__all__ = [
    "FailureType",
    "LATENCY_SPIKE",
    "DEPENDENCY_FAILURE",
    "CONFIG_CORRUPTION",
    "VERSION_DRIFT",
    "RESOURCE_EXHAUSTION",
    "CASCADING_FAILURE",
    "ALL_FAILURE_TYPES",
    "get_failure_type",
    "list_failure_types",
    "FailureInjector",
]
