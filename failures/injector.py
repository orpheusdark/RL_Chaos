"""Failure injection engine for ChaosOps-RC."""

import random
from typing import Any, Dict, List, Optional, Set

from envs.models import Service, SystemGraph
from .failure_types import FailureType, get_failure_type


class FailureInjector:
    """Manages failure injection and propagation."""

    def __init__(
        self,
        system_graph: SystemGraph,
        rng: random.Random,
        allowed_failures: Optional[List[str]] = None,
    ):
        """Initialize the failure injector.

        Args:
            system_graph: System dependency graph
            rng: Random number generator
            allowed_failures: List of failure types to allow (None = all)
        """
        self.system_graph = system_graph
        self.rng = rng
        self.allowed_failures = allowed_failures or []
        self.active_failures: Dict[str, List[Dict[str, Any]]] = {
            sid: [] for sid in system_graph.services
        }
        self.failure_countdown: Dict[str, int] = {}

    def inject_failure(
        self,
        service_id: str,
        failure_type: str,
        logs_callback,
        alerts_callback,
        step_count: int = 0,
    ) -> bool:
        """Inject a failure into a service as a root cause.

        Args:
            service_id: Target service
            failure_type: Type of failure to inject
            logs_callback: Callback to add logs
            alerts_callback: Callback to add alerts
            step_count: Current step number

        Returns:
            True if failure was injected, False if invalid
        """
        if service_id not in self.system_graph.services:
            return False

        failure_def = get_failure_type(failure_type)
        if not failure_def:
            return False

        service = self.system_graph.services[service_id]

        # Degrade service health
        service.degrade_health(failure_def.base_health_loss)

        # Add logs
        signal = self.rng.choice(failure_def.observable_signals)
        logs_callback(service_id=service_id, level="error", message=signal)

        # Add alert
        alerts_callback(
            service_id=service_id,
            alert_type=failure_type,
            severity="critical", # Root cause is critical
        )

        # Track active failure as root cause
        self.active_failures[service_id].append({
            "type": failure_type,
            "injected_at": step_count,
            "duration": failure_def.duration_steps,
            "is_root_cause": True,
        })

        # Propagate to dependents stochastically
        impacted_services = self.system_graph.propagate_failure(
            service_id, 
            self.rng, 
            base_rate=failure_def.propagation_factor * 2.0 # Scale factor for stochasticity
        )

        # Log cascading effects for all impacted services
        for imp_id in impacted_services:
            # Mark as dependent failure
            self.active_failures[imp_id].append({
                "type": "dependency_failure",
                "injected_at": step_count,
                "duration": failure_def.duration_steps,
                "is_root_cause": False,
                "root_cause_service": service_id
            })
            
            logs_callback(
                service_id=imp_id,
                level="warn",
                message=f"Cascading failure from {service_id}",
            )
            alerts_callback(
                service_id=imp_id,
                alert_type="dependency_failure",
                severity="high",
            )

        return True

    def _propagate_failure(self, *args, **kwargs):
        """Deprecated: Logic moved to SystemGraph.propagate_failure."""
        pass

    def update_failures(self, step_count: int) -> None:
        """Update active failures (tick countdowns, resolve effects).

        Args:
            step_count: Current step number
        """
        for service_id, failures in self.active_failures.items():
            remaining = []

            for failure in failures:
                failure_def = get_failure_type(failure["type"])
                if not failure_def:
                    continue

                # Check if failure duration expired
                steps_active = step_count - failure["injected_at"]
                if steps_active >= failure_def.duration_steps:
                    # Failure resolved - restore some health
                    service = self.system_graph.services.get(service_id)
                    if service:
                        service.restore_health(0.05)
                else:
                    remaining.append(failure)

            self.active_failures[service_id] = remaining

    def get_active_failures(self, service_id: str) -> List[Dict[str, Any]]:
        """Get active failure records for a service."""
        return self.active_failures.get(service_id, [])

    def is_service_failing(self, service_id: str) -> bool:
        """Check if a service has active failures."""
        return len(self.active_failures.get(service_id, [])) > 0

    def get_critical_services_failing(self) -> List[str]:
        """Get critical services that are currently failing."""
        critical = ["db", "api", "gateway"]
        failing = [s for s in critical if self.is_service_failing(s)]
        return failing
