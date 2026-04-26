"""System dependency graph for ChaosOps-RC."""

from typing import Dict, List, Set, Tuple
from .service import Service


class SystemGraph:
    """Manages the dependency graph and service propagation."""

    def __init__(self, services: Dict[str, Service]):
        """Initialize the system graph.

        Args:
            services: Dict of service_id -> Service objects
        """
        self.services = services
        self._reverse_deps: Dict[str, Set[str]] = {}
        self._compute_reverse_deps()

    def _compute_reverse_deps(self) -> None:
        """Compute reverse dependencies (dependents)."""
        self._reverse_deps = {sid: set() for sid in self.services}
        for sid, service in self.services.items():
            for dep in service.dependencies:
                if dep in self._reverse_deps:
                    self._reverse_deps[dep].add(sid)

    def get_dependents(self, service_id: str) -> Set[str]:
        """Get all services that depend on this service."""
        return self._reverse_deps.get(service_id, set())

    def get_affected_services(self, service_id: str) -> Set[str]:
        """Get all services transitively affected by failure of service_id."""
        affected = set()
        to_visit = [service_id]

        while to_visit:
            current = to_visit.pop(0)
            if current in affected:
                continue
            affected.add(current)

            # Find all dependents of current
            for dependent in self.get_dependents(current):
                if dependent not in affected:
                    to_visit.append(dependent)

        return affected

    def propagate_failure(self, failed_service_id: str, rng, base_rate: float = 0.5) -> Set[str]:
        """Propagate failure through the dependency graph stochastically.

        Args:
            failed_service_id: Service that has failed
            rng: Random number generator
            base_rate: Base probability of failure propagation
        
        Returns:
            Set of newly impacted service IDs
        """
        impacted = set()
        to_check = [(failed_service_id, base_rate)]

        while to_check:
            current_id, current_rate = to_check.pop(0)
            dependents = self.get_dependents(current_id)
            
            for dep_id in dependents:
                # Stochastic chance to propagate
                if rng.random() < current_rate:
                    dep_service = self.services.get(dep_id)
                    if dep_service and not dep_service.is_crashed():
                        # Variable health loss
                        loss = rng.uniform(0.1, 0.4)
                        dep_service.degrade_health(loss)
                        impacted.add(dep_id)
                        
                        # Decay propagation rate for transitive dependents
                        to_check.append((dep_id, current_rate * 0.7))

        return impacted

    def compute_system_health(self) -> float:
        """Compute overall system health as mean of all services."""
        if not self.services:
            return 1.0
        total = sum(s.health for s in self.services.values())
        return total / len(self.services)

    def get_service_dependency_chain(self, service_id: str) -> List[str]:
        """Get the chain of dependencies for a service."""
        if service_id not in self.services:
            return []

        chain = []
        visited = set()
        to_visit = [service_id]

        while to_visit:
            current = to_visit.pop(0)
            if current in visited:
                continue
            visited.add(current)

            service = self.services.get(current)
            if service:
                chain.append(current)
                for dep in service.dependencies:
                    if dep not in visited:
                        to_visit.append(dep)

        return chain

    def get_critical_services(self) -> List[str]:
        """Get critical services (have many dependents)."""
        criticality = {
            sid: len(self.get_dependents(sid))
            for sid in self.services
        }
        # Return services sorted by number of dependents (most critical first)
        return sorted(criticality.items(), key=lambda x: x[1], reverse=True)
