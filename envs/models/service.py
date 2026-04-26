"""Service state model for ChaosOps-RC."""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class ServiceMetrics:
    """Metrics for a service."""
    latency_p99: float = 0.1  # seconds
    error_rate: float = 0.0   # [0, 1]
    cpu_usage: float = 0.3    # [0, 1]
    memory_usage: float = 0.4 # [0, 1]
    request_queue_depth: int = 0


@dataclass
class Service:
    """Represents a single service in the system."""

    service_id: str
    health: float = 1.0  # [0, 1]
    status: str = "healthy"  # "healthy" | "degraded" | "failing" | "crashed"
    version: int = 1  # Tracks rollback capability
    dependencies: List[str] = field(default_factory=list)  # List of service IDs this depends on
    replicas: int = 1  # Number of running replicas

    # Metrics (observable to agent)
    metrics: ServiceMetrics = field(default_factory=ServiceMetrics)

    # Resource allocation
    allocated_cpu: int = 256    # millicores
    allocated_memory: int = 512 # MB

    # Event tracking
    restart_count: int = 0
    last_restart_step: int = -100
    version_history: List[int] = field(default_factory=list)

    # Internal state (not directly observable)
    pending_effects: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        """Ensure metrics is initialized."""
        if self.metrics is None:
            self.metrics = ServiceMetrics()
        if self.version_history is None:
            self.version_history = [self.version]
        elif self.version not in self.version_history:
            self.version_history.append(self.version)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "service_id": self.service_id,
            "health": self.health,
            "status": self.status,
            "version": self.version,
            "dependencies": self.dependencies,
            "replicas": self.replicas,
            "metrics": {
                "latency_p99": self.metrics.latency_p99,
                "error_rate": self.metrics.error_rate,
                "cpu_usage": self.metrics.cpu_usage,
                "memory_usage": self.metrics.memory_usage,
                "queue_depth": self.metrics.request_queue_depth,
            },
            "restart_count": self.restart_count,
            "allocated_cpu": self.allocated_cpu,
            "allocated_memory": self.allocated_memory,
        }

    def degrade_health(self, amount: float) -> None:
        """Degrade health by amount."""
        self.health = max(0.0, self.health - amount)
        self._update_status()

    def restore_health(self, amount: float) -> None:
        """Restore health by amount."""
        self.health = min(1.0, self.health + amount)
        self._update_status()

    def _update_status(self) -> None:
        """Update status based on health."""
        if self.health >= 0.8:
            self.status = "healthy"
        elif self.health >= 0.5:
            self.status = "degraded"
        elif self.health >= 0.2:
            self.status = "failing"
        else:
            self.status = "crashed"

    def is_healthy(self) -> bool:
        """Check if service is in healthy state."""
        return self.status == "healthy"

    def is_crashed(self) -> bool:
        """Check if service is crashed."""
        return self.status == "crashed"
