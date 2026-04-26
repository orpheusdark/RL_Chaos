"""Multi-service distributed system environment for ChaosOps-RC."""

import random
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
from threading import Lock

from .base import OpenEnvEnv
from .models import Service, ServiceMetrics, SystemGraph
from failures import FailureInjector
from reward import RewardComputer, AntiCheatDetector


class ChaosOpsRCEnv(OpenEnvEnv):
    """Multi-service recovery control environment for RL training.

    Key features:
    - Multiple interdependent services
    - Structured failure injection with propagation
    - Partial observability (agent sees limited logs/metrics)
    - Strict tool-based action interface
    - Multi-signal reward with anti-exploitation rules
    - Curriculum learning support
    """

    def __init__(
        self,
        max_steps: int = 30,
        seed: Optional[int] = None,
        curriculum_level: int = 1,
    ):
        """Initialize the environment.

        Args:
            max_steps: Maximum steps per episode
            seed: Random seed for reproducibility
            curriculum_level: Difficulty level (1-4)
        """
        self.max_steps = max_steps
        self.curriculum_level = curriculum_level
        self.rng = random.Random(seed)
        self._lock = Lock()

        # Valid action names
        self.valid_actions = {
            "inspect_logs",
            "restart_service",
            "rollback_service",
            "patch_config",
            "allocate_resources",
            "inspect_metrics",
            "drain_requests",
            "promote_replica",
        }

        # Initialize state
        self.services: Dict[str, Service] = {}
        self.system_graph: Optional[SystemGraph] = None
        self.failure_injector: Optional[FailureInjector] = None
        self.reward_computer = RewardComputer()
        self.anti_cheat_detector = AntiCheatDetector()
        self.step_count = 0
        self.episode_reward = 0.0
        self.logs: List[Dict[str, Any]] = []
        self.alerts: List[Dict[str, Any]] = []
        self._previous_health = 0.0
        self.misleading_log_chance = 0.15 # 15% chance of misleading logs
        self.diagnosed_root_cause = False

        # Anti-cheat tracking
        self.action_history: List[str] = []
        self.restart_counts: Dict[str, int] = {}
        self.resource_allocations: Dict[str, int] = {}

        # Curriculum settings
        self._init_curriculum_settings()

        # Reset to initialize
        self.reset()

    def _init_curriculum_settings(self) -> None:
        """Initialize settings based on curriculum level."""
        if self.curriculum_level == 1:
            # Simple: 2 services, latency spike only
            self.num_services = 2
            self.failure_types = ["latency_spike"]
            self.max_steps = 15
            self.observability_level = "high"  # Full logs
        elif self.curriculum_level == 2:
            # Linear deps: 3 services, latency + config
            self.num_services = 3
            self.failure_types = ["latency_spike", "config_corruption"]
            self.max_steps = 20
            self.observability_level = "medium"  # 5 recent logs
        elif self.curriculum_level == 3:
            # Multi-deps: 4 services, all types
            self.num_services = 4
            self.failure_types = ["latency_spike", "config_corruption", "version_drift", "resource_exhaustion"]
            self.max_steps = 25
            self.observability_level = "low"  # 3 recent logs
        else:
            # Complex: 5+ services, cascading
            self.num_services = 5
            self.failure_types = ["latency_spike", "config_corruption", "version_drift", "resource_exhaustion", "cascading_failure"]
            self.max_steps = 30
            self.observability_level = "minimal"  # Sampled logs
            self.misleading_log_chance = 0.25

    def _create_system(self) -> None:
        """Create services and dependency graph."""
        if self.num_services == 2:
            self.services = {
                "api": Service(
                    service_id="api",
                    health=1.0,
                    status="healthy",
                    version=1,
                    dependencies=["db"],
                    replicas=1,
                ),
                "db": Service(
                    service_id="db",
                    health=1.0,
                    status="healthy",
                    version=1,
                    dependencies=[],
                    replicas=1,
                ),
            }
        elif self.num_services == 3:
            self.services = {
                "api": Service(
                    service_id="api",
                    health=1.0,
                    status="healthy",
                    version=1,
                    dependencies=["auth", "db"],
                ),
                "auth": Service(
                    service_id="auth",
                    health=1.0,
                    status="healthy",
                    version=1,
                    dependencies=["db"],
                ),
                "db": Service(
                    service_id="db",
                    health=1.0,
                    status="healthy",
                    version=1,
                    dependencies=[],
                ),
            }
        elif self.num_services == 4:
            self.services = {
                "gateway": Service(
                    service_id="gateway",
                    health=1.0,
                    status="healthy",
                    version=1,
                    dependencies=["api"],
                ),
                "api": Service(
                    service_id="api",
                    health=1.0,
                    status="healthy",
                    version=1,
                    dependencies=["auth", "db"],
                ),
                "auth": Service(
                    service_id="auth",
                    health=1.0,
                    status="healthy",
                    version=1,
                    dependencies=["db"],
                ),
                "db": Service(
                    service_id="db",
                    health=1.0,
                    status="healthy",
                    version=1,
                    dependencies=[],
                ),
            }
        else:
            self.services = {
                "gateway": Service(
                    service_id="gateway",
                    health=1.0,
                    status="healthy",
                    version=1,
                    dependencies=["api", "payment"],
                ),
                "api": Service(
                    service_id="api",
                    health=1.0,
                    status="healthy",
                    version=1,
                    dependencies=["auth", "db"],
                ),
                "auth": Service(
                    service_id="auth",
                    health=1.0,
                    status="healthy",
                    version=1,
                    dependencies=["db"],
                ),
                "payment": Service(
                    service_id="payment",
                    health=1.0,
                    status="healthy",
                    version=1,
                    dependencies=["db"],
                ),
                "db": Service(
                    service_id="db",
                    health=1.0,
                    status="healthy",
                    version=1,
                    dependencies=[],
                ),
            }

        self.system_graph = SystemGraph(self.services)

        # Initialize failure injector
        self.failure_injector = FailureInjector(
            system_graph=self.system_graph,
            rng=self.rng,
            allowed_failures=self.failure_types,
        )

    def reset(self, **kwargs) -> Dict[str, Any]:
        """Reset the environment to initial state.

        Returns:
            observation dict
        """
        with self._lock:
            self._create_system()
            self.step_count = 0
            self.episode_reward = 0.0
            self.logs = []
            self.alerts = []
            self.action_history = []
            self.restart_counts = {sid: 0 for sid in self.services}
            self.resource_allocations = {sid: 0 for sid in self.services}
            self._previous_health = 1.0

            # Reset reward tracking
            self.reward_computer.reset()
            self.anti_cheat_detector.reset()

            # Inject initial failure
            self._inject_initial_failure()

            return self.get_observation()

    def _inject_initial_failure(self) -> None:
        """Inject an initial failure to start the episode."""
        # Select a random service to fail (not DB to start with)
        non_db_services = [s for s in self.services.keys() if s != "db"]
        if non_db_services and self.failure_injector:
            failed_service_id = self.rng.choice(non_db_services)
            failure_type = self.rng.choice(self.failure_types)

            self.failure_injector.inject_failure(
                service_id=failed_service_id,
                failure_type=failure_type,
                logs_callback=self._add_log,
                alerts_callback=self._add_alert,
                step_count=self.step_count
            )

    def _add_log(self, service_id: str, level: str, message: str) -> None:
        """Add a log entry."""
        self.logs.append({
            "timestamp": self.step_count,
            "service": service_id,
            "level": level,
            "message": message,
        })

    def _add_alert(self, service_id: str, alert_type: str, severity: str) -> None:
        """Add an alert."""
        self.alerts.append({
            "service": service_id,
            "type": alert_type,
            "severity": severity,
        })

    def get_observation(self) -> Dict[str, Any]:
        """Get current observation (partial view of state).

        Returns:
            observation dict with logs, metrics, alerts, topology
        """
        # Limit visible logs based on observability level
        if self.observability_level == "high":
            visible_logs = self.logs[-10:]
        elif self.observability_level == "medium":
            visible_logs = self.logs[-5:]
        elif self.observability_level == "low":
            visible_logs = self.logs[-3:]
        else:  # minimal
            # Sample 50% of logs randomly
            visible_logs = self.rng.sample(self.logs, max(0, len(self.logs) // 2))

        # Aggregate metrics per service
        metrics = {}
        for sid, service in self.services.items():
            # Add observation noise to metrics
            noise = self.rng.uniform(-0.05, 0.05)
            metrics[f"{sid}_health"] = max(0.0, min(1.0, service.health + noise))
            metrics[f"{sid}_latency"] = service.metrics.latency_p99 * (1.0 + self.rng.uniform(-0.1, 0.1))
            metrics[f"{sid}_error_rate"] = service.metrics.error_rate

        # System-wide metrics
        if self.system_graph:
            metrics["system_health"] = self.system_graph.compute_system_health()

        # Topology (observable)
        topology = {
            "services": list(self.services.keys()),
            "dependencies": {
                sid: service.dependencies
                for sid, service in self.services.items()
            },
        }

        return {
            "logs": visible_logs,
            "metrics": metrics,
            "alerts": self.alerts[-10:],  # Last 10 alerts
            "topology": topology,
            "step": self.step_count,
            "episode_reward": self.episode_reward,
        }

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute one step of the environment.

        Args:
            action: Action dict with "action" and "params" keys

        Returns:
            Tuple of (observation, reward, terminated, info)
        """
        with self._lock:
            self.step_count += 1
            self.action_history.append(action.get("action", "unknown"))

            # Track previous health for reward computation
            previous_health = self.system_graph.compute_system_health() if self.system_graph else 0.0

            # Update active failures
            if self.failure_injector:
                self.failure_injector.update_failures(self.step_count)

            # Execute action
            action_result = self._execute_action(action)

            # Process pending effects (e.g., recovery after restart)
            for sid, service in self.services.items():
                remaining_effects = []
                for effect in service.pending_effects:
                    effect["delay"] -= 1
                    if effect["delay"] <= 0:
                        if effect["type"] == "recovery":
                            service.restore_health(0.6) # Major recovery boost
                            self._add_log(sid, "info", "Recovery effect completed")
                    else:
                        remaining_effects.append(effect)
                service.pending_effects = remaining_effects

            # Compute reward
            reward = self._compute_reward(action, action_result, previous_health, self.diagnosed_root_cause)
            self.episode_reward += reward

            # Check termination
            terminated = self.step_count >= self.max_steps or self._is_terminal()

            # Build observation
            obs = self.get_observation()

            info = {
                "action_result": action_result,
                "previous_health": previous_health,
            }

            return obs, reward, terminated, info

    def _execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an action.

        Returns:
            result dict with "ok", "error_code", and other fields
        """
        action_name = action.get("action")
        params = action.get("params", {})

        # Validate action
        if action_name not in self.valid_actions:
            return {"ok": False, "error_code": "INVALID_ACTION"}

        # Dispatch to action handler
        if action_name == "inspect_logs":
            return self._handle_inspect_logs(params)
        elif action_name == "restart_service":
            return self._handle_restart_service(params)
        elif action_name == "rollback_service":
            return self._handle_rollback_service(params)
        elif action_name == "patch_config":
            return self._handle_patch_config(params)
        elif action_name == "allocate_resources":
            return self._handle_allocate_resources(params)
        elif action_name == "inspect_metrics":
            return self._handle_inspect_metrics(params)
        elif action_name == "drain_requests":
            return self._handle_drain_requests(params)
        elif action_name == "promote_replica":
            return self._handle_promote_replica(params)

        return {"ok": False, "error_code": "UNKNOWN_ACTION"}

    def _handle_inspect_logs(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Inspect logs for a service."""
        service_id = params.get("service_id")

        if service_id not in self.services:
            return {"ok": False, "error_code": "SERVICE_NOT_FOUND"}

        # Return last 5 logs for this service
        service_logs = [
            log for log in self.logs
            if log["service"] == service_id
        ][-10:]

        # Chance of misleading logs
        if self.rng.random() < self.misleading_log_chance:
            service_logs = deepcopy(service_logs)
            for log in service_logs:
                if "failure" in log["message"] or "error" in log["message"]:
                    log["message"] = "Background noise: service heartbeat okay"
                    log["level"] = "info"

        # Check if inspecting the root cause
        if self.failure_injector:
            active = self.failure_injector.active_failures.get(service_id, [])
            if any(f.get("is_root_cause") for f in active):
                self.diagnosed_root_cause = True

        return {"ok": True, "logs": service_logs}

    def _handle_restart_service(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Restart a service."""
        service_id = params.get("service_id")

        if service_id not in self.services:
            return {"ok": False, "error_code": "SERVICE_NOT_FOUND"}

        service = self.services[service_id]

        # Rate limiting: max 3 restarts per episode
        self.restart_counts[service_id] += 1
        if self.restart_counts[service_id] > 3:
            return {"ok": False, "error_code": "RESTART_LIMIT_EXCEEDED"}

        # Restart effect (1-step delay before taking effect)
        service.restart_count += 1
        service.last_restart_step = self.step_count

        # Mark pending recovery
        service.pending_effects.append({
            "type": "recovery",
            "delay": 1,
        })

        self._add_log(service_id, "info", "Service restart initiated")

        return {"ok": True, "delayed": True}

    def _handle_rollback_service(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Rollback a service to previous version."""
        service_id = params.get("service_id")

        if service_id not in self.services:
            return {"ok": False, "error_code": "SERVICE_NOT_FOUND"}

        service = self.services[service_id]

        if len(service.version_history) <= 1:
            return {"ok": False, "error_code": "NO_PREVIOUS_VERSION"}

        # Rollback version
        service.version = service.version_history[-2]
        service.restore_health(0.1)

        self._add_log(service_id, "info", f"Rolled back to version {service.version}")

        return {"ok": True}

    def _handle_patch_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a config patch to a service."""
        service_id = params.get("service_id")
        patch = params.get("patch", {})

        if service_id not in self.services:
            return {"ok": False, "error_code": "SERVICE_NOT_FOUND"}

        if not isinstance(patch, dict):
            return {"ok": False, "error_code": "INVALID_PATCH"}

        service = self.services[service_id]

        # Apply patch (simple implementation)
        service.restore_health(0.1)
        self._add_log(service_id, "info", "Config patched")

        return {"ok": True}

    def _handle_allocate_resources(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Allocate resources to a service."""
        service_id = params.get("service_id")
        cpu = params.get("cpu", 0)
        memory = params.get("memory", 0)

        if service_id not in self.services:
            return {"ok": False, "error_code": "SERVICE_NOT_FOUND"}

        if cpu <= 0 or memory <= 0:
            return {"ok": False, "error_code": "INVALID_RESOURCES"}

        service = self.services[service_id]

        # Limit allocations per service
        self.resource_allocations[service_id] += 1
        if self.resource_allocations[service_id] > 2:
            return {"ok": False, "error_code": "ALLOCATION_LIMIT_EXCEEDED"}

        service.allocated_cpu = cpu
        service.allocated_memory = memory
        service.restore_health(0.15)

        self._add_log(service_id, "info", f"Allocated {cpu}m CPU, {memory}MB memory")

        return {"ok": True}

    def _handle_inspect_metrics(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Inspect metrics for a service."""
        service_id = params.get("service_id")

        if service_id not in self.services:
            return {"ok": False, "error_code": "SERVICE_NOT_FOUND"}

        service = self.services[service_id]

        return {
            "ok": True,
            "metrics": {
                "latency_p99": service.metrics.latency_p99,
                "error_rate": service.metrics.error_rate,
                "cpu_usage": service.metrics.cpu_usage,
                "memory_usage": service.metrics.memory_usage,
                "queue_depth": service.metrics.request_queue_depth,
            },
        }

    def _handle_drain_requests(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Gracefully drain requests from a service."""
        service_id = params.get("service_id")

        if service_id not in self.services:
            return {"ok": False, "error_code": "SERVICE_NOT_FOUND"}

        service = self.services[service_id]
        service.metrics.request_queue_depth = 0

        self._add_log(service_id, "info", "Draining requests")

        return {"ok": True}

    def _handle_promote_replica(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Promote a replica (failover)."""
        service_id = params.get("service_id")

        if service_id not in self.services:
            return {"ok": False, "error_code": "SERVICE_NOT_FOUND"}

        service = self.services[service_id]

        if service.replicas <= 1:
            return {"ok": False, "error_code": "NO_REPLICAS"}

        # Promote replica
        service.replicas -= 1
        service.restore_health(0.2)

        self._add_log(service_id, "info", "Replica promoted")

        return {"ok": True}

    def _compute_reward(self, action: Dict[str, Any], action_result: Dict[str, Any], previous_health: float, diagnosed_root_cause: bool = False) -> float:
        """Compute reward for this step using multi-signal reward function."""
        current_health = self.system_graph.compute_system_health() if self.system_graph else 0.0

        # Get status info
        all_healthy = all(s.is_healthy() for s in self.services.values())
        any_critical_crashed = any(
            s.is_crashed() and s.service_id in ["db", "api", "gateway"]
            for s in self.services.values()
        )

        # Get active failures
        active_failures = {
            sid: self.failure_injector.get_active_failures(sid) if self.failure_injector else []
            for sid in self.services
        }

        # Compute reward
        reward = self.reward_computer.compute_reward(
            action=action,
            action_result=action_result,
            current_health=current_health,
            previous_health=previous_health,
            step_count=self.step_count,
            num_services=len(self.services),
            active_failures=active_failures,
            all_healthy=all_healthy,
            any_critical_crashed=any_critical_crashed,
            diagnosed_root_cause=diagnosed_root_cause,
        )

        # Track anti-cheat
        self.anti_cheat_detector.record_action(action.get("action", "unknown"))

        # Apply anti-cheat penalty if exploitation detected
        exploitation_score = self.anti_cheat_detector.get_exploitation_score()
        if exploitation_score > 0.5:
            reward -= exploitation_score * 0.5

        return reward

    def _is_terminal(self) -> bool:
        """Check if episode is terminal.

        Terminal if all services are healthy or all services are crashed.
        """
        if not self.services:
            return True

        all_healthy = all(s.is_healthy() for s in self.services.values())
        any_critical_crashed = any(
            s.is_crashed() and s.service_id in ["db", "api", "gateway"]
            for s in self.services.values()
        )

        return all_healthy or any_critical_crashed
