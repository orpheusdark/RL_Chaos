"""Multi-signal reward function for ChaosOps-RC."""

from typing import Any, Dict, List, Optional


class RewardComputer:
    """Computes multi-signal reward with anti-exploitation safeguards."""

    def __init__(self):
        """Initialize reward computer."""
        self.restart_spam_threshold = 3
        self.allocation_spam_threshold = 2
        self.invalid_action_threshold = 0.2
        self.noop_loop_threshold = 3
        self.health_delta_weight = 2.0
        self.incident_resolution_weight = 1.5
        self.invalid_action_penalty = 0.5
        self.time_penalty_weight = 0.01
        self.stability_bonus_threshold = 10
        self.stability_bonus_reward = 2.0
        self.efficiency_bonus_reward = 0.5

        # Tracking for anti-cheat
        self.restart_counts: Dict[str, int] = {}
        self.action_history: List[str] = []
        self.last_health_state: Optional[float] = None
        self.stability_counter: int = 0

    def reset(self) -> None:
        """Reset tracking for new episode."""
        self.restart_counts = {}
        self.action_history = []
        self.last_health_state = None
        self.stability_counter = 0

    def compute_reward(
        self,
        action: Dict[str, Any],
        action_result: Dict[str, Any],
        current_health: float,
        previous_health: float,
        step_count: int,
        num_services: int,
        active_failures: Dict[str, List[Dict[str, Any]]],
        all_healthy: bool,
        any_critical_crashed: bool,
        diagnosed_root_cause: bool = False,
    ) -> float:
        """Compute reward for this step with structured shaping."""
        reward = 0.0

        # 1. MTTR Penalty (Exponential over time)
        # Higher penalty the longer the system stays unhealthy
        if not all_healthy:
            reward -= 0.01 * (1.1 ** step_count)

        # 2. Health delta (primary signal)
        health_delta = current_health - previous_health
        reward += health_delta * 4.0 # Increased weight

        # 3. Diagnostic & Root Cause Rewards
        if action_result.get("ok"):
            action_name = action.get("action")
            
            # Bonus for diagnosing root cause
            if diagnosed_root_cause and action_name in ["inspect_logs", "inspect_metrics"]:
                # Only reward the first time diagnosis happens or if continuing to monitor
                reward += 0.5
            
            # 4. Sequence-based Recovery Bonus
            # Example: Penalty for restarting before scaling if resource exhaustion
            if action_name == "restart_service":
                service_id = action.get("params", {}).get("service_id")
                failures = active_failures.get(service_id, [])
                has_resource_issue = any(f["type"] == "resource_exhaustion" for f in failures)
                
                if has_resource_issue and "allocate_resources" not in self.action_history:
                    reward -= 1.0 # Penalty for wrong sequence
                elif not has_resource_issue:
                    reward += 0.2 # Small bonus for valid restart

        else:
            # Penalty for invalid actions or failed recovery attempts
            reward -= self.invalid_action_penalty

        # 5. Anti-cheat & Spam Penalties
        action_name = action.get("action")
        self.action_history.append(action_name)
        
        # Penalize repetition
        if len(self.action_history) >= 2 and self.action_history[-1] == self.action_history[-2]:
            reward -= 0.3

        # 6. Terminal Rewards
        if all_healthy:
            # Success bonus scales inversely with steps taken
            reward += max(5.0, 10.0 - (step_count * 0.2))
        elif any_critical_crashed:
            reward -= 5.0 # Heavy failure penalty

        return reward

    def can_perform_action(
        self,
        action: Dict[str, Any],
        restart_counts: Dict[str, int],
        allocation_counts: Dict[str, int],
    ) -> tuple[bool, Optional[str]]:
        """Check if action can be performed (anti-cheat validation).

        Args:
            action: Action dict
            restart_counts: Per-service restart counts
            allocation_counts: Per-service allocation counts

        Returns:
            Tuple of (can_perform, error_code)
        """
        action_name = action.get("action")
        service_id = action.get("params", {}).get("service_id")

        # Restart limit check
        if action_name == "restart_service":
            if service_id and restart_counts.get(service_id, 0) >= self.restart_spam_threshold:
                return False, "RESTART_LIMIT_EXCEEDED"

        # Allocation limit check
        if action_name == "allocate_resources":
            if service_id and allocation_counts.get(service_id, 0) >= self.allocation_spam_threshold:
                return False, "ALLOCATION_LIMIT_EXCEEDED"

        return True, None

    def reset_service_counters(self) -> None:
        """Reset per-service counters for new episode."""
        self.restart_counts = {}
