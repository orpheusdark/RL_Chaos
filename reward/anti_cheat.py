"""Anti-exploitation detection for ChaosOps-RC."""

from typing import Dict, List, Set


class AntiCheatDetector:
    """Detects and prevents agent exploitation of the reward function."""

    def __init__(self):
        """Initialize anti-cheat detector."""
        self.action_sequences: List[str] = []
        self.restart_patterns: Dict[str, List[int]] = {}
        self.allocation_patterns: Dict[str, List[int]] = {}
        self.max_history = 100

    def reset(self) -> None:
        """Reset for new episode."""
        self.action_sequences = []
        self.restart_patterns = {}
        self.allocation_patterns = {}

    def record_action(self, action_name: str) -> None:
        """Record an action."""
        self.action_sequences.append(action_name)
        if len(self.action_sequences) > self.max_history:
            self.action_sequences.pop(0)

    def record_restart(self, service_id: str, step_count: int) -> None:
        """Record a restart action."""
        if service_id not in self.restart_patterns:
            self.restart_patterns[service_id] = []
        self.restart_patterns[service_id].append(step_count)

    def record_allocation(self, service_id: str, step_count: int) -> None:
        """Record a resource allocation."""
        if service_id not in self.allocation_patterns:
            self.allocation_patterns[service_id] = []
        self.allocation_patterns[service_id].append(step_count)

    def detect_restart_spam(self, service_id: str) -> bool:
        """Detect if agent is spamming restart on a service."""
        if service_id not in self.restart_patterns:
            return False

        restarts = self.restart_patterns[service_id]
        if len(restarts) <= 2:
            return False

        # Check if restarts are happening frequently (close together)
        recent_restarts = restarts[-3:]
        intervals = [recent_restarts[i + 1] - recent_restarts[i] for i in range(len(recent_restarts) - 1)]

        # If all recent restarts are 1 step apart, it's spam
        return all(interval <= 2 for interval in intervals)

    def detect_allocation_spam(self, service_id: str) -> bool:
        """Detect if agent is spamming resource allocation."""
        if service_id not in self.allocation_patterns:
            return False

        allocations = self.allocation_patterns[service_id]
        if len(allocations) <= 1:
            return False

        # If allocating to same service more than twice, it's spam
        return len(allocations) > self.allocation_spam_threshold

    def detect_action_repetition(self, window_size: int = 5) -> bool:
        """Detect if agent is repeating same action too much."""
        if len(self.action_sequences) < window_size:
            return False

        recent = self.action_sequences[-window_size:]
        return len(set(recent)) == 1

    def detect_noop_loop(self, window_size: int = 10) -> bool:
        """Detect if agent is stuck in a no-op loop (inspection actions only)."""
        if len(self.action_sequences) < window_size:
            return False

        recent = self.action_sequences[-window_size:]
        noop_actions = {"inspect_logs", "inspect_metrics", "query_system"}
        noop_count = sum(1 for action in recent if action in noop_actions)

        # If 80%+ are noop, agent is stuck
        return noop_count / window_size > 0.8

    def detect_invalid_action_spam(self, window_size: int = 20) -> float:
        """Detect rate of invalid actions.

        Returns:
            Fraction of invalid actions in recent history
        """
        if len(self.action_sequences) < window_size:
            return 0.0

        recent = self.action_sequences[-window_size:]
        invalid_count = sum(1 for action in recent if action in ["INVALID_ACTION", "UNKNOWN_ACTION"])

        return invalid_count / window_size

    def get_exploitation_score(self) -> float:
        """Get overall exploitation score [0, 1].

        Returns:
            0 = no exploitation detected
            1 = severe exploitation
        """
        score = 0.0

        # Action repetition
        if self.detect_action_repetition():
            score += 0.2

        # No-op loop
        if self.detect_noop_loop():
            score += 0.3

        # Invalid action spam
        invalid_rate = self.detect_invalid_action_spam()
        score += invalid_rate * 0.3

        # Restart spam on any service
        for service_id in self.restart_patterns:
            if self.detect_restart_spam(service_id):
                score += 0.2
                break

        return min(score, 1.0)

    # Configuration
    allocation_spam_threshold = 2
