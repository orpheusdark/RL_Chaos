"""TRL integration wrapper for ChaosOps-RC.

This module provides adapters to integrate ChaosOps with HuggingFace TRL.
"""

from typing import Any, Dict, Optional, Tuple, Callable

from envs import ChaosOpsRCEnv
from training.utils import format_observation_for_llm


class ChaosOpsRLAdapter:
    """Adapter to make ChaosOps compatible with TRL trainers."""

    def __init__(
        self,
        curriculum_level: int = 1,
        seed: Optional[int] = None,
        max_steps: int = 50,
    ):
        """Initialize adapter.

        Args:
            curriculum_level: Difficulty level
            seed: Random seed
            max_steps: Max steps per episode
        """
        self.env = ChaosOpsRCEnv(
            max_steps=max_steps,
            curriculum_level=curriculum_level,
            seed=seed,
        )
        self.current_observation = None

    def reset(self) -> str:
        """Reset environment and return initial observation as string.

        Returns:
            JSON string representation of initial observation
        """
        self.current_observation = self.env.reset()
        return format_observation_for_llm(self.current_observation)

    def step(self, action: Dict[str, Any]) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Execute one step.

        Args:
            action: Action dict {"action": "...", "params": {...}}

        Returns:
            Tuple of (observation_str, reward, done, info)
        """
        obs, reward, done, info = self.env.step(action)
        self.current_observation = obs

        return format_observation_for_llm(obs), reward, done, info

    def get_current_observation(self) -> str:
        """Get current observation as JSON string."""
        if self.current_observation is None:
            return self.reset()
        return format_observation_for_llm(self.current_observation)

    def is_done(self) -> bool:
        """Check if episode is done."""
        all_healthy = all(s.is_healthy() for s in self.env.services.values())
        any_critical_crashed = any(
            s.is_crashed() and s.service_id in ["db", "api", "gateway"]
            for s in self.env.services.values()
        )
        return all_healthy or any_critical_crashed or self.env.step_count >= self.env.max_steps


def make_chaosops_env(curriculum_level: int = 1, **kwargs) -> ChaosOpsRLAdapter:
    """Factory function to create ChaosOps environment for TRL.

    Compatible with gym.make() interface.

    Args:
        curriculum_level: Difficulty level
        **kwargs: Additional arguments (seed, max_steps, etc.)

    Returns:
        ChaosOpsRLAdapter instance
    """
    return ChaosOpsRLAdapter(curriculum_level=curriculum_level, **kwargs)
