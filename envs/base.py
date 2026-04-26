"""Base OpenEnv interface for ChaosOps-RC."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple


class OpenEnvEnv(ABC):
    """Abstract base class for OpenEnv-compliant environments."""

    @abstractmethod
    def reset(self, **kwargs) -> Dict[str, Any]:
        """Reset the environment.

        Returns:
            observation: Initial observation dict
        """
        pass

    @abstractmethod
    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute one step of the environment.

        Args:
            action: Action dict with "action" and "params" keys

        Returns:
            Tuple of (observation, reward, terminated, info)
        """
        pass

    @abstractmethod
    def get_observation(self) -> Dict[str, Any]:
        """Get current observation (partial view of state)."""
        pass
