"""Training utilities for TRL integration."""

import json
from typing import Any, Dict, List, Optional, Tuple

from envs import ChaosOpsRCEnv


class EpisodeCollector:
    """Collects episodes from ChaosOps environment."""

    def __init__(self, curriculum_level: int = 1, seed: Optional[int] = None):
        """Initialize collector.

        Args:
            curriculum_level: Difficulty level
            seed: Random seed
        """
        self.env = ChaosOpsRCEnv(
            max_steps=50,
            curriculum_level=curriculum_level,
            seed=seed,
        )
        self.episode_buffer: List[Dict[str, Any]] = []

    def collect_episodes(self, num_episodes: int) -> List[Dict[str, Any]]:
        """Collect num_episodes from the environment.

        Returns:
            List of episode dicts, each with trajectory of steps
        """
        episodes = []

        for _ in range(num_episodes):
            episode = self._collect_single_episode()
            episodes.append(episode)
            self.episode_buffer.append(episode)

        return episodes

    def _collect_single_episode(self) -> Dict[str, Any]:
        """Collect one complete episode.

        Returns:
            Episode dict with trajectory, rewards, actions
        """
        obs = self.env.reset()
        trajectory = []
        done = False
        step_count = 0

        while not done and step_count < self.env.max_steps:
            # In real training, agent would choose action
            # For now, use random action for testing
            action = self._random_action()

            # Step environment
            new_obs, reward, done, info = self.env.step(action)

            # Record step
            trajectory.append({
                "observation": obs,
                "action": action,
                "reward": reward,
                "next_observation": new_obs,
                "done": done,
                "info": info,
            })

            obs = new_obs
            step_count += 1

        return {
            "trajectory": trajectory,
            "total_reward": self.env.episode_reward,
            "length": len(trajectory),
            "curriculum_level": self.env.curriculum_level,
        }

    def _random_action(self) -> Dict[str, Any]:
        """Sample a random valid action."""
        import random

        action_names = ["inspect_logs", "restart_service", "inspect_metrics", "allocate_resources"]
        action_name = random.choice(action_names)

        service_ids = list(self.env.services.keys())
        service_id = random.choice(service_ids)

        action = {"action": action_name, "params": {"service_id": service_id}}

        if action_name == "allocate_resources":
            action["params"]["cpu"] = random.randint(100, 500)
            action["params"]["memory"] = random.randint(256, 1024)

        return action

    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get statistics on collected episodes."""
        if not self.episode_buffer:
            return {}

        rewards = [ep["total_reward"] for ep in self.episode_buffer]
        lengths = [ep["length"] for ep in self.episode_buffer]

        return {
            "num_episodes": len(self.episode_buffer),
            "avg_reward": sum(rewards) / len(rewards),
            "max_reward": max(rewards),
            "min_reward": min(rewards),
            "avg_length": sum(lengths) / len(lengths),
            "total_steps": sum(lengths),
        }


def format_observation_for_llm(obs: Dict[str, Any]) -> str:
    """Format observation as JSON string for LLM agent.

    Args:
        obs: Observation from environment

    Returns:
        JSON string representation
    """
    return json.dumps(obs, indent=2)


def parse_llm_action(llm_output: str) -> Dict[str, Any]:
    """Parse LLM output into action dict.

    Expects JSON format:
    {
        "action": "restart_service",
        "params": {"service_id": "auth"}
    }

    Args:
        llm_output: LLM model output

    Returns:
        Action dict, or default invalid action if parsing fails
    """
    try:
        action = json.loads(llm_output)
        if "action" in action and "params" in action:
            return action
    except (json.JSONDecodeError, ValueError):
        pass

    # Default invalid action
    return {"action": "invalid_action", "params": {}}


def batch_generator(
    episodes: List[Dict[str, Any]],
    batch_size: int = 16,
) -> List[List[Dict[str, Any]]]:
    """Group episodes into batches for training.

    Args:
        episodes: List of episode dicts
        batch_size: Batch size

    Yields:
        Batches of episodes
    """
    for i in range(0, len(episodes), batch_size):
        yield episodes[i : i + batch_size]
