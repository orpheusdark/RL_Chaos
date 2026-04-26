"""Evaluation utilities for ChaosOps-RC."""

import random
from typing import Any, Dict, List, Optional, Tuple

from envs import ChaosOpsRCEnv


class BaselineAgent:
    """Baseline agent for evaluation (random actions)."""

    def __init__(self):
        """Initialize baseline agent."""
        self.action_names = [
            "inspect_logs",
            "restart_service",
            "inspect_metrics",
            "allocate_resources",
            "rollback_service",
        ]

    def get_action(self, observation: Dict[str, Any], services: List[str]) -> Dict[str, Any]:
        """Get next action (random).

        Args:
            observation: Current observation
            services: List of available services

        Returns:
            Action dict
        """
        action_name = random.choice(self.action_names)
        service_id = random.choice(services)

        action = {"action": action_name, "params": {"service_id": service_id}}

        if action_name == "allocate_resources":
            action["params"]["cpu"] = random.randint(100, 500)
            action["params"]["memory"] = random.randint(256, 1024)
        elif action_name == "patch_config":
            action["params"]["patch"] = {"timeout": random.randint(1, 10)}

        return action


def evaluate_agent(
    agent_policy,
    num_episodes: int = 50,
    curriculum_level: int = 1,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate an agent on ChaosOps environment.

    Args:
        agent_policy: Agent that has get_action(obs, services) method
        num_episodes: Number of episodes to run
        curriculum_level: Difficulty level
        seed: Random seed

    Returns:
        Evaluation metrics dict
    """
    env = ChaosOpsRCEnv(
        curriculum_level=curriculum_level,
        seed=seed,
    )

    rewards = []
    lengths = []
    successes = 0
    failures = 0

    for episode_num in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        steps = 0

        while not done and steps < env.max_steps:
            # Get action from agent
            action = agent_policy.get_action(obs, list(env.services.keys()))

            # Step environment
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1

        # Record episode result
        rewards.append(episode_reward)
        lengths.append(steps)

        # Check success
        all_healthy = all(s.is_healthy() for s in env.services.values())
        any_critical_crashed = any(
            s.is_crashed() and s.service_id in ["db", "api", "gateway"]
            for s in env.services.values()
        )

        if all_healthy:
            successes += 1
        elif any_critical_crashed:
            failures += 1

    # Compute statistics
    stats = {
        "num_episodes": num_episodes,
        "mean_reward": sum(rewards) / len(rewards),
        "max_reward": max(rewards) if rewards else 0,
        "min_reward": min(rewards) if rewards else 0,
        "std_reward": (
            (sum((r - sum(rewards) / len(rewards)) ** 2 for r in rewards) / len(rewards)) ** 0.5
            if rewards
            else 0
        ),
        "mean_episode_length": sum(lengths) / len(lengths) if lengths else 0,
        "success_rate": successes / num_episodes,
        "failure_rate": failures / num_episodes,
        "curriculum_level": curriculum_level,
        "rewards": rewards,
        "lengths": lengths,
    }

    return stats


def compare_agents(
    baseline_agent,
    trained_agent,
    num_episodes: int = 50,
    curriculum_level: int = 1,
) -> Dict[str, Any]:
    """Compare two agents head-to-head.

    Args:
        baseline_agent: Baseline agent
        trained_agent: Trained agent
        num_episodes: Episodes per agent
        curriculum_level: Difficulty level

    Returns:
        Comparison dict with both agents' stats
    """
    print("Evaluating baseline agent...")
    baseline_stats = evaluate_agent(
        baseline_agent,
        num_episodes=num_episodes,
        curriculum_level=curriculum_level,
        seed=42,
    )

    print("Evaluating trained agent...")
    trained_stats = evaluate_agent(
        trained_agent,
        num_episodes=num_episodes,
        curriculum_level=curriculum_level,
        seed=42,
    )

    return {
        "baseline": baseline_stats,
        "trained": trained_stats,
        "improvement": {
            "reward_delta": trained_stats["mean_reward"] - baseline_stats["mean_reward"],
            "success_rate_delta": trained_stats["success_rate"] - baseline_stats["success_rate"],
            "efficiency_delta": (
                baseline_stats["mean_episode_length"] - trained_stats["mean_episode_length"]
            ),
        },
    }
