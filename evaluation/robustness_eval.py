"""Robustness evaluation suite for ChaosOps-RC.

This module implements a multi-tier adversarial evaluation framework that tests
whether a trained policy still works when reality stops cooperating.
"""

import random
from dataclasses import dataclass, field
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Tuple

from envs import ChaosOpsRCEnv
from .eval_script import BaselineAgent


@dataclass
class PerturbationConfig:
    obs_noise_level: float = 0.0
    obs_mask_rate: float = 0.0
    obs_delay_steps: int = 0
    action_failure_rate: float = 0.0
    delayed_action_rate: float = 0.0
    reward_scale_mid_episode: Optional[float] = None
    reward_shift_step: Optional[int] = None
    drop_tools: int = 0
    structural_shift: bool = False
    worst_case_injection: bool = False
    hidden_reward_bias: float = 0.0
    randomize_seed: bool = True


@dataclass
class EvaluationTier:
    name: str
    description: str
    config: PerturbationConfig


@dataclass
class TierResult:
    name: str
    config: PerturbationConfig
    stats: Dict[str, Any] = field(default_factory=dict)


class RobustnessEnvWrapper:
    """Wraps ChaosOpsRCEnv to apply evaluation perturbations."""

    def __init__(self, env: ChaosOpsRCEnv, rng: random.Random, config: PerturbationConfig):
        self.env = env
        self.rng = rng
        self.config = config
        self.step_count = 0
        self.pending_actions: List[Dict[str, Any]] = []
        self.current_reward_scale = 1.0
        self.hidden_reward_bias = config.hidden_reward_bias
        self._initialized = False

    @property
    def services(self) -> Dict[str, Any]:
        return self.env.services

    @property
    def max_steps(self) -> int:
        return self.env.max_steps

    def _is_terminal(self) -> bool:
        """Check if episode is terminal by delegating to the underlying environment."""
        return self.env._is_terminal()

    def reset(self) -> Dict[str, Any]:
        if self.config.randomize_seed:
            self.env.rng = random.Random(self.rng.randint(0, 2 ** 32 - 1))
        observation = self.env.reset()
        self.step_count = 0
        self.pending_actions = []
        self.current_reward_scale = 1.0
        self.hidden_reward_bias = self.config.hidden_reward_bias
        self._apply_structural_shifts()
        if self.config.worst_case_injection:
            self._apply_worst_case_injection()
        observation = self._corrupt_observation(observation)
        self._initialized = True
        return observation

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        self.step_count += 1

        # Handle delayed execution queue first
        if self.pending_actions:
            for queued in self.pending_actions:
                queued["delay"] -= 1
            ready = [q for q in self.pending_actions if q["delay"] <= 0]
            self.pending_actions = [q for q in self.pending_actions if q["delay"] > 0]
            if ready:
                action = ready[0]["action"]

        previous_health = self.env.system_graph.compute_system_health() if self.env.system_graph else 0.0

        # Update failures even when the current action is delayed or fails
        if self.env.failure_injector:
            self.env.failure_injector.update_failures(self.step_count)

        action_result = None
        executed_action = action
        if self.rng.random() < self.config.action_failure_rate:
            # Simulate silent or explicit failure
            if self.rng.random() < 0.5:
                action_result = {"ok": True, "failed_silently": True}
            else:
                action_result = {"ok": False, "error_code": "ACTION_FAILED"}
            executed_action = None

        if self.rng.random() < self.config.delayed_action_rate and executed_action is not None:
            delay = self.rng.randint(1, max(1, self.config.obs_delay_steps or 1))
            self.pending_actions.append({"action": executed_action, "delay": delay})
            action_result = {"ok": True, "delayed": True, "delay": delay}
            executed_action = None

        if executed_action is not None and action_result is None:
            action_result = self.env._execute_action(executed_action)

        # Apply recovery effects and pending action propagation
        self._process_pending_effects()

        reward = self.env._compute_reward(
            executed_action if executed_action is not None else action,
            action_result,
            previous_health,
            self.env.diagnosed_root_cause,
        )
        reward = self._distort_reward(reward)
        self.env.episode_reward += reward

        terminated = self.step_count >= self.env.max_steps or self.env._is_terminal()

        observation = self._corrupt_observation(self.env.get_observation())
        info = {
            "action_result": action_result,
            "previous_health": previous_health,
            "tier_step": self.step_count,
        }
        return observation, reward, terminated, info

    def _process_pending_effects(self) -> None:
        for service in self.env.services.values():
            remaining_effects = []
            for effect in service.pending_effects:
                effect["delay"] -= 1
                if effect["delay"] <= 0:
                    if effect["type"] == "recovery":
                        service.restore_health(0.6)
                        self.env._add_log(service.service_id, "info", "Recovery effect completed")
                else:
                    remaining_effects.append(effect)
            service.pending_effects = remaining_effects

    def _distort_reward(self, reward: float) -> float:
        if self.config.reward_shift_step and self.step_count >= self.config.reward_shift_step:
            self.current_reward_scale = self.config.reward_scale_mid_episode or 1.0
        return reward * self.current_reward_scale + self.hidden_reward_bias

    def _corrupt_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        corrupted = {**observation}
        corrupted_metrics = dict(observation["metrics"])

        for key, value in observation["metrics"].items():
            if isinstance(value, float) and self.rng.random() < self.config.obs_noise_level:
                noise = self.rng.uniform(-self.config.obs_noise_level, self.config.obs_noise_level)
                corrupted_metrics[key] = max(0.0, min(1.0, value + noise))

        corrupted["metrics"] = corrupted_metrics

        # Mask logs and alerts
        if self.config.obs_mask_rate > 0:
            corrupted["logs"] = [log for log in corrupted["logs"] if self.rng.random() > self.config.obs_mask_rate]
            corrupted["alerts"] = [alert for alert in corrupted["alerts"] if self.rng.random() > self.config.obs_mask_rate]

        # Simulate delayed observations by occasionally returning stale step index and previous slices
        if self.config.obs_delay_steps > 0 and self.rng.random() < 0.2:
            corrupted["step"] = max(0, observation["step"] - self.rng.randint(1, self.config.obs_delay_steps))

        if self.config.worst_case_injection and self.rng.random() < 0.2:
            corrupted["logs"] = corrupted["logs"] + [
                {"timestamp": observation["step"], "service": self.rng.choice(list(self.env.services.keys())), "level": "error", "message": "Misleading symptom detected"}
            ]

        return corrupted

    def _apply_structural_shifts(self) -> None:
        if not self.config.structural_shift:
            return

        drop_count = min(self.config.drop_tools, len(self.env.valid_actions) - 1)
        if drop_count > 0:
            to_drop = self.rng.sample(list(self.env.valid_actions), drop_count)
            for action_name in to_drop:
                self.env.valid_actions.discard(action_name)

        # Randomly remove some dependencies to create runtime layout variation.
        candidate_services = [s for s in self.env.services.values() if s.dependencies]
        if candidate_services:
            target = self.rng.choice(candidate_services)
            if target.dependencies:
                removed = self.rng.choice(target.dependencies)
                target.dependencies = [d for d in target.dependencies if d != removed]
                self.env.system_graph._compute_reverse_deps()
                self.env._add_log(target.service_id, "warn", f"Runtime dependency {removed} removed")

    def _apply_worst_case_injection(self) -> None:
        if not self.config.worst_case_injection or not self.env.failure_injector:
            return

        critical_services = [sid for sid in self.env.services if sid in ["api", "db", "gateway"]]
        if len(critical_services) >= 2:
            failed = self.rng.sample(critical_services, 2)
            for sid in failed:
                self.env.failure_injector.inject_failure(
                    service_id=sid,
                    failure_type=self.rng.choice(self.env.failure_types),
                    logs_callback=self.env._add_log,
                    alerts_callback=self.env._add_alert,
                    step_count=self.step_count,
                )
            self.env._add_log(failed[0], "error", "Worst-case fault chain started")


def get_tier_definitions() -> List[EvaluationTier]:
    return [
        EvaluationTier(
            name="Tier 0 - Sanity",
            description="Deterministic baseline check on the clean environment.",
            config=PerturbationConfig(),
        ),
        EvaluationTier(
            name="Tier 1 - Noisy realism",
            description="Mild observation noise and light action failures.",
            config=PerturbationConfig(
                obs_noise_level=0.08,
                obs_mask_rate=0.1,
                obs_delay_steps=1,
                action_failure_rate=0.1,
                delayed_action_rate=0.1,
                reward_scale_mid_episode=1.0,
                reward_shift_step=None,
            ),
        ),
        EvaluationTier(
            name="Tier 2 - Adversarial stress",
            description="Stronger tool failure, reward drift, and schema perturbation.",
            config=PerturbationConfig(
                obs_noise_level=0.18,
                obs_mask_rate=0.4,
                obs_delay_steps=2,
                action_failure_rate=0.25,
                delayed_action_rate=0.2,
                reward_scale_mid_episode=0.8,
                reward_shift_step=8,
                drop_tools=1,
                structural_shift=True,
                hidden_reward_bias=-0.02,
            ),
        ),
        EvaluationTier(
            name="Tier 3 - Distribution shift",
            description="Service graph shifts, tools removed, and reward weights change.",
            config=PerturbationConfig(
                obs_noise_level=0.25,
                obs_mask_rate=0.5,
                obs_delay_steps=3,
                action_failure_rate=0.3,
                delayed_action_rate=0.25,
                reward_scale_mid_episode=1.2,
                reward_shift_step=6,
                drop_tools=2,
                structural_shift=True,
                hidden_reward_bias=-0.05,
            ),
        ),
        EvaluationTier(
            name="Tier 4 - Worst-case injection",
            description="Cascading failures, poisoned signals, and forced recovery loops.",
            config=PerturbationConfig(
                obs_noise_level=0.3,
                obs_mask_rate=0.5,
                obs_delay_steps=4,
                action_failure_rate=0.3,
                delayed_action_rate=0.3,
                reward_scale_mid_episode=0.9,
                reward_shift_step=5,
                drop_tools=2,
                structural_shift=True,
                worst_case_injection=True,
                hidden_reward_bias=-0.1,
            ),
        ),
    ]


class RobustnessEvaluator:
    """Runs tiered evaluations and computes robustness metrics."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.tiers = get_tier_definitions()

    def evaluate_agent(
        self,
        agent_policy,
        num_episodes: int = 30,
        curriculum_level: int = 2,
        chaos_holdout: bool = False,
    ) -> Dict[str, Any]:
        results = {}
        for tier in self.tiers:
            if chaos_holdout and tier.name == "Tier 0 - Sanity":
                continue
            results[tier.name] = self._evaluate_tier(
                agent_policy,
                tier,
                num_episodes=num_episodes,
                curriculum_level=curriculum_level,
            )
        summary = self._compute_suite_summary(results)
        return {"tiers": results, "summary": summary}

    def _evaluate_tier(
        self,
        agent_policy,
        tier: EvaluationTier,
        num_episodes: int,
        curriculum_level: int,
    ) -> Dict[str, Any]:
        rewards: List[float] = []
        lengths: List[int] = []
        successes = 0
        mttrs: List[float] = []
        tail_rewards: List[float] = []

        for episode in range(num_episodes):
            seed = self.rng.randint(0, 2 ** 32 - 1)
            base_env = ChaosOpsRCEnv(curriculum_level=curriculum_level, seed=seed)
            env = RobustnessEnvWrapper(base_env, random.Random(seed), tier.config)
            obs = env.reset()
            done = False
            episode_reward = 0.0
            steps = 0
            recovery_step = None

            while not done and steps < env.max_steps:
                action = agent_policy.get_action(obs, list(env.services.keys()))
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                steps += 1
                if recovery_step is None and env._is_terminal():
                    if all(s.is_healthy() for s in env.services.values()):
                        recovery_step = steps

            rewards.append(episode_reward)
            lengths.append(steps)
            tail_rewards.append(episode_reward)

            if all(s.is_healthy() for s in env.services.values()):
                successes += 1
                mttrs.append(recovery_step or steps)
            else:
                mttrs.append(env.max_steps)

        return {
            "num_episodes": num_episodes,
            "mean_reward": mean(rewards) if rewards else 0.0,
            "std_reward": stdev(rewards) if len(rewards) > 1 else 0.0,
            "mean_episode_length": mean(lengths) if lengths else 0.0,
            "success_rate": successes / num_episodes if num_episodes else 0.0,
            "mean_mttr": mean(mttrs) if mttrs else 0.0,
            "std_mttr": stdev(mttrs) if len(mttrs) > 1 else 0.0,
            "worst_10pct_reward": self._worst_quantile(rewards, 0.1),
            "worst_10pct_mttr": self._worst_quantile(mttrs, 0.1),
            "graceful_degradation_score": self._compute_graceful_score(rewards, mttrs),
            "config": tier.config.__dict__,
        }

    def _worst_quantile(self, values: List[float], quantile: float) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = max(0, int(len(sorted_values) * quantile) - 1)
        return sorted_values[index]

    def _compute_graceful_score(self, rewards: List[float], mttrs: List[float]) -> float:
        if not rewards or not mttrs:
            return 0.0
        reward_std = stdev(rewards) if len(rewards) > 1 else 0.0
        mttr_std = stdev(mttrs) if len(mttrs) > 1 else 0.0
        score = 1.0 - min(1.0, (reward_std + mttr_std) / (abs(mean(rewards)) + 1.0))
        return max(0.0, min(1.0, score))

    def _compute_suite_summary(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        curve = [
            {
                "tier": tier_name,
                "success_rate": data["success_rate"],
                "mean_reward": data["mean_reward"],
                "mean_mttr": data["mean_mttr"],
            }
            for tier_name, data in results.items()
        ]
        return {
            "curve": curve,
            "overall_success_rate": mean([data["success_rate"] for data in results.values()]) if results else 0.0,
            "overall_mean_reward": mean([data["mean_reward"] for data in results.values()]) if results else 0.0,
            "overall_mean_mttr": mean([data["mean_mttr"] for data in results.values()]) if results else 0.0,
        }


def run_holdout_chaos_set(agent_policy, num_episodes: int = 20, curriculum_level: int = 2) -> Dict[str, Any]:
    evaluator = RobustnessEvaluator()
    return evaluator.evaluate_agent(agent_policy, num_episodes=num_episodes, curriculum_level=curriculum_level, chaos_holdout=True)


def benchmark_baseline(num_episodes: int = 30, curriculum_level: int = 2) -> Dict[str, Any]:
    baseline = BaselineAgent()
    evaluator = RobustnessEvaluator()
    return evaluator.evaluate_agent(baseline, num_episodes=num_episodes, curriculum_level=curriculum_level)
