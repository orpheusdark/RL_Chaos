import argparse
import json
import os
import random
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

from envs.multi_service_env import ChaosOpsRCEnv

# Action space mapping
ACTIONS = [
    "inspect_logs",
    "restart_service",
    "rollback_service",
    "patch_config",
    "allocate_resources",
    "inspect_metrics",
    "drain_requests",
    "promote_replica",
]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}

class PolicyValueNet(nn.Module):
    def __init__(self, input_dim: int, action_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

def encode_observation(obs: Dict[str, Any]) -> torch.Tensor:
    """Encode observation dict into a fixed-size vector."""
    metrics = obs.get("metrics", {})
    topology = obs.get("topology", {})
    services = topology.get("services", [])
    
    # 1. Health metrics (normalized)
    health_vec = []
    for sid in ["gateway", "api", "auth", "payment", "db"]:
        health_vec.append(metrics.get(f"{sid}_health", 1.0))
    
    # 2. System health
    sys_health = [metrics.get("system_health", 1.0)]
    
    # 3. Step context
    step_info = [obs.get("step", 0) / 30.0]
    
    # 4. Alerts multi-hot (simplified)
    alert_vec = [0.0] * 5
    for alert in obs.get("alerts", []):
        if alert["severity"] == "critical":
            alert_vec[0] = 1.0
        elif alert["severity"] == "high":
            alert_vec[1] = 1.0
            
    # Combine
    vec = health_vec + sys_health + step_info + alert_vec
    return torch.tensor(vec, dtype=torch.float32)

class PPOAgent:
    def __init__(self, input_dim: int, action_dim: int, lr: float = 3e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PolicyValueNet(input_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.clip_eps = 0.2
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.entropy_coef = 0.01

    def select_action(self, obs_tensor: torch.Tensor) -> Tuple[int, float, float]:
        obs_tensor = obs_tensor.to(self.device).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.policy(obs_tensor)
            dist = Categorical(logits=logits)
            action = dist.sample()
            return action.item(), dist.log_prob(action).item(), value.item()

    def update(self, rollouts: List[Dict[str, Any]]):
        if not rollouts:
            return
            
        # Flatten rollouts
        states = torch.stack([r["state"] for r in rollouts]).to(self.device)
        actions = torch.tensor([r["action"] for r in rollouts]).to(self.device)
        old_log_probs = torch.tensor([r["log_prob"] for r in rollouts]).to(self.device)
        rewards = torch.tensor([r["reward"] for r in rollouts]).to(self.device)
        values = torch.tensor([r["value"] for r in rollouts]).to(self.device)
        dones = torch.tensor([r["done"] for r in rollouts]).to(self.device)

        # Compute GAE
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        for t in reversed(range(len(rewards))):
            next_mask = (1 - dones[t])
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t+1]
            
            delta = rewards[t] + self.gamma * next_value * next_mask - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_mask * last_gae
            
        returns = advantages + values
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO Update
        for _ in range(5): # Epochs
            logits, current_values = self.policy(states)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (returns - current_values).pow(2).mean()
            
            loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

def evaluate(env, agent, num_episodes=20):
    all_rewards = []
    successes = 0
    mttr = []
    
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        steps = 0
        while not done:
            obs_vec = encode_observation(obs)
            action_idx, _, _ = agent.select_action(obs_vec)
            
            # Map action idx to complex action dict
            # For simplicity, we target a random service or the first degraded one
            services = obs["topology"]["services"]
            target_service = services[0]
            for s in services:
                if obs["metrics"].get(f"{s}_health", 1.0) < 0.8:
                    target_service = s
                    break
                    
            action = {"action": ACTIONS[action_idx], "params": {"service_id": target_service}}
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            steps += 1
            
        all_rewards.append(ep_reward)
        if all(obs["metrics"].get(f"{s}_health", 0.0) >= 0.8 for s in services):
             successes += 1
             mttr.append(steps)
             
    return {
        "mean_reward": np.mean(all_rewards),
        "success_rate": successes / num_episodes,
        "mean_mttr": np.mean(mttr) if mttr else 30
    }

def run_all():
    print("Initializing ChaosOps RL Pipeline...")
    env = ChaosOpsRCEnv(curriculum_level=2) # Moderate difficulty
    input_dim = 12 # From encode_observation
    agent = PPOAgent(input_dim, len(ACTIONS))
    
    # 1. Baseline Eval
    print("Evaluating Baseline (Random Policy)...")
    baseline_stats = evaluate(env, PPOAgent(input_dim, len(ACTIONS)), num_episodes=50)
    print(f"Baseline: {baseline_stats}")
    
    # 2. Training
    print("Starting PPO Training...")
    history = []
    for ep in range(100):
        obs = env.reset()
        done = False
        ep_rollouts = []
        total_r = 0
        while not done:
            obs_vec = encode_observation(obs)
            action_idx, log_prob, value = agent.select_action(obs_vec)
            
            services = obs["topology"]["services"]
            target_service = services[0]
            for s in services:
                if obs["metrics"].get(f"{s}_health", 1.0) < 0.8:
                    target_service = s
                    break
            
            action = {"action": ACTIONS[action_idx], "params": {"service_id": target_service}}
            next_obs, reward, done, _ = env.step(action)
            
            ep_rollouts.append({
                "state": obs_vec,
                "action": action_idx,
                "log_prob": log_prob,
                "reward": reward,
                "value": value,
                "done": float(done)
            })
            obs = next_obs
            total_r += reward
            
        agent.update(ep_rollouts)
        history.append(total_r)
        if ep % 10 == 0:
            print(f"Episode {ep}: Reward = {total_r:.2f}")
            
    # 3. Trained Eval
    print("Evaluating Trained Agent...")
    trained_stats = evaluate(env, agent, num_episodes=50)
    print(f"Trained: {trained_stats}")
    
    # 4. Plotting
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(history)
    plt.title("Training Reward Curve")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig("results/reward_curve.png")
    
    # Save metrics
    metrics = {
        "baseline": baseline_stats,
        "trained": trained_stats,
        "improvement": (trained_stats["success_rate"] - baseline_stats["success_rate"])
    }
    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    print("Pipeline Complete. Results saved to /results/")

if __name__ == "__main__":
    run_all()