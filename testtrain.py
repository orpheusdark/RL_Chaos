import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque

from envs.multi_service_env import ChaosOpsRCEnv

# -----------------------------
# ACTION SPACE (FULL CONTROL)
# -----------------------------
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

SERVICES = ["gateway", "api", "auth", "payment", "db"]

ACTION_DIM = len(ACTIONS)
SERVICE_DIM = len(SERVICES)

# -----------------------------
# OBS ENCODING
# -----------------------------

def encode_observation(obs):
    metrics = obs.get("metrics", {})

    vec = []

    # service health
    for s in SERVICES:
        vec.append(metrics.get(f"{s}_health", 1.0))

    # system health
    vec.append(metrics.get("system_health", 1.0))

    # timestep
    vec.append(obs.get("step", 0) / 30.0)

    # alerts
    critical = 0
    high = 0
    for a in obs.get("alerts", []):
        if a.get("severity") == "critical":
            critical = 1
        if a.get("severity") == "high":
            high = 1

    vec.extend([critical, high])

    return torch.tensor(vec, dtype=torch.float32)

INPUT_DIM = len(SERVICES) + 3

# -----------------------------
# MODEL
# -----------------------------

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(INPUT_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.action_head = nn.Linear(128, ACTION_DIM * SERVICE_DIM)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        h = self.base(x)
        logits = self.action_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value

    def get_dist(self, x):
        logits, value = self.forward(x)
        logits = logits.view(-1, ACTION_DIM, SERVICE_DIM)
        return logits, value

# -----------------------------
# PPO AGENT
# -----------------------------

class PPO:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4)

        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip = 0.2

    def act(self, obs):
        obs = obs.to(self.device).unsqueeze(0)

        with torch.no_grad():
            logits, value = self.model.get_dist(obs)

            logits = logits.squeeze(0)

            action_dist = Categorical(logits=logits.view(-1))
            idx = action_dist.sample()

            a = idx // SERVICE_DIM
            s = idx % SERVICE_DIM

            log_prob = action_dist.log_prob(idx)

        return (a.item(), s.item()), log_prob.item(), value.item()

    def compute_gae(self, rewards, values, dones):
        adv = []
        gae = 0
        values = values + [0]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            adv.insert(0, gae)

        return torch.tensor(adv, dtype=torch.float32)

    def update(self, batch):
        states = torch.stack(batch["states"]).to(self.device)
        actions = torch.tensor(batch["actions"]).to(self.device)
        old_log_probs = torch.tensor(batch["log_probs"]).to(self.device)
        rewards = batch["rewards"]
        values = batch["values"]
        dones = batch["dones"]

        advantages = self.compute_gae(rewards, values, dones)
        returns = advantages + torch.tensor(values[:-1])

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(5):
            logits, v = self.model(states)

            logits = logits.view(-1, ACTION_DIM * SERVICE_DIM)

            dist = Categorical(logits=logits)

            new_log_probs = dist.log_prob(actions)

            ratio = torch.exp(new_log_probs - old_log_probs)

            s1 = ratio * advantages
            s2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantages

            policy_loss = -torch.min(s1, s2).mean()
            value_loss = (returns - v.squeeze()).pow(2).mean()

            loss = policy_loss + 0.5 * value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

# -----------------------------
# RANDOM BASELINE
# -----------------------------

class RandomPolicy:
    def act(self, obs):
        return (
            random.randint(0, ACTION_DIM - 1),
            random.randint(0, SERVICE_DIM - 1)
        ), 0.0, 0.0

# -----------------------------
# TRAIN / EVAL
# -----------------------------

def evaluate(env, agent, episodes=20):
    rewards = []
    success = 0

    for _ in range(episodes):
        obs = env.reset()
        done = False
        total = 0

        while not done:
            o = encode_observation(obs)
            (a, s), _, _ = agent.act(o)

            action = {
                "action": ACTIONS[a],
                "params": {"service_id": SERVICES[s]}
            }

            obs, r, done, _ = env.step(action)
            total += r

        rewards.append(total)

        if obs["metrics"].get("system_health", 0) > 0.8:
            success += 1

    return {
        "mean_reward": float(np.mean(rewards)),
        "success_rate": success / episodes
    }

# -----------------------------
# TRAIN LOOP
# -----------------------------

def train():
    env = ChaosOpsRCEnv(curriculum_level=2)
    agent = PPO()

    print("Baseline...")
    baseline = evaluate(env, RandomPolicy())

    print("Training...")

    history = []

    for ep in range(100):
        obs = env.reset()
        done = False

        batch = {
            "states": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "values": [],
            "dones": []
        }

        total = 0

        while not done:
            o = encode_observation(obs)
            (a, s), logp, v = agent.act(o)

            action = {
                "action": ACTIONS[a],
                "params": {"service_id": SERVICES[s]}
            }

            next_obs, r, done, _ = env.step(action)

            batch["states"].append(o)
            batch["actions"].append(a * SERVICE_DIM + s)
            batch["log_probs"].append(logp)
            batch["rewards"].append(r)
            batch["values"].append(v)
            batch["dones"].append(float(done))

            obs = next_obs
            total += r

        agent.update(batch)
        history.append(total)

        if ep % 10 == 0:
            print(f"Ep {ep} reward {total}")

    print("Evaluating trained...")
    trained = evaluate(env, agent)

    os.makedirs("results", exist_ok=True)

    with open("results/metrics.json", "w") as f:
        json.dump({"baseline": baseline, "trained": trained}, f, indent=2)

    print("Done")


if __name__ == "__main__":
    train()
