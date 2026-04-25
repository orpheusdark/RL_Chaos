from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from torch.distributions import Categorical
from transformers import AutoModelForCausalLM, AutoTokenizer

from env import ChaosOpsEnv
from wrapper import ChaosOpsWrapper


@dataclass
class Transition:
    reward: float
    logprob: torch.Tensor
    action: str


@dataclass
class EpisodeTrace:
    success: bool
    total_reward: float
    steps: int
    transitions: List[Transition]


def _make_env(max_steps: int, seed: int) -> ChaosOpsEnv:
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        return ChaosOpsEnv(max_steps=max_steps, seed=seed)
    except TypeError:
        return ChaosOpsEnv(max_steps=max_steps)


def _reset_task3(env: ChaosOpsEnv, variation_mode: bool) -> Dict[str, Any]:
    try:
        return env.reset("task3", variation_mode=variation_mode)
    except TypeError:
        return env.reset("task3")


def _observation_from_state(st: Dict[str, Any]) -> Dict[str, Any]:
    if st.get("api_schema_version", 1) == 1:
        return {
            "service": "auth_service",
            "status": st.get("service_status", "crashed"),
            "cpu_limit": "500m",
        }
    return {
        "service_name": "auth_service",
        "condition": st.get("service_status", "crashed"),
        "max_compute": "1",
    }


def _extract_observation(out: Dict[str, Any]) -> Dict[str, Any]:
    obs = out.get("observation")
    if isinstance(obs, dict):
        return obs
    st = out.get("state")
    if isinstance(st, dict):
        return _observation_from_state(st)
    return {"service": "auth_service", "status": "crashed", "cpu_limit": "500m"}


def load_unsloth_qwen(model_name: str = "Qwen/Qwen2.5-0.5B"):
    """
    Preferred path: Unsloth + QLoRA.
    Fallback path: plain Transformers CausalLM.
    """
    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=1024,
            dtype=None,
            load_in_4bit=True,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
        return model, tokenizer, FastLanguageModel
    except Exception as exc:
        print(f"[WARN] Unsloth unavailable, using Transformers fallback: {exc}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name)
        return model, tokenizer, None


def _top_p_filter_logits(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    sorted_mask = cumulative_probs > top_p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(dim=-1, index=sorted_indices, src=sorted_mask)
    return logits.masked_fill(mask, float("-inf"))


def sample_action_and_logprob(
    model: Any,
    tokenizer: Any,
    prompt: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> Tuple[str, torch.Tensor]:
    """
    Sample one action text with token-level logprob sum tracked for policy gradient.
    """
    model.train()
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    sampled_token_ids: List[int] = []
    token_logprobs: List[torch.Tensor] = []
    eos_id = tokenizer.eos_token_id
    temp = max(0.2, float(temperature))

    for _ in range(max_new_tokens):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        next_logits = outputs.logits[:, -1, :] / temp
        next_logits = _top_p_filter_logits(next_logits, top_p=top_p)
        dist = Categorical(logits=next_logits)
        next_token = dist.sample()  # [1]
        token_logprob = dist.log_prob(next_token).squeeze(0)

        sampled_token_ids.append(int(next_token.item()))
        token_logprobs.append(token_logprob)

        next_token_expanded = next_token.unsqueeze(-1)
        input_ids = torch.cat([input_ids, next_token_expanded], dim=-1)
        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token_expanded)], dim=-1)

        if eos_id is not None and int(next_token.item()) == int(eos_id):
            break

    if sampled_token_ids:
        action_text = tokenizer.decode(sampled_token_ids, skip_special_tokens=True)
        total_logprob = torch.stack(token_logprobs).sum()
    else:
        action_text = ""
        # Zero with gradient path if model output exists.
        total_logprob = outputs.logits[:, -1, 0].sum() * 0.0

    return action_text[-1200:], total_logprob


def generate_model_action(
    model: Any,
    tokenizer: Any,
    fastlm: Any,
    prompt: str,
    temperature: float,
    max_new_tokens: int = 96,
) -> str:
    """
    Inference helper kept for eval.py compatibility.
    """
    if fastlm is not None:
        fastlm.for_inference(model)
    model.eval()
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.max_length = None
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=max(0.2, temperature),
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text[-1200:]


def run_episode(
    env: ChaosOpsEnv,
    wrapper: ChaosOpsWrapper,
    model: Any,
    tokenizer: Any,
    temperature: float,
    variation_mode: bool,
    top_p: float,
    max_new_tokens: int,
) -> EpisodeTrace:
    reset_out = _reset_task3(env, variation_mode=variation_mode)
    obs = _extract_observation(reset_out)
    transitions: List[Transition] = []
    final = None

    for _ in range(env.max_steps):
        prompt = wrapper.observation_to_prompt(obs)
        raw_text, action_logprob = sample_action_and_logprob(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
        action_obj = wrapper.parse_model_output(raw_text)
        out = env.step(action_obj["action"], action_obj["payload"])
        reward = float(out.get("reward", out.get("reward_delta", 0.0)))
        transitions.append(
            Transition(
                reward=reward,
                logprob=action_logprob,
                action=str(action_obj["action"]),
            )
        )
        obs = _extract_observation(out)
        final = out
        if out["done"]:
            break

    if final is None:
        raise RuntimeError("Episode produced no steps.")

    st = final["state"]
    return EpisodeTrace(
        success=st["service_status"] == "running",
        total_reward=float(final["total_reward"]),
        steps=int(st["step_count"]),
        transitions=transitions,
    )


def compute_discounted_returns(episode: EpisodeTrace, gamma: float) -> List[float]:
    returns = [0.0] * len(episode.transitions)
    running = 0.0
    for idx in reversed(range(len(episode.transitions))):
        running = episode.transitions[idx].reward + gamma * running
        returns[idx] = running
    return returns


def run_sanity_check(
    env: ChaosOpsEnv,
    wrapper: ChaosOpsWrapper,
    model: Any,
    tokenizer: Any,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> None:
    print("[SANITY] Running one manual environment interaction...")
    out = _reset_task3(env, variation_mode=False)
    obs = _extract_observation(out)
    prompt = wrapper.observation_to_prompt(obs)
    raw_text, _logprob = sample_action_and_logprob(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )
    action_obj = wrapper.parse_model_output(raw_text)
    step_out = env.step(action_obj["action"], action_obj["payload"])
    print(
        json.dumps(
            {
                "state": obs,
                "action": action_obj,
                "reward": float(step_out.get("reward", step_out.get("reward_delta", 0.0))),
            },
            ensure_ascii=True,
        )
    )


def train_loop(
    output_dir: str,
    model_name: str,
    train_steps: int,
    group_size: int,
    variation_prob: float,
    learning_rate: float,
    gamma: float,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    max_grad_norm: float,
) -> Dict[str, Any]:
    if learning_rate <= 0:
        raise ValueError("learning_rate must be > 0 for RL updates.")

    model, tokenizer, fastlm = load_unsloth_qwen(model_name=model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if fastlm is not None:
        # Needed after inference patching.
        model.train()

    env = _make_env(max_steps=10, seed=13)
    wrapper = ChaosOpsWrapper()
    os.makedirs(output_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    logs: List[Dict[str, Any]] = []
    interrupted = False
    recent_rewards: List[float] = []
    recent_losses: List[float] = []

    run_sanity_check(
        env=env,
        wrapper=wrapper,
        model=model,
        tokenizer=tokenizer,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )

    for update in range(train_steps):
        try:
            episodes: List[EpisodeTrace] = []
            for _ in range(group_size):
                ep = run_episode(
                    env=env,
                    wrapper=wrapper,
                    model=model,
                    tokenizer=tokenizer,
                    temperature=temperature,
                    variation_mode=(torch.rand(1).item() < variation_prob),
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                )
                episodes.append(ep)

            all_returns: List[float] = []
            logprob_terms: List[torch.Tensor] = []
            action_counter: Counter[str] = Counter()
            for ep in episodes:
                ep_returns = compute_discounted_returns(ep, gamma=gamma)
                all_returns.extend(ep_returns)
                for tr, ret in zip(ep.transitions, ep_returns):
                    action_counter.update([tr.action])
                    logprob_terms.append(tr.logprob)

            if not logprob_terms:
                raise RuntimeError("No transitions collected; cannot update policy.")

            returns_tensor = torch.tensor(all_returns, dtype=torch.float32, device=model.device)
            # Normalize returns to reduce gradient variance / NaN risk.
            advantages = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std(unbiased=False) + 1e-6)
            if torch.isnan(advantages).any():
                raise RuntimeError("NaN detected in normalized advantages.")

            logprob_tensor = torch.stack(logprob_terms)
            # Clamp extreme values for stability.
            logprob_tensor = torch.clamp(logprob_tensor, min=-100.0, max=0.0)
            loss = -(logprob_tensor * advantages).mean()
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError("NaN/Inf policy loss before backward.")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm))
            if math.isnan(grad_norm) or math.isinf(grad_norm):
                raise RuntimeError("NaN/Inf gradient norm after backward.")
            optimizer.step()

            avg_reward = sum(ep.total_reward for ep in episodes) / len(episodes)
            reward_var = float(torch.tensor([ep.total_reward for ep in episodes]).var(unbiased=False).item())
            success_rate = sum(1 for ep in episodes if ep.success) / len(episodes)
            avg_steps = sum(ep.steps for ep in episodes) / len(episodes)
            loss_value = float(loss.detach().item())

            row = {
                "step": update,
                "avg_reward": avg_reward,
                "reward_var": reward_var,
                "success_rate": success_rate,
                "avg_steps": avg_steps,
                "loss": loss_value,
                "grad_norm": grad_norm,
                "action_distribution": dict(action_counter),
            }
            logs.append(row)
            print(json.dumps(row))

            recent_rewards.append(avg_reward)
            recent_losses.append(loss_value)
            if len(recent_rewards) >= 5:
                assert (
                    max(recent_rewards[-5:]) - min(recent_rewards[-5:])
                ) > 1e-6, "avg_reward unchanged over 5 updates"
            if len(recent_losses) >= 5:
                assert (
                    max(recent_losses[-5:]) - min(recent_losses[-5:])
                ) > 1e-8, "loss unchanged over 5 updates"
        except KeyboardInterrupt:
            interrupted = True
            print("\n[WARN] Training interrupted. Saving partial model and metrics...")
            break

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    metrics_path = os.path.join(output_dir, "train_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)

    return {
        "output_dir": output_dir,
        "metrics_path": metrics_path,
        "interrupted": interrupted,
        "completed_steps": len(logs),
        "learning_rate": learning_rate,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./chaosops-qwen-grpo")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--train_steps", type=int, default=6)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--variation_prob", type=float, default=0.3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    args = parser.parse_args()

    train_steps = args.episodes if args.episodes is not None else args.train_steps
    result = train_loop(
        output_dir=args.output_dir,
        model_name=args.model_name,
        train_steps=train_steps,
        group_size=args.group_size,
        variation_prob=args.variation_prob,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        max_grad_norm=args.max_grad_norm,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
