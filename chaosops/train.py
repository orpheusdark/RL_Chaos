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
from transformers import AutoModelForCausalLM, AutoTokenizer

from env import ChaosOpsEnv
from wrapper import ChaosOpsWrapper

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


@dataclass
class Transition:
    state: Dict[str, Any]
    action_text: str
    action: str
    token_ids: List[int]
    reward: float
    old_logprob: float


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


def _extract_reward(out: Dict[str, Any]) -> float:
    return float(out.get("reward", out.get("reward_delta", 0.0)))


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


def sample_action_with_generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> Tuple[str, List[int], float]:
    model.eval()
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.max_length = None
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    input_len = int(enc["input_ids"].shape[1])
    with torch.no_grad():
        outputs = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=max(0.7, min(1.0, float(temperature))),
            top_p=top_p,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    sequences = outputs.sequences[0]
    generated_ids = sequences[input_len:].tolist()
    action_text = tokenizer.decode(generated_ids, skip_special_tokens=True)[-1200:]

    token_logprobs: List[float] = []
    if outputs.scores:
        for idx, score_t in enumerate(outputs.scores):
            if idx >= len(generated_ids):
                break
            log_probs = torch.log_softmax(score_t[0], dim=-1)
            token_id = int(generated_ids[idx])
            token_logprobs.append(float(log_probs[token_id].item()))
    old_logprob = float(sum(token_logprobs))

    del outputs, enc, sequences
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return action_text, generated_ids, old_logprob


def compute_logprob_for_tokens_teacher_forced(
    model: Any,
    tokenizer: Any,
    prompt: str,
    target_token_ids: List[int],
) -> torch.Tensor:
    if not target_token_ids:
        return torch.zeros((), device=model.device, dtype=torch.float32, requires_grad=True)

    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    prompt_ids = enc["input_ids"]
    prompt_mask = enc.get("attention_mask")
    if prompt_mask is None:
        prompt_mask = torch.ones_like(prompt_ids, device=model.device)
    target = torch.tensor([target_token_ids], device=model.device, dtype=prompt_ids.dtype)
    target_mask = torch.ones_like(target, device=model.device)

    full_ids = torch.cat([prompt_ids, target], dim=1)
    full_mask = torch.cat([prompt_mask, target_mask], dim=1)
    out = model(input_ids=full_ids, attention_mask=full_mask)
    logits = out.logits[:, :-1, :]
    labels = full_ids[:, 1:]
    prompt_len = int(prompt_ids.shape[1])

    start = max(0, prompt_len - 1)
    end = start + len(target_token_ids)
    target_logits = logits[:, start:end, :]
    target_labels = labels[:, start:end]

    log_probs = torch.log_softmax(target_logits, dim=-1)
    gathered = torch.gather(log_probs, dim=-1, index=target_labels.unsqueeze(-1)).squeeze(-1)
    gathered = torch.clamp(gathered, min=-100.0, max=0.0)
    total = gathered.sum()

    del enc, prompt_ids, prompt_mask, target, target_mask, full_ids, full_mask, out, logits, labels, target_logits, target_labels, log_probs, gathered
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return total


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

    prev_action = ""
    for _ in range(env.max_steps):
        prompt = wrapper.observation_to_prompt(obs)
        raw_text, token_ids, old_logprob = sample_action_with_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
        action_obj = wrapper.parse_model_output(raw_text)
        out = env.step(action_obj["action"], action_obj["payload"])

        base_reward = _extract_reward(out)
        repeated = prev_action == str(action_obj["action"])
        useless = (not out.get("result", {}).get("ok", False)) and (not out.get("done", False))
        resolved = bool(out.get("state", {}).get("service_status") == "running")
        worsened = out.get("state", {}).get("schema_fail_count", 0) > 0 or (
            out.get("result", {}).get("error_code") == "NO_PERMISSION"
        )

        shaped_reward = -0.01 + base_reward
        if resolved:
            shaped_reward += 1.0
        if useless:
            shaped_reward -= 0.5
        if repeated:
            shaped_reward -= 0.2
        if worsened:
            shaped_reward -= 0.3

        transitions.append(
            Transition(
                state=obs,
                action_text=raw_text,
                action=str(action_obj["action"]),
                token_ids=token_ids,
                reward=float(shaped_reward),
                old_logprob=float(old_logprob),
            )
        )
        obs = _extract_observation(out)
        final = out
        prev_action = str(action_obj["action"])
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
    raw_text, _tokens, _logprob = sample_action_with_generate(
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
    if max_new_tokens > 32:
        raise ValueError("max_new_tokens must be <= 32 for T4-safe training.")
    if group_size > 2:
        raise ValueError("group_size must be <= 2 for T4-safe training.")
    if fastlm is not None:
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
            transition_pack: List[Tuple[Transition, float]] = []
            action_counter: Counter[str] = Counter()
            for ep in episodes:
                ep_returns = compute_discounted_returns(ep, gamma=gamma)
                all_returns.extend(ep_returns)
                for tr, ret in zip(ep.transitions, ep_returns):
                    action_counter.update([tr.action])
                    transition_pack.append((tr, ret))

            if not transition_pack:
                raise RuntimeError("No transitions collected; cannot update policy.")

            returns_tensor = torch.tensor(all_returns, dtype=torch.float32, device=model.device)
            advantages = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std(unbiased=False) + 1e-6)
            if torch.isnan(advantages).any():
                raise RuntimeError("NaN detected in normalized advantages.")

            logprob_terms: List[torch.Tensor] = []
            for (tr, _ret), adv in zip(transition_pack, advantages):
                prompt = wrapper.observation_to_prompt(tr.state)
                lp = compute_logprob_for_tokens_teacher_forced(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    target_token_ids=tr.token_ids,
                )
                if torch.isnan(lp) or torch.isinf(lp):
                    raise RuntimeError(f"Invalid logprob for action={tr.action} text={tr.action_text[:120]}")
                logprob_terms.append(lp * adv)

            loss = -torch.stack(logprob_terms).mean()
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

            total_actions = max(1, sum(action_counter.values()))
            top_action_count = max(action_counter.values()) if action_counter else 0
            repeat_ratio = top_action_count / total_actions
            if repeat_ratio > 0.70 and update >= 3:
                raise RuntimeError(f"Action repetition too high: {repeat_ratio:.2f}")

            if episodes and episodes[0].transitions:
                debug_traj = []
                for tr in episodes[0].transitions:
                    debug_traj.append(
                        {
                            "state": tr.state,
                            "action": tr.action,
                            "reward": tr.reward,
                        }
                    )
                print(json.dumps({"trajectory_debug": debug_traj}, ensure_ascii=True))

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
    parser.add_argument("--max_new_tokens", type=int, default=32)
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
