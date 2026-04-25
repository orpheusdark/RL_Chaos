from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

import torch

from env import ChaosOpsEnv
from wrapper import ChaosOpsWrapper

try:
    # Preferred path: reuse helpers from training module.
    from train import generate_model_action, load_unsloth_qwen
except Exception:
    # Fallback path for environments where train.py cannot import TRL extras.
    # Evaluation only needs inference helpers, not trainer classes.
    def load_unsloth_qwen(model_name: str = "Qwen/Qwen2.5-0.5B"):
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=1024,
            dtype=None,
            load_in_4bit=True,
        )

        return model, tokenizer, FastLanguageModel


    def generate_model_action(
        model: Any,
        tokenizer: Any,
        fastlm: Any,
        prompt: str,
        temperature: float,
        max_new_tokens: int = 120,
    ) -> str:
        fastlm.for_inference(model)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=max(0.2, temperature),
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )

        text = tokenizer.decode(out[0], skip_special_tokens=True)
        return text[-1200:]


def _make_env(max_steps: int, seed: int) -> ChaosOpsEnv:
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


def run_policy_eval(
    model: Any,
    tokenizer: Any,
    fastlm: Any,
    episodes: int,
    variation_mode: bool,
    temperature: float,
) -> Dict[str, Any]:
    env = _make_env(max_steps=10, seed=99)
    wrapper = ChaosOpsWrapper()

    successes = 0
    rewards = []
    steps = []
    wrong_schema = 0
    no_permission = 0
    repeated = 0

    for _ in range(episodes):
        out = _reset_task3(env, variation_mode=variation_mode)
        obs = _extract_observation(out)
        final = None

        for _s in range(env.max_steps):
            prompt = wrapper.observation_to_prompt(obs)
            raw = generate_model_action(model, tokenizer, fastlm, prompt, temperature=temperature)
            action_obj = wrapper.parse_model_output(raw)
            final = env.step(action_obj["action"], action_obj["payload"])
            obs = _extract_observation(final)
            if final["done"]:
                break

        if final is None:
            continue

        st = final["state"]
        ec = st.get("error_counts", {})

        successes += 1 if st["service_status"] == "running" else 0
        rewards.append(float(final["total_reward"]))
        steps.append(int(st["step_count"]))
        wrong_schema += int(ec.get("WRONG_SCHEMA", st.get("schema_fail_count", 0)))
        no_permission += int(ec.get("NO_PERMISSION", 0))
        repeated += int(ec.get("REPEATED_USELESS_ACTIONS", 0))

    denom = max(1, len(rewards))
    return {
        "success_rate": successes / denom,
        "avg_reward": sum(rewards) / denom,
        "avg_steps": sum(steps) / denom,
        "error_counts": {
            "WRONG_SCHEMA": wrong_schema,
            "NO_PERMISSION": no_permission,
            "REPEATED_USELESS_ACTIONS": repeated,
        },
    }


def run_random_baseline(episodes: int) -> Dict[str, Any]:
    env = _make_env(max_steps=10, seed=55)

    actions = ["query_system", "get_schema", "request_access", "fix_service"]
    successes = 0
    rewards = []
    steps = []
    wrong_schema = 0
    no_permission = 0
    repeated = 0

    for _ in range(episodes):
        _reset_task3(env, variation_mode=False)
        final = None
        token = ""

        for _s in range(env.max_steps):
            action = actions[torch.randint(0, len(actions), (1,)).item()]
            payload = {}
            if action == "request_access":
                payload = {"justification": "please approve quick fix"}
            elif action == "fix_service":
                payload = {
                    "config": {"service": "auth_service", "status": "running", "cpu_limit": "1000m"},
                    "token": token,
                }

            out = env.step(action, payload)
            final = out
            if action == "request_access" and out["result"].get("ok"):
                token = out["result"].get("token", "")
            if out["done"]:
                break

        if final is None:
            continue

        st = final["state"]
        ec = st.get("error_counts", {})

        successes += 1 if st["service_status"] == "running" else 0
        rewards.append(float(final["total_reward"]))
        steps.append(int(st["step_count"]))
        wrong_schema += int(ec.get("WRONG_SCHEMA", st.get("schema_fail_count", 0)))
        no_permission += int(ec.get("NO_PERMISSION", 0))
        repeated += int(ec.get("REPEATED_USELESS_ACTIONS", 0))

    denom = max(1, len(rewards))
    return {
        "success_rate": successes / denom,
        "avg_reward": sum(rewards) / denom,
        "avg_steps": sum(steps) / denom,
        "error_counts": {
            "WRONG_SCHEMA": wrong_schema,
            "NO_PERMISSION": no_permission,
            "REPEATED_USELESS_ACTIONS": repeated,
        },
    }


def verdict(baseline: Dict[str, Any], trained: Dict[str, Any], variation: Dict[str, Any]) -> str:
    success_gain = trained["success_rate"] - baseline["success_rate"]
    if success_gain >= 0.20 and variation["success_rate"] >= 0.50:
        return "ROBUST LEARNING"
    if trained["success_rate"] > 0.75 and variation["success_rate"] < 0.30:
        return "SCRIPTED POLICY"
    return "WEAK LEARNING"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_dir", type=str, default="./chaosops-qwen-grpo")
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B")
    args = parser.parse_args()

    # Baseline: random policy.
    baseline = run_random_baseline(args.episodes)

    # Trained model: load base + adapter if present.
    model, tokenizer, fastlm = load_unsloth_qwen(model_name=args.base_model)
    if os.path.isdir(args.adapter_dir):
        try:
            model.load_adapter(args.adapter_dir)
        except Exception:
            try:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, args.adapter_dir)
            except Exception:
                # Fallback for runs that saved a full Transformers model directory.
                model = AutoModelForCausalLM.from_pretrained(args.adapter_dir)
                model.eval()

    trained = run_policy_eval(
        model=model,
        tokenizer=tokenizer,
        fastlm=fastlm,
        episodes=args.episodes,
        variation_mode=False,
        temperature=0.3,
    )

    variation = run_policy_eval(
        model=model,
        tokenizer=tokenizer,
        fastlm=fastlm,
        episodes=args.episodes,
        variation_mode=True,
        temperature=0.3,
    )

    output = {
        "baseline": baseline,
        "trained": trained,
        "variation": variation,
        "success_improvement": trained["success_rate"] - baseline["success_rate"],
        "reward_improvement": trained["avg_reward"] - baseline["avg_reward"],
        "efficiency_gain": baseline["avg_steps"] - trained["avg_steps"],
        "robustness_drop": trained["success_rate"] - variation["success_rate"],
        "verdict": verdict(baseline, trained, variation),
    }

    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
