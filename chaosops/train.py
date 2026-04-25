from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from datasets import Dataset

from env import ChaosOpsEnv
from wrapper import ChaosOpsWrapper


@dataclass
class EpisodeTrace:
    success: bool
    total_reward: float
    steps: int
    score: float
    transitions: List[Dict[str, Any]]


def _make_env(max_steps: int, seed: int) -> ChaosOpsEnv:
    try:
        return ChaosOpsEnv(max_steps=max_steps, seed=seed)
    except TypeError:
        # Newer env versions removed seed from constructor.
        return ChaosOpsEnv(max_steps=max_steps)


def _reset_task3(env: ChaosOpsEnv, variation_mode: bool) -> Dict[str, Any]:
    try:
        return env.reset("task3", variation_mode=variation_mode)
    except TypeError:
        # Newer env versions removed variation_mode from reset.
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
    # Colab install command:
    # pip install -U unsloth trl transformers datasets accelerate bitsandbytes peft
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


def generate_model_action(
    model: Any,
    tokenizer: Any,
    fastlm: Any,
    prompt: str,
    temperature: float,
    max_new_tokens: int = 64,
) -> str:
    fastlm.for_inference(model)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    input_len = int(inputs["input_ids"].shape[1])
    # Use max_length directly to avoid repeated warnings when both max_new_tokens
    # and generation_config.max_length are set by upstream libraries.
    generation_max_length = min(1024, input_len + max_new_tokens)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_length=generation_max_length,
            do_sample=True,
            temperature=max(0.2, temperature),
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    # Keep tail where model usually emits action JSON.
    return text[-1200:]


def run_episode(
    env: ChaosOpsEnv,
    wrapper: ChaosOpsWrapper,
    model: Any,
    tokenizer: Any,
    fastlm: Any,
    temperature: float,
    variation_mode: bool,
) -> EpisodeTrace:
    reset_out = _reset_task3(env, variation_mode=variation_mode)
    obs = _extract_observation(reset_out)

    transitions: List[Dict[str, Any]] = []
    final = None

    for _ in range(env.max_steps):
        prompt = wrapper.observation_to_prompt(obs)
        raw_text = generate_model_action(model, tokenizer, fastlm, prompt, temperature=temperature)
        action_obj = wrapper.parse_model_output(raw_text)

        out = env.step(action_obj["action"], action_obj["payload"])
        obs = _extract_observation(out)

        transitions.append(
            {
                "prompt": prompt,
                "response": json.dumps(action_obj, ensure_ascii=True),
                "reward": float(out.get("reward", out.get("reward_delta", 0.0))),
            }
        )

        final = out
        if out["done"]:
            break

    if final is None:
        raise RuntimeError("episode produced no steps")

    st = final["state"]
    score = final["score"] if final["score"] is not None else 0.01

    return EpisodeTrace(
        success=st["service_status"] == "running",
        total_reward=float(final["total_reward"]),
        steps=int(st["step_count"]),
        score=float(score),
        transitions=transitions,
    )


def build_grpo_style_dataset(group: List[EpisodeTrace]) -> Dataset:
    # Group-relative advantage: center rewards inside sampled group.
    rewards = [ep.total_reward for ep in group]
    baseline = sum(rewards) / max(1, len(rewards))

    rows: List[Dict[str, str]] = []
    for ep in group:
        advantage = ep.total_reward - baseline
        if advantage <= 0:
            continue

        # Repeat high-advantage samples to approximate weighted policy update.
        repeat = 1 + int(min(4.0, advantage * 2.0))
        for tr in ep.transitions:
            text = (
                "<s>[INST] "
                + tr["prompt"]
                + " [/INST] "
                + tr["response"]
                + "</s>"
            )
            for _ in range(repeat):
                rows.append({"text": text})

    if not rows:
        # Keep trainer alive with a tiny fallback sample.
        rows = [{"text": "<s>[INST] Return JSON action. [/INST] {\"action\":\"query_system\",\"payload\":{}}</s>"}]

    return Dataset.from_list(rows)


def train_loop(
    output_dir: str,
    model_name: str,
    train_steps: int,
    group_size: int,
    variation_prob: float,
) -> Dict[str, Any]:
    model, tokenizer, fastlm = load_unsloth_qwen(model_name=model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Import training backends after Unsloth model setup so Unsloth can patch first.
    use_trl = True
    try:
        from trl import SFTConfig, SFTTrainer
    except Exception as exc:  # pragma: no cover
        use_trl = False
        trl_error = str(exc)
        print(
            "[WARN] TRL import failed; falling back to transformers Trainer. "
            "Install/upgrade with: pip install -U trl mergekit"
        )
        print(f"[WARN] TRL error: {trl_error}")
        from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

    env = _make_env(max_steps=10, seed=13)
    wrapper = ChaosOpsWrapper()

    os.makedirs(output_dir, exist_ok=True)

    logs: List[Dict[str, Any]] = []
    interrupted = False

    for step in range(train_steps):
        try:
            group: List[EpisodeTrace] = []
            for _ in range(group_size):
                ep = run_episode(
                    env=env,
                    wrapper=wrapper,
                    model=model,
                    tokenizer=tokenizer,
                    fastlm=fastlm,
                    temperature=0.9,
                    variation_mode=(torch.rand(1).item() < variation_prob),
                )
                group.append(ep)

            ds = build_grpo_style_dataset(group)
            if use_trl:
                cfg = SFTConfig(
                    output_dir=output_dir,
                    per_device_train_batch_size=2,
                    gradient_accumulation_steps=2,
                    num_train_epochs=1,
                    learning_rate=2e-4,
                    logging_steps=1,
                    max_seq_length=1024,
                    save_strategy="no",
                    report_to=[],
                )

                trainer = SFTTrainer(
                    model=model,
                    tokenizer=tokenizer,
                    train_dataset=ds,
                    dataset_text_field="text",
                    args=cfg,
                )
                trainer.train()
            else:
                tokenized = ds.map(
                    lambda batch: tokenizer(batch["text"], truncation=True, max_length=1024),
                    batched=True,
                    remove_columns=["text"],
                )
                args = TrainingArguments(
                    output_dir=output_dir,
                    per_device_train_batch_size=2,
                    gradient_accumulation_steps=2,
                    num_train_epochs=1,
                    learning_rate=2e-4,
                    logging_steps=1,
                    save_strategy="no",
                    report_to=[],
                    remove_unused_columns=False,
                )
                collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
                trainer = Trainer(
                    model=model,
                    args=args,
                    train_dataset=tokenized,
                    data_collator=collator,
                )
                trainer.train()

            avg_reward = sum(ep.total_reward for ep in group) / len(group)
            success_rate = sum(1 for ep in group if ep.success) / len(group)
            avg_steps = sum(ep.steps for ep in group) / len(group)

            row = {
                "step": step,
                "avg_reward": avg_reward,
                "success_rate": success_rate,
                "avg_steps": avg_steps,
            }
            logs.append(row)
            print(json.dumps(row))
        except KeyboardInterrupt:
            interrupted = True
            print("\n[WARN] Training interrupted. Saving partial adapter and metrics...")
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
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./chaosops-qwen-grpo")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    # Backward compatibility: docs and older notebooks use --episodes.
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--train_steps", type=int, default=6)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--variation_prob", type=float, default=0.3)
    args = parser.parse_args()

    train_steps = args.episodes if args.episodes is not None else args.train_steps

    result = train_loop(
        output_dir=args.output_dir,
        model_name=args.model_name,
        train_steps=train_steps,
        group_size=args.group_size,
        variation_prob=args.variation_prob,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
