from __future__ import annotations

import json
import sys
from typing import Any, Dict, List

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from env import ChaosOpsEnv


MODEL_NAME = "sshleifer/tiny-gpt2"


def build_dataset() -> Dataset:
    prompts = [
        "Recover auth_service after OOM crash. Request access, adapt schema if needed, then fix.",
        "Investigate crashed service and complete repair workflow safely.",
        "Handle schema drift and restore auth_service to running.",
    ]
    return Dataset.from_dict({"prompt": prompts})


def _parse_completion_to_actions(text: str) -> List[Dict[str, Any]]:
    text_lower = text.lower()
    actions: List[Dict[str, Any]] = [{"action": "query_system", "payload": {}}]

    if "schema" in text_lower:
        actions.append({"action": "get_schema", "payload": {}})

    if "permission" in text_lower or "token" in text_lower or "access" in text_lower:
        actions.append(
            {
                "action": "request_access",
                "payload": {"justification": "Need token to recover oom crash"},
            }
        )
    else:
        actions.append(
            {
                "action": "request_access",
                "payload": {"justification": "OOM crash requires authorized fix"},
            }
        )

    actions.append({"action": "get_schema", "payload": {}})
    return actions


def _run_env_from_completion(completion_text: str) -> float:
    env = ChaosOpsEnv(max_steps=10)
    env.reset("task3")

    actions = _parse_completion_to_actions(completion_text)
    token = ""

    for item in actions:
        out = env.step(item["action"], item["payload"])
        if item["action"] == "request_access":
            token = out.get("result", {}).get("token", "")

    schema_info = env.step("get_schema", {}).get("result", {}).get("schema", {})
    version = schema_info.get("version", 2)

    if version == 1:
        config = {
            "service": "auth_service",
            "status": "running",
            "cpu_limit": "1",
        }
    else:
        config = {
            "service_name": "auth_service",
            "condition": "running",
            "max_compute": "1",
        }

    final = env.step("fix_service", {"config": config, "token": token})
    if final.get("done") and isinstance(final.get("score"), (float, int)):
        return float(final["score"])
    return float(max(0.01, min(0.99, final.get("total_reward", 0.0))))


def reward_from_env(prompts: List[str], completions: List[Any], **_: Any) -> List[float]:
    rewards: List[float] = []
    for completion in completions:
        text = ""
        if isinstance(completion, str):
            text = completion
        elif isinstance(completion, list):
            text = " ".join(str(part) for part in completion)
        elif isinstance(completion, dict):
            text = json.dumps(completion)
        else:
            text = str(completion)
        rewards.append(_run_env_from_completion(text))
    return rewards


def main() -> int:
    try:
        dataset = build_dataset()

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

        config = GRPOConfig(
            output_dir="./chaosops-grpo-out",
            per_device_train_batch_size=2,
            num_generations=2,
            max_steps=1,
            learning_rate=1e-5,
            logging_steps=1,
            use_cpu=True,
            bf16=False,
            fp16=False,
            remove_unused_columns=False,
            report_to=[],
        )

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_from_env,
            args=config,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

        trainer.train()
        return 0
    except Exception as exc:
        print(f"training failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
