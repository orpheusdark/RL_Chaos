from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


def run_command(cmd: list[str], cwd: Path) -> None:
    print(f"[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


def ensure_hf_hub_installed() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        ) from exc


def upload_adapter_folder(adapter_dir: Path, model_repo: str, token: Optional[str], private: bool) -> str:
    ensure_hf_hub_installed()
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=model_repo, private=private, exist_ok=True)
    api.upload_folder(
        repo_id=model_repo,
        folder_path=str(adapter_dir),
        path_in_repo=".",
        commit_message="Upload ChaosOps trained adapter",
    )
    return f"https://huggingface.co/{model_repo}"


def maybe_write_model_card(adapter_dir: Path, model_repo: str, train_steps: int, group_size: int) -> None:
    readme_path = adapter_dir / "README.md"
    if readme_path.exists():
        return

    card = (
        f"# ChaosOps Adapter\n\n"
        f"This adapter was trained with the ChaosOps environment.\n\n"
        f"- Base model: `Qwen/Qwen2.5-0.5B`\n"
        f"- Train steps: `{train_steps}`\n"
        f"- Group size: `{group_size}`\n"
        f"- Repository: `{model_repo}`\n"
    )
    readme_path.write_text(card, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train ChaosOps model and upload adapter to Hugging Face model hub")
    parser.add_argument("--model_repo", required=True, help="Target HF model repo, e.g. username/chaosops-qwen-grpo")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--train_steps", type=int, default=10)
    parser.add_argument("--group_size", type=int, default=2)
    parser.add_argument("--variation_prob", type=float, default=0.3)
    parser.add_argument("--eval_episodes", type=int, default=20)
    parser.add_argument("--adapter_dir", default="chaosops-qwen-grpo")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--private", action="store_true", help="Create private HF model repo")
    parser.add_argument("--hf_token", default=None, help="HF token. Defaults to HF_TOKEN env var")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    work_dir = repo_root / "chaosops"
    adapter_dir = repo_root / args.adapter_dir

    print("=" * 72)
    print("ChaosOps HF train + upload")
    print("=" * 72)
    print(f"Repo root: {repo_root}")
    print(f"Work dir:  {work_dir}")
    print(f"Adapter:   {adapter_dir}")

    run_command(
        [
            sys.executable,
            "train.py",
            "--train_steps",
            str(args.train_steps),
            "--group_size",
            str(args.group_size),
            "--variation_prob",
            str(args.variation_prob),
            "--model_name",
            args.base_model,
            "--output_dir",
            str(adapter_dir),
        ],
        cwd=work_dir,
    )

    eval_results = None
    if not args.skip_eval:
        print("\n[INFO] Running evaluation...")
        eval_proc = subprocess.run(
            [
                sys.executable,
                "eval.py",
                "--episodes",
                str(args.eval_episodes),
                "--adapter_dir",
                str(adapter_dir),
                "--base_model",
                args.base_model,
            ],
            cwd=str(work_dir),
            check=False,
            capture_output=True,
            text=True,
        )
        if eval_proc.returncode == 0:
            try:
                eval_results = json.loads(eval_proc.stdout)
                out_path = adapter_dir / "eval_results.json"
                out_path.write_text(json.dumps(eval_results, indent=2), encoding="utf-8")
                print(f"[OK] Saved evaluation results to {out_path}")
            except Exception:
                print("[WARN] Evaluation output was not valid JSON; continuing upload.")
        else:
            print("[WARN] Evaluation failed; continuing upload.")
            print(eval_proc.stderr)

    maybe_write_model_card(adapter_dir, args.model_repo, args.train_steps, args.group_size)

    token = args.hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Missing token. Set HF_TOKEN env var or pass --hf_token")

    model_url = upload_adapter_folder(
        adapter_dir=adapter_dir,
        model_repo=args.model_repo,
        token=token,
        private=args.private,
    )

    print("\n" + "=" * 72)
    print(f"[OK] Uploaded adapter to: {model_url}")
    if eval_results:
        print(f"[OK] Verdict: {eval_results.get('verdict', 'N/A')}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
