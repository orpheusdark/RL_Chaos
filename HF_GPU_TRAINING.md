# HF GPU Training Guide (One Page)

This is the fastest way to use your Hugging Face credits for stable training (no Colab disconnects).

## What You Need

- Hugging Face account with available credits
- A write token with access to create/upload model repos
- This repository

## 1. Create a GPU Space for Training

1. Go to https://huggingface.co/new-space
2. Choose:
- SDK: Docker or Gradio (either is fine for terminal usage)
- Hardware: start with a small GPU to save credits
- Visibility: Private recommended
3. In Space settings, add a secret:
- `HF_TOKEN`: your HF write token

## 2. Start Training + Upload in One Command

In the Space terminal:

```bash
git clone https://github.com/orpheusdark/Chaosops.git
cd Chaosops
python -m pip install -U pip
python -m pip install -r chaosops/requirements.txt
python -m pip install -U unsloth trl transformers datasets accelerate bitsandbytes peft huggingface_hub
python scripts/train_and_upload_hf.py --model_repo <your-username>/chaosops-qwen-grpo --train_steps 10 --group_size 2
```

What this command does:

1. Runs training with `chaosops/train.py`
2. Runs evaluation with `chaosops/eval.py`
3. Uploads the adapter folder to your HF model repo

## 3. Useful Variants

Run faster / cheaper:

```bash
python scripts/train_and_upload_hf.py --model_repo <your-username>/chaosops-qwen-grpo --train_steps 5 --group_size 2 --skip_eval
```

Run longer / better quality:

```bash
python scripts/train_and_upload_hf.py --model_repo <your-username>/chaosops-qwen-grpo --train_steps 30 --group_size 4
```

Create private model repo:

```bash
python scripts/train_and_upload_hf.py --model_repo <your-username>/chaosops-qwen-grpo --private
```

## 4. Credit Control Tips

1. Start with `--train_steps 5` first to validate the pipeline
2. Keep `--group_size 2` on smaller GPUs
3. Stop GPU runtime immediately after training
4. Re-run only when code/data changes

## 5. Notes

- Cursor credits are for coding assistance, not GPU training compute.
- HF credits are the right resource for training runs.
- If training is interrupted, rerun the same command and continue with a short test pass.
