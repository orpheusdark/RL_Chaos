# ⚡ Chaosops Colab Quick-Start Reference

**Copy-paste these commands in order into Google Colab cells**

---

## CELL 1: Setup & Clone (1 minute)

```python
import torch
import subprocess
import sys

# Check GPU
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Clone repo
!git clone https://github.com/orpheusdark/Chaosops.git 2>/dev/null
%cd /content/Chaosops
print("✅ Repository cloned")
```

---

## CELL 2: Install Dependencies (5-10 minutes)

```python
print("Installing packages...")
!pip install -q -U unsloth trl transformers datasets accelerate bitsandbytes peft torch

# Verify
from unsloth import FastLanguageModel
from peft import get_peft_model, LoraConfig
print("✅ All packages installed")
```

---

## CELL 3: Run Training (10-15 minutes)

```python
import os
import subprocess

os.chdir("/content/Chaosops/chaosops")

print("🚀 Starting Training...")
result = subprocess.run(
    ["python", "train.py", 
    "--train_steps", "10",
     "--model_name", "Qwen/Qwen2.5-0.5B",
     "--output_dir", "../chaosops-qwen-grpo"],
    capture_output=True,
    text=True,
    timeout=1800
)

if result.returncode == 0:
    print("\n✅ Training complete")
else:
    print(f"\n❌ Training failed (code {result.returncode})")
    print(result.stderr)
```

---

## CELL 4: Run Evaluation (5-10 minutes)

```python
import json

result = subprocess.run(
    ["python", "eval.py", "--episodes", "20"],
    cwd="/content/Chaosops/chaosops",
    capture_output=True,
    text=True,
    timeout=900
)

eval_results = json.loads(result.stdout)

print("\n" + "="*60)
print("📊 RESULTS")
print("="*60)
print(f"\n🎲 Baseline:      {eval_results['baseline']['success_rate']:.1%} success")
print(f"🤖 Trained:       {eval_results['trained']['success_rate']:.1%} success")
print(f"🔄 Variation:     {eval_results['variation']['success_rate']:.1%} success")
print(f"\n📈 Improvement:   +{eval_results['success_improvement']:.1%}")
print(f"✅ Verdict:       {eval_results['verdict']}")
print("="*60)
```

---

## CELL 5: Visualize (1 minute)

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Success Rate
success = [eval_results['baseline']['success_rate'],
           eval_results['trained']['success_rate'],
           eval_results['variation']['success_rate']]
axes[0].bar(['Baseline', 'Trained', 'Variation'], success, 
            color=['#FF6B6B', '#51CF66', '#4DABF7'], alpha=0.8, edgecolor='black', linewidth=2)
axes[0].set_ylabel('Success Rate', fontweight='bold')
axes[0].set_ylim([0, 1.1])
axes[0].set_title('Success Rate Comparison')
axes[0].grid(axis='y', alpha=0.3)

# Reward
rewards = [eval_results['baseline']['avg_reward'],
           eval_results['trained']['avg_reward'],
           eval_results['variation']['avg_reward']]
axes[1].bar(['Baseline', 'Trained', 'Variation'], rewards,
            color=['#FF6B6B', '#51CF66', '#4DABF7'], alpha=0.8, edgecolor='black', linewidth=2)
axes[1].set_ylabel('Average Reward', fontweight='bold')
axes[1].set_title('Reward Comparison')
axes[1].grid(axis='y', alpha=0.3)
axes[1].axhline(0, color='black', linestyle='--', linewidth=1)

plt.tight_layout()
plt.show()
print("✅ Visualization complete")
```

---

## CELL 6: Save to Google Drive (Optional - 2 minutes)

```python
from google.colab import drive
import shutil

drive.mount('/content/drive', force_remount=True)

results_dir = "/content/drive/MyDrive/Chaosops_Results"
os.makedirs(results_dir, exist_ok=True)

# Copy adapter & results
shutil.copytree("/content/Chaosops/chaosops-qwen-grpo",
                f"{results_dir}/chaosops-qwen-grpo", dirs_exist_ok=True)
with open(f"{results_dir}/eval_results.json", "w") as f:
    json.dump(eval_results, f, indent=2)

print(f"✅ Saved to Google Drive: {results_dir}")
```

---

## Key Parameters

### Training
- `--train_steps 10` → Change to `50` for longer training
- `--model_name` → "Qwen/Qwen2.5-0.5B" (or other Qwen models)
- `--output_dir` → Where to save LoRA adapter

### Evaluation
- `--episodes 20` → Episodes per test (baseline/trained/variation)
- `--adapter_dir` → Path to LoRA adapter

---

## Expected Outputs

**Baseline (Random):**
- Success: 5%
- Reward: -0.04
- Steps: 7-8

**After Training:**
- Success: 95%+
- Reward: 1.0+
- Steps: 4

**Variation Test:**
- Success: 85%+ (shows robustness)

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **Out of Memory** | Reduce `--train_steps 10` to `5` |
| **GPU not available** | Runtime → Change runtime type → GPU |
| **Import errors** | `!pip install --force-reinstall -q unsloth peft` |
| **Model download fails** | `!huggingface-cli login` (paste token) |

---

## Files & Links

**Repository:** https://github.com/orpheusdark/Chaosops  
**HF Space:** https://huggingface.co/spaces/orpheusdark/chaosops  
**Model:** Qwen/Qwen2.5-0.5B (0.5B parameters, 4-bit quantized)

**Output Files:**
- `chaosops-qwen-grpo/` → Trained LoRA adapter
- `training_results_dashboard.png` → Visualization
- `eval_results.json` → Detailed metrics

---

## One-Cell Alternative (All-in-One)

Copy this into ONE cell to run everything sequentially:

```python
import torch, subprocess, sys, os, json, matplotlib.pyplot as plt
from google.colab import drive

# Setup
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
!git clone https://github.com/orpheusdark/Chaosops.git 2>/dev/null
%cd /content/Chaosops
!pip install -q -U unsloth trl transformers datasets accelerate bitsandbytes peft torch

# Train
os.chdir("/content/Chaosops/chaosops")
subprocess.run(["python", "train.py", "--train_steps", "10", "--output_dir", "../chaosops-qwen-grpo"],
               capture_output=False, timeout=1800)

# Evaluate
result = subprocess.run(["python", "eval.py", "--episodes", "20"],
                        capture_output=True, text=True, timeout=900)
eval_results = json.loads(result.stdout)

# Display
print(f"\nBaseline: {eval_results['baseline']['success_rate']:.1%}")
print(f"Trained:  {eval_results['trained']['success_rate']:.1%}")
print(f"Verdict:  {eval_results['verdict']}")
```

---

**Last Updated:** April 25, 2026  
**Status:** ✅ Ready for Colab training
