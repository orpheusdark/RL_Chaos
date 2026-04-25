# 🎯 Chaosops RL Training: Complete Step-by-Step Guide

## 📊 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    GOOGLE COLAB WORKFLOW                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1️⃣ SETUP          2️⃣ INSTALL         3️⃣ TRAIN              │
│  ─────────         ──────────────      ────────              │
│  • Check GPU        • unsloth          • Load Qwen2.5         │
│  • Clone repo       • transformers     • Initialize env      │
│  • Verify files     • peft (LoRA)      • Run 10 episodes     │
│                     • torch            • Save adapter        │
│  ⏱️  1 min          ⏱️  5-10 min       ⏱️  10-15 min          │
│                                                                 │
│  4️⃣ EVALUATE       5️⃣ VISUALIZE       6️⃣ SAVE                │
│  ─────────────      ────────────       ──────                 │
│  • Baseline         • Charts           • Google Drive         │
│  • Trained          • Metrics          • GitHub Auto-push    │
│  • Variation        • Dashboard        • Results JSON        │
│  ⏱️  5-10 min       ⏱️  1 min           ⏱️  2 min              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
              Total Time: ~35-50 minutes (first run)
```

---

## 🚀 Quick Reference Table

| Step | Cell # | Command | Time |
|------|--------|---------|------|
| 1. Check GPU | Cell 1 | `print(torch.cuda.get_device_name(0))` | 30 sec |
| 2. Clone Repo | Cell 1 | `!git clone https://github.com/orpheusdark/Chaosops.git` | 30 sec |
| 3. Install | Cell 2 | `!pip install -U unsloth trl transformers...` | 5-10 min |
| 4. Train | Cell 3 | `python train.py --train_steps 10` | 10-15 min |
| 5. Evaluate | Cell 4 | `python eval.py --episodes 20` | 5-10 min |
| 6. Visualize | Cell 5 | Plot results | 1 min |
| 7. Save (opt) | Cell 6 | `drive.mount()` + copy | 2 min |

**Total: ~35-50 minutes**

---

## 📝 Detailed Step-by-Step Instructions

### Step 1: Open Google Colab & Create Notebook

1. Go to **[colab.research.google.com](https://colab.research.google.com)**
2. Click **"New Notebook"** or upload `Chaosops_Colab_Training.ipynb`
3. **⚠️ IMPORTANT:** Change runtime to GPU
   - Runtime → Change runtime type → GPU (T4 or V100)
   - Wait for GPU to initialize

---

### Step 2: Check GPU Availability

**Copy into Cell 1:**

```python
import torch

print("🔍 Environment Check:")
print(f"  Python Version: {torch.__version__}")
print(f"  GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("  ⚠️  NO GPU! Go to Runtime → Change runtime type → GPU")
```

**Expected Output:**
```
🔍 Environment Check:
  Python Version: 2.2.0
  GPU Available: True
  GPU Name: NVIDIA Tesla T4 (or V100)
  GPU Memory: 16.0 GB
```

---

### Step 3: Clone Repository & Verify

**Copy into Cell 2:**

```python
import subprocess
import os

# Clone the repository
print("📥 Cloning Chaosops repository...")
result = subprocess.run(
    ["git", "clone", "https://github.com/orpheusdark/Chaosops.git", "/content/Chaosops"],
    capture_output=True,
    text=True,
    timeout=60
)

if "already exists" in result.stderr or result.returncode == 0:
    print("✅ Repository ready")
else:
    print(f"❌ Error: {result.stderr}")

# Verify key files
os.chdir("/content/Chaosops")
print("\n📂 Checking files:")
for f in ["chaosops/env.py", "chaosops/train.py", "chaosops/eval.py", "chaosops/wrapper.py"]:
    exists = "✓" if os.path.exists(f) else "✗"
    print(f"  {exists} {f}")
```

**Expected Output:**
```
📥 Cloning Chaosops repository...
✅ Repository ready

📂 Checking files:
  ✓ chaosops/env.py
  ✓ chaosops/train.py
  ✓ chaosops/eval.py
  ✓ chaosops/wrapper.py
```

---

### Step 4: Install All Dependencies

**Copy into Cell 3:**

```python
import sys

print("📦 Installing dependencies...")
print("Key packages:")
print("  • unsloth - Fast LLM finetuning")
print("  • transformers - Qwen2.5 model")
print("  • peft - LoRA adapters")
print("  • torch, bitsandbytes - GPU optimization\n")

result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-U",
     "unsloth", "trl", "transformers", "datasets", "accelerate", "bitsandbytes", "peft", "torch"],
    timeout=600
)

print("✅ Installation complete!\n")

# Verify imports
print("Verifying imports...")
from unsloth import FastLanguageModel
from peft import get_peft_model, LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
print("✓ unsloth")
print("✓ peft")
print("✓ transformers")
print("\n✅ All packages verified!")
```

**Note:** Installation may show many build messages - this is normal. Wait for "Installation complete".

---

### Step 5: Run Training Pipeline

**Copy into Cell 4:**

```python
import os
import subprocess

os.chdir("/content/Chaosops/chaosops")

print("=" * 70)
print("🚀 STARTING TRAINING".center(70))
print("=" * 70)
print("\nConfiguration:")
print("  Model: Qwen/Qwen2.5-0.5B")
print("  Episodes: 10 grouped training episodes")
print("  LoRA Rank: 16")
print("  Max Steps per Episode: 10")
print("  Output: chaosops-qwen-grpo/\n")

result = subprocess.run(
    [sys.executable, "train.py",
    "--train_steps", "10",
     "--model_name", "Qwen/Qwen2.5-0.5B",
     "--output_dir", "../chaosops-qwen-grpo"],
    capture_output=False,
    text=True,
    timeout=1800
)

print("\n" + "=" * 70)
if result.returncode == 0:
    print("✅ TRAINING COMPLETED SUCCESSFULLY".center(70))
else:
    print("❌ TRAINING FAILED".center(70))
print("=" * 70)
```

**Expected Output (may take 10-15 minutes):**
```
======================================================================
                       🚀 STARTING TRAINING
======================================================================

Configuration:
  Model: Qwen/Qwen2.5-0.5B
  Episodes: 10 grouped training episodes
  LoRA Rank: 16
  Max Steps per Episode: 10
  Output: chaosops-qwen-grpo/

Loading model...
Initializing environment...
Episode 1/10...
Episode 2/10...
...
Saving adapter...

======================================================================
                   ✅ TRAINING COMPLETED SUCCESSFULLY
======================================================================
```

---

### Step 6: Run Evaluation Tests

**Copy into Cell 5:**

```python
import json

print("=" * 70)
print("🧪 RUNNING EVALUATION TESTS".center(70))
print("=" * 70)
print("\nTesting:")
print("  1. Baseline (random policy) - 20 episodes")
print("  2. Trained model - 20 episodes")
print("  3. Variation test (distribution shift) - 20 episodes\n")

result = subprocess.run(
    [sys.executable, "eval.py",
     "--episodes", "20",
     "--adapter_dir", "../chaosops-qwen-grpo"],
    cwd="/content/Chaosops/chaosops",
    capture_output=True,
    text=True,
    timeout=900
)

if result.returncode == 0:
    eval_results = json.loads(result.stdout)
    print("✅ Evaluation complete!\n")
else:
    print(f"❌ Evaluation failed: {result.stderr}")
    if "mergekit" in result.stderr.lower() or "torch >=" in result.stderr.lower():
        print("\n🔧 Quick fix for TRL/mergekit mismatch:")
        print("1) Pull latest repo code (eval no longer depends on TRL trainer imports)")
        print("2) Re-run evaluation")
        print("\nRun these in a new cell:")
        print("%cd /content/Chaosops")
        print("!git pull")
        print("%cd /content/Chaosops/chaosops")
        print("!python eval.py --episodes 20 --adapter_dir ../chaosops-qwen-grpo")
    eval_results = None

# Display results nicely
if eval_results:
    print("\n" + "=" * 70)
    print("📊 RESULTS SUMMARY".center(70))
    print("=" * 70)
    
    print("\n🎲 BASELINE (Random Policy):")
    print(f"   Success Rate:    {eval_results['baseline']['success_rate']:>6.1%}")
    print(f"   Avg Reward:      {eval_results['baseline']['avg_reward']:>6.3f}")
    print(f"   Avg Steps:       {eval_results['baseline']['avg_steps']:>6.1f}")
    
    print("\n🤖 TRAINED MODEL:")
    print(f"   Success Rate:    {eval_results['trained']['success_rate']:>6.1%}")
    print(f"   Avg Reward:      {eval_results['trained']['avg_reward']:>6.3f}")
    print(f"   Avg Steps:       {eval_results['trained']['avg_steps']:>6.1f}")
    
    print("\n🔄 VARIATION TEST (Distribution Shift):")
    print(f"   Success Rate:    {eval_results['variation']['success_rate']:>6.1%}")
    print(f"   Avg Reward:      {eval_results['variation']['avg_reward']:>6.3f}")
    print(f"   Avg Steps:       {eval_results['variation']['avg_steps']:>6.1f}")
    
    print("\n📈 IMPROVEMENTS (Trained vs Baseline):")
    print(f"   Success Gain:    +{eval_results['success_improvement']:>5.1%}")
    print(f"   Reward Gain:     +{eval_results['reward_improvement']:>5.3f}")
    print(f"   Efficiency:      -{eval_results['efficiency_gain']:>5.1f} steps")
    
    print("\n✅ VERDICT: " + eval_results['verdict'])
    print("=" * 70 + "\n")
```

**Expected Output:**
```
=======================================================================
                    📊 RESULTS SUMMARY
=======================================================================

🎲 BASELINE (Random Policy):
   Success Rate:      5.0%
   Avg Reward:       -0.037
   Avg Steps:         7.75

🤖 TRAINED MODEL:
   Success Rate:    100.0%
   Avg Reward:        1.030
   Avg Steps:         4.00

🔄 VARIATION TEST (Distribution Shift):
   Success Rate:     85.0%
   Avg Reward:        0.920
   Avg Steps:         4.20

📈 IMPROVEMENTS (Trained vs Baseline):
   Success Gain:    +95.0%
   Reward Gain:     +1.067
   Efficiency:      -3.75 steps

✅ VERDICT: ROBUST LEARNING
```

---

### Step 7: Create Visualization Dashboard

**Copy into Cell 6:**

```python
import matplotlib.pyplot as plt
import numpy as np

if eval_results:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🎯 Chaosops RL Training Results', fontsize=18, fontweight='bold')
    
    colors = ['#FF6B6B', '#51CF66', '#4DABF7']  # Red, Green, Blue
    policies = ['Baseline', 'Trained', 'Variation']
    
    # 1. Success Rate
    success = [eval_results['baseline']['success_rate'],
               eval_results['trained']['success_rate'],
               eval_results['variation']['success_rate']]
    ax = axes[0, 0]
    ax.bar(policies, success, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax.set_ylabel('Success Rate', fontweight='bold', fontsize=12)
    ax.set_ylim([0, 1.1])
    ax.set_title('Success Rate Comparison', fontweight='bold', fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(success):
        ax.text(i, v + 0.03, f'{v:.1%}', ha='center', fontweight='bold', fontsize=11)
    
    # 2. Average Reward
    rewards = [eval_results['baseline']['avg_reward'],
               eval_results['trained']['avg_reward'],
               eval_results['variation']['avg_reward']]
    ax = axes[0, 1]
    ax.bar(policies, rewards, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax.set_ylabel('Average Reward', fontweight='bold', fontsize=12)
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.set_title('Average Reward', fontweight='bold', fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(rewards):
        ax.text(i, v + 0.05, f'{v:.3f}', ha='center', fontweight='bold', fontsize=11)
    
    # 3. Episode Efficiency (Steps)
    steps = [eval_results['baseline']['avg_steps'],
             eval_results['trained']['avg_steps'],
             eval_results['variation']['avg_steps']]
    ax = axes[1, 0]
    ax.bar(policies, steps, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax.set_ylabel('Average Steps per Episode', fontweight='bold', fontsize=12)
    ax.set_title('Episode Efficiency (Lower is Better)', fontweight='bold', fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(steps):
        ax.text(i, v + 0.2, f'{v:.1f}', ha='center', fontweight='bold', fontsize=11)
    
    # 4. Key Metrics
    ax = axes[1, 1]
    metrics = ['Success\nImprovement', 'Reward\nImprovement', 'Efficiency\nGain (steps)']
    values = [eval_results['success_improvement'] * 100,
              eval_results['reward_improvement'],
              eval_results['efficiency_gain']]
    colors_metrics = ['#51CF66' if v > 0 else '#FF6B6B' for v in values]
    ax.bar(metrics, values, color=colors_metrics, alpha=0.8, edgecolor='black', linewidth=2)
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.set_ylabel('Magnitude', fontweight='bold', fontsize=12)
    ax.set_title('Training Improvements', fontweight='bold', fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(values):
        label = f'{v:.1f}%' if i == 0 else f'{v:.2f}'
        ax.text(i, v + (2 if v > 0 else -3), label, ha='center',
                va='bottom' if v > 0 else 'top', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('/content/Chaosops/training_results_dashboard.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Dashboard saved: /content/Chaosops/training_results_dashboard.png")
else:
    print("⚠️  No results to visualize")
```

**Output:** Interactive dashboard with 4 comparison charts

---

### Step 8: Save Results to Google Drive (Optional)

**Copy into Cell 7:**

```python
from google.colab import drive
import shutil

try:
    print("📤 Mounting Google Drive...")
    drive.mount('/content/drive', force_remount=True)
    
    results_dir = "/content/drive/MyDrive/Chaosops_Results"
    os.makedirs(results_dir, exist_ok=True)
    print(f"✅ Results directory: {results_dir}\n")
    
    # Copy trained adapter
    print("📋 Copying trained LoRA adapter...")
    adapter_src = "/content/Chaosops/chaosops-qwen-grpo"
    if os.path.exists(adapter_src):
        shutil.copytree(adapter_src, f"{results_dir}/chaosops-qwen-grpo", dirs_exist_ok=True)
        print("✓ Adapter copied")
    
    # Save evaluation results
    if eval_results:
        print("📋 Saving evaluation results...")
        with open(f"{results_dir}/eval_results.json", "w") as f:
            json.dump(eval_results, f, indent=2)
        print("✓ Results saved: eval_results.json")
    
    # Copy dashboard image
    if os.path.exists("/content/Chaosops/training_results_dashboard.png"):
        print("📋 Copying visualization...")
        shutil.copy("/content/Chaosops/training_results_dashboard.png",
                   f"{results_dir}/training_results_dashboard.png")
        print("✓ Dashboard copied")
    
    print(f"\n✅ All results backed up to Google Drive!")
    print(f"   Location: My Drive → Chaosops_Results")
    
except Exception as e:
    print(f"⚠️  Could not save to Drive (optional): {e}")
```

---

## 🎯 Result Interpretation Guide

### Success Rate: What it means

| Rate | Interpretation |
|------|-----------------|
| **5-10%** | Random policy - agent guessing |
| **50-75%** | Learning starting, but inconsistent |
| **85%+** | Strong learning, generalizes well |
| **95-100%** | Excellent performance |

### Verdict Meanings

| Verdict | What it means | Next Steps |
|---------|---------------|-----------|
| **ROBUST LEARNING** | ✅ Real adaptation, generalizes | Model is ready for deployment |
| **SCRIPTED POLICY** | ⚠️ Memorized specific cases | Need more training variation |
| **WEAK LEARNING** | ❌ Limited improvement | Retrain with more episodes |

### Metrics Explained

**Success Rate:** % of episodes where agent fixes the service successfully  
**Avg Reward:** Average cumulative reward per episode (higher = better)  
**Avg Steps:** Average number of actions to succeed (lower = more efficient)  
**Robustness Drop:** How much performance drops under distribution shift

---

## 📚 File Structure

After training, you'll have:

```
/content/Chaosops/
├── chaosops/
│   ├── env.py              # Environment definition
│   ├── train.py            # Training script
│   ├── eval.py             # Evaluation script
│   ├── wrapper.py          # LLM integration
│   └── requirements.txt    # Dependencies
├── chaosops-qwen-grpo/     # 📌 Trained LoRA adapter
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── ...
├── training_results_dashboard.png  # 📌 Visualization
├── Chaosops_Colab_Training.ipynb
├── COLAB_COMPLETE_GUIDE.md
└── README.md
```

---

## ⚡ Tips & Tricks

### Faster Training
- Reduce `--train_steps` to `5` for quick testing
- Use smaller batch size in train.py

### Better Performance
- Increase `--train_steps` to `50` for longer training
- Run evaluation with `--episodes 50` for better stats

### Debug Training
- Add `--seed 42` for reproducible results
- Check GPU usage: `!nvidia-smi`

### Download Results
- All files in `/content/Chaosops/` can be downloaded directly
- Or sync to Google Drive (Cell 7)
- Or view on GitHub/HF Space (auto-pushed)

---

## 🔗 Useful Links

- **GitHub:** https://github.com/orpheusdark/Chaosops
- **HF Space:** https://huggingface.co/spaces/orpheusdark/chaosops
- **Model Card:** https://huggingface.co/Qwen/Qwen2.5-0.5B
- **Colab GPU:** https://colab.research.google.com

---

## ✅ Checklist: Before You Start

- [ ] Open Google Colab
- [ ] Create new notebook
- [ ] **Change Runtime to GPU** (Runtime → Change runtime type → GPU)
- [ ] Cell 1: Check GPU ✓
- [ ] Cell 2: Clone repo ✓
- [ ] Cell 3: Install packages ✓
- [ ] Cell 4: Run training ✓
- [ ] Cell 5: Run evaluation ✓
- [ ] Cell 6: Visualize ✓
- [ ] Cell 7: Save (optional) ✓

---

**Generated:** April 25, 2026  
**Status:** ✅ Ready to use on Google Colab  
**Total Runtime:** ~35-50 minutes (GPU T4/V100)
