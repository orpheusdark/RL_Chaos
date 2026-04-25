---
title: ChaosOps
emoji: "🛠️"
colorFrom: blue
colorTo: green
sdk: docker
sdk_version: "latest"
python_version: "3.10"
app_file: app.py
pinned: false
app_port: 8000
---

# ChaosOps

ChaosOps is a runnable OpenEnv-style RL environment for multi-step LLM agent recovery workflows.

## What It Simulates

1. Auth service starts crashed (`OOMKilled`)
2. Agent inspects the system
3. Agent requests access token with proper justification
4. API schema drifts mid-episode (v1 -> v2)
5. Agent adapts and fixes service

## Local Run

```powershell
cd chaosops
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

## Training Guide

Use one canonical document for Colab training and evaluation:

- `COLAB_COMPLETE_GUIDE.md`

## Auto Benchmark + Progress Chart

```powershell
python .\scripts\auto_run_and_chart.py
```

Generated artifacts:

- `charts/rl_progress.svg`
- `charts/rl_progress.json`

## API Smoke Test

```powershell
$resetBody = @{ task = 'task3' } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/reset -ContentType 'application/json' -Body $resetBody
```

## Web UI

Open the dashboard at:

- [`/ui`](https://orpheusdark-chaosops.hf.space/ui)

It provides a guided mission console with reset, step actions, schema-aware fix flow, and live state timeline.

## Architecture Visual

```mermaid
flowchart LR
    A[Agent Policy] --> B[OpenEnv: ChaosOpsEnv]
    B --> C[Action: query_system]
    B --> D[Action: request_access]
    B --> E[Action: get_schema]
    B --> F[Action: fix_service]
    B --> G[State Machine]
    G --> G1[service_status]
    G --> G2[error_log]
    G --> G3[api_schema_version]
    G --> G4[has_permission]
    G --> G5[step_count]
    B --> H[Reward Clamp 0.01..0.99]
    H --> I[TRL GRPO Training Loop]
    I --> A
```

## GitHub Push

```powershell
git add .
git commit -m "Update ChaosOps run/train settings and deployment docs"
git push origin main
```

## Hugging Face Space Push (Docker)

1. Create a new Space with SDK = Docker.
2. Clone your Space repo locally.
3. Copy this project into the Space repository root.
4. Push to Hugging Face:

```powershell
git remote add hf https://huggingface.co/spaces/<HF_USERNAME>/chaosops
git push hf main
```

If prompted, use your Hugging Face username and a write token as password.
