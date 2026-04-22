from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from env import ChaosOpsEnv


app = FastAPI(title="ChaosOps", version="1.0.0")
env = ChaosOpsEnv()


class ResetRequest(BaseModel):
    task: str = Field(default="task3")


class StepRequest(BaseModel):
    action: str
    payload: Optional[Dict[str, Any]] = Field(default_factory=dict)


@app.post("/reset")
def reset_endpoint(request: ResetRequest) -> Dict[str, Any]:
    try:
        return env.reset(task_name=request.task)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"unexpected error: {exc}") from exc


@app.post("/step")
def step_endpoint(request: StepRequest) -> Dict[str, Any]:
    try:
        return env.step(action=request.action, payload=request.payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"unexpected error: {exc}") from exc
