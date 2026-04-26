"""FastAPI deployment for ChaosOps-RC.

Provides REST API for OpenEnv submission and interactive evaluation.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional

from envs import ChaosOpsRCEnv

# Initialize FastAPI app
app = FastAPI(
    title="ChaosOps-RC",
    description="Multi-service RL environment for distributed system recovery",
    version="1.0.0",
)

# Global environment instance
environment: Optional[ChaosOpsRCEnv] = None


class ResetRequest(BaseModel):
    """Reset request."""
    curriculum_level: int = 1
    seed: Optional[int] = None
    max_steps: int = 50


class ActionRequest(BaseModel):
    """Action request."""
    action: str
    params: Dict[str, Any] = {}


@app.on_event("startup")
async def startup_event():
    """Initialize environment on startup."""
    global environment
    environment = ChaosOpsRCEnv(curriculum_level=1, seed=42)
    environment.reset()


@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "running",
        "mode": "headless",
        "version": "1.0.0",
        "description": "ChaosOps-RC OpenEnv submission",
    }


@app.post("/reset")
async def reset(request: ResetRequest):
    """Reset the environment.

    Args:
        request: Reset parameters

    Returns:
        Initial observation
    """
    global environment
    try:
        environment = ChaosOpsRCEnv(
            curriculum_level=request.curriculum_level,
            seed=request.seed,
            max_steps=request.max_steps,
        )
        obs = environment.reset()
        return {
            "ok": True,
            "observation": obs,
            "curriculum_level": request.curriculum_level,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
async def step(request: ActionRequest):
    """Execute one step of the environment.

    Args:
        request: Action to execute

    Returns:
        Observation, reward, done flag, and info
    """
    global environment
    if environment is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")

    try:
        action = {
            "action": request.action,
            "params": request.params,
        }

        obs, reward, done, info = environment.step(action)

        return {
            "ok": True,
            "observation": obs,
            "reward": reward,
            "done": done,
            "info": info,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
async def get_state():
    """Get current environment state (for debugging).

    Returns:
        Current state and observation
    """
    global environment
    if environment is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")

    try:
        return {
            "ok": True,
            "step_count": environment.step_count,
            "episode_reward": environment.episode_reward,
            "services": {
                sid: service.to_dict()
                for sid, service in environment.services.items()
            },
            "observation": environment.get_observation(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/info")
async def get_info():
    """Get environment info."""
    global environment
    if environment is None:
        raise HTTPException(status_code=400, detail="Environment not initialized.")

    return {
        "ok": True,
        "max_steps": environment.max_steps,
        "curriculum_level": environment.curriculum_level,
        "num_services": len(environment.services),
        "services": list(environment.services.keys()),
        "failure_types": environment.failure_types,
        "valid_actions": list(environment.valid_actions),
    }


@app.get("/metrics")
async def get_metrics():
    """Get episode metrics."""
    global environment
    if environment is None:
        raise HTTPException(status_code=400, detail="Environment not initialized.")

    return {
        "ok": True,
        "step_count": environment.step_count,
        "episode_reward": environment.episode_reward,
        "system_health": environment.system_graph.compute_system_health() if environment.system_graph else 0.0,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
