from __future__ import annotations

from env import ChaosOpsEnv


def run_task() -> float:
    env = ChaosOpsEnv()
    env.reset("task1")

    env.step("query_system", {})
    access = env.step("request_access", {"justification": "Need access to resolve OOM crash"})
    token = access.get("result", {}).get("token", "")
    fixed = env.step(
        "fix_service",
        {
            "config": {
                "service": "auth_service",
                "status": "running",
                "cpu_limit": "1",
            },
            "token": token,
        },
    )

    success = fixed.get("result", {}).get("ok", False) and fixed.get("state", {}).get("service_status") == "running"
    score = fixed.get("score") if fixed.get("done") else max(0.01, min(0.99, fixed.get("total_reward", 0.0)))
    if not success:
        return 0.01
    return float(max(0.01, min(0.99, score)))


if __name__ == "__main__":
    print(run_task())
