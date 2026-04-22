from __future__ import annotations

from env import ChaosOpsEnv


def run_task() -> float:
    env = ChaosOpsEnv()
    env.reset("task2")

    env.step("query_system", {})
    schema = env.step("get_schema", {})
    version = schema.get("result", {}).get("schema", {}).get("version", 1)

    access = env.step("request_access", {"justification": "System crash caused by oom, requesting token"})
    token = access.get("result", {}).get("token", "")

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

    fixed = env.step("fix_service", {"config": config, "token": token})
    success = fixed.get("result", {}).get("ok", False) and fixed.get("state", {}).get("service_status") == "running"
    score = fixed.get("score") if fixed.get("done") else max(0.01, min(0.99, fixed.get("total_reward", 0.0)))
    if not success:
        return 0.01
    return float(max(0.01, min(0.99, score)))


if __name__ == "__main__":
    print(run_task())
