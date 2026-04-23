from __future__ import annotations

from copy import deepcopy
from threading import Lock
from typing import Any, Dict, Optional


class ChaosOpsEnv:
    def __init__(self, max_steps: int = 8) -> None:
        if not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        self.max_steps = max_steps
        self._lock = Lock()
        self.task_name = "task3"
        self.task_settings = {
            "task1": {"schema_drift": False, "permission_required": False},
            "task2": {"schema_drift": True, "permission_required": False},
            "task3": {"schema_drift": True, "permission_required": True},
        }
        self.state: Dict[str, Any] = {}
        self.total_reward = 0.0
        self.reset("task3")

    def _clamp_score(self, reward: float) -> float:
        return max(0.01, min(0.99, reward))

    def _get_schema_definition(self) -> Dict[str, Any]:
        if self.state["api_schema_version"] == 1:
            return {
                "version": 1,
                "required_config_keys": ["service", "status", "cpu_limit"],
                "observation_keys": ["service", "status", "cpu_limit"],
            }
        return {
            "version": 2,
            "required_config_keys": ["service_name", "condition", "max_compute"],
            "observation_keys": ["service_name", "condition", "max_compute"],
        }

    def _build_observation(self) -> Dict[str, Any]:
        if self.state["api_schema_version"] == 1:
            return {
                "service": "auth_service",
                "status": self.state["service_status"],
                "cpu_limit": "500m",
            }
        return {
            "service_name": "auth_service",
            "condition": self.state["service_status"],
            "max_compute": "1",
        }

    def _maybe_drift_schema(self) -> None:
        should_drift = self.task_settings.get(self.task_name, {}).get("schema_drift", True)
        if should_drift and self.state["step_count"] == 2 and self.state["api_schema_version"] == 1:
            self.state["api_schema_version"] = 2
            self.state["drift_active"] = True

    def _ensure_token(self, token: str) -> bool:
        return token == self.state.get("access_token") and bool(token)

    def _permission_required(self) -> bool:
        return bool(self.task_settings.get(self.task_name, {}).get("permission_required", True))

    def reset(self, task_name: str = "task3") -> Dict[str, Any]:
        if task_name not in self.task_settings:
            raise ValueError(f"unknown task_name: {task_name}")

        with self._lock:
            self.task_name = task_name
            self.state = {
                "service_status": "crashed",
                "error_log": "OOMKilled",
                "api_schema_version": 1,
                "has_permission": False,
                "step_count": 0,
                "schema_fail_count": 0,
                "drift_active": False,
                "used_schema_after_drift": False,
                "trajectory": [],
            }
            self.total_reward = 0.0
            return {
                "ok": True,
                "task": self.task_name,
                "state": deepcopy(self.state),
                "observation": self._build_observation(),
            }

    def get_schema(self) -> Dict[str, Any]:
        if self.state.get("drift_active", False) and self.state["api_schema_version"] == 2:
            self.state["used_schema_after_drift"] = True
        return {
            "ok": True,
            "schema": self._get_schema_definition(),
        }

    def query_system(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "error_log": self.state["error_log"],
            "observation": self._build_observation(),
        }

    def request_access(self, justification: str) -> Dict[str, Any]:
        if not isinstance(justification, str) or not justification.strip():
            return {
                "ok": False,
                "error_code": "INVALID_JUSTIFICATION",
                "message": "justification must be a non-empty string",
            }

        lowered = justification.lower()
        approved = "oom" in lowered or "crash" in lowered
        if approved:
            self.state["has_permission"] = True
            self.state["access_token"] = "ops-approved-token"
            return {
                "ok": True,
                "token": self.state["access_token"],
                "message": "access granted",
            }

        return {
            "ok": False,
            "error_code": "ACCESS_DENIED",
            "message": "justification did not mention oom/crash",
        }

    def fix_service(self, config: Dict[str, Any], token: str) -> Dict[str, Any]:
        if not isinstance(config, dict):
            return {
                "ok": False,
                "error_code": "INVALID_CONFIG",
                "message": "config must be a dict",
            }
        if self._permission_required():
            if not isinstance(token, str) or not token.strip():
                return {
                    "ok": False,
                    "error_code": "INVALID_TOKEN",
                    "message": "token must be a non-empty string",
                }
            if not self.state.get("has_permission", False) or not self._ensure_token(token.strip()):
                return {
                    "ok": False,
                    "error_code": "NO_PERMISSION",
                    "message": "permission required before fix",
                }

        required_keys = set(self._get_schema_definition()["required_config_keys"])
        provided_keys = set(config.keys())
        if required_keys != provided_keys:
            return {
                "ok": False,
                "error_code": "WRONG_SCHEMA",
                "message": "config keys do not match current schema",
                "expected_keys": sorted(required_keys),
                "provided_keys": sorted(provided_keys),
            }

        self.state["service_status"] = "running"
        self.state["error_log"] = ""
        return {
            "ok": True,
            "message": "service recovered",
            "service_status": self.state["service_status"],
        }

    def step(self, action: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            raise ValueError("payload must be a dict")
        if not isinstance(action, str) or not action.strip():
            raise ValueError("action must be a non-empty string")

        with self._lock:
            self.state["step_count"] += 1
            self._maybe_drift_schema()

            reward_delta = 0.0
            result: Dict[str, Any]

            action = action.strip()
            if action == "query_system":
                result = self.query_system()
                reward_delta -= 0.02
            elif action == "get_schema":
                result = self.get_schema()
                if result.get("ok"):
                    reward_delta += 0.1
                    if self.state.get("drift_active", False):
                        reward_delta += 0.15
            elif action == "request_access":
                result = self.request_access(payload.get("justification", ""))
                if result.get("ok"):
                    reward_delta += 0.2
                else:
                    reward_delta -= 0.1
            elif action == "fix_service":
                if self.state.get("drift_active", False) and not self.state.get("used_schema_after_drift", False):
                    reward_delta -= 0.3
                result = self.fix_service(
                    config=payload.get("config", {}),
                    token=payload.get("token", ""),
                )
                if result.get("ok"):
                    reward_delta += 0.6
                elif result.get("error_code") == "WRONG_SCHEMA":
                    self.state["schema_fail_count"] += 1
                    reward_delta -= 0.2 * self.state["schema_fail_count"]
                elif result.get("error_code") == "NO_PERMISSION":
                    reward_delta -= 0.2
            else:
                result = {
                    "ok": False,
                    "error_code": "UNKNOWN_ACTION",
                    "message": f"unknown action: {action}",
                }
                reward_delta -= 0.05

            self.total_reward += reward_delta

            trace_event = {
                "step": self.state["step_count"],
                "action": action,
                "reward_delta": reward_delta,
                "total_reward": self.total_reward,
                "schema_version": self.state["api_schema_version"],
                "result_ok": bool(result.get("ok", False)),
                "error_code": result.get("error_code"),
            }
            self.state["trajectory"].append(trace_event)

            done = self.state["service_status"] == "running" or self.state["step_count"] >= self.max_steps
            score = self._clamp_score(self.total_reward) if done else None

            return {
                "ok": True,
                "action": action,
                "result": result,
                "reward_delta": reward_delta,
                "total_reward": self.total_reward,
                "done": done,
                "score": score,
                "state": deepcopy(self.state),
                "trace_event": trace_event,
            }
