"""Evaluation framework for ChaosOps-RC."""

from .eval_script import BaselineAgent, evaluate_agent, compare_agents
from .metrics import (
    format_metrics_table,
    format_comparison_table,
    save_metrics_json,
    generate_report,
)
from .robustness_eval import (
    RobustnessEvaluator,
    benchmark_baseline,
    run_holdout_chaos_set,
    get_tier_definitions,
)

__all__ = [
    "BaselineAgent",
    "evaluate_agent",
    "compare_agents",
    "format_metrics_table",
    "format_comparison_table",
    "save_metrics_json",
    "generate_report",
    "RobustnessEvaluator",
    "benchmark_baseline",
    "run_holdout_chaos_set",
    "get_tier_definitions",
]
