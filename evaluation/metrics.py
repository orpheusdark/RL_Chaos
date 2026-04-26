"""Metrics and reporting utilities for ChaosOps-RC."""

import json
from typing import Any, Dict, List, Optional


def format_metrics_table(stats: Dict[str, Any]) -> str:
    """Format evaluation metrics as a markdown table.

    Args:
        stats: Stats dict from evaluate_agent

    Returns:
        Markdown formatted table
    """
    lines = [
        "| Metric | Value |",
        "|--------|-------|",
        f"| Episodes | {stats.get('num_episodes', 'N/A')} |",
        f"| Mean Reward | {stats.get('mean_reward', 0):.3f} |",
        f"| Max Reward | {stats.get('max_reward', 0):.3f} |",
        f"| Min Reward | {stats.get('min_reward', 0):.3f} |",
        f"| Std Dev | {stats.get('std_reward', 0):.3f} |",
        f"| Mean Episode Length | {stats.get('mean_episode_length', 0):.1f} |",
        f"| Success Rate | {stats.get('success_rate', 0):.1%} |",
        f"| Failure Rate | {stats.get('failure_rate', 0):.1%} |",
    ]
    return "\n".join(lines)


def format_comparison_table(comparison: Dict[str, Any]) -> str:
    """Format agent comparison as markdown table.

    Args:
        comparison: Comparison dict from compare_agents

    Returns:
        Markdown formatted comparison
    """
    baseline = comparison.get("baseline", {})
    trained = comparison.get("trained", {})
    improvement = comparison.get("improvement", {})

    lines = [
        "| Metric | Baseline | Trained | Delta |",
        "|--------|----------|---------|-------|",
        f"| Mean Reward | {baseline.get('mean_reward', 0):.3f} | {trained.get('mean_reward', 0):.3f} | {improvement.get('reward_delta', 0):+.3f} |",
        f"| Success Rate | {baseline.get('success_rate', 0):.1%} | {trained.get('success_rate', 0):.1%} | {improvement.get('success_rate_delta', 0):+.1%} |",
        f"| Mean Episode Length | {baseline.get('mean_episode_length', 0):.1f} | {trained.get('mean_episode_length', 0):.1f} | {improvement.get('efficiency_delta', 0):+.1f} |",
    ]
    return "\n".join(lines)


def save_metrics_json(stats: Dict[str, Any], filepath: str) -> None:
    """Save metrics to JSON file.

    Args:
        stats: Metrics dict
        filepath: File path to save to
    """
    with open(filepath, "w") as f:
        json.dump(stats, f, indent=2)


def generate_report(
    baseline_stats: Dict[str, Any],
    trained_stats: Dict[str, Any],
    comparison: Dict[str, Any],
) -> str:
    """Generate a full evaluation report.

    Args:
        baseline_stats: Baseline agent stats
        trained_stats: Trained agent stats
        comparison: Comparison dict

    Returns:
        Formatted report string
    """
    report = []
    report.append("# ChaosOps-RC Evaluation Report\n")

    report.append("## Baseline Agent\n")
    report.append(format_metrics_table(baseline_stats))
    report.append("\n")

    report.append("## Trained Agent\n")
    report.append(format_metrics_table(trained_stats))
    report.append("\n")

    report.append("## Comparison\n")
    report.append(format_comparison_table(comparison))
    report.append("\n")

    improvement = comparison.get("improvement", {})
    reward_improvement = improvement.get("reward_delta", 0)
    success_improvement = improvement.get("success_rate_delta", 0)

    report.append("## Key Findings\n\n")
    if reward_improvement > 0:
        report.append(f"- **Reward Improvement**: +{reward_improvement:.3f} ({reward_improvement/abs(baseline_stats.get('mean_reward', 1)):+.1%})\n")
    else:
        report.append(f"- **Reward Decline**: {reward_improvement:.3f}\n")

    if success_improvement > 0:
        report.append(f"- **Success Rate Improvement**: +{success_improvement:.1%}\n")
    else:
        report.append(f"- **Success Rate Decline**: {success_improvement:.1%}\n")

    report.append("\n## Curriculum Levels\n\n")
    report.append(f"- Baseline Level: {baseline_stats.get('curriculum_level', 'N/A')}\n")
    report.append(f"- Trained Level: {trained_stats.get('curriculum_level', 'N/A')}\n")

    return "".join(report)
