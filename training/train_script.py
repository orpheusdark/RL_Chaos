"""Training script for ChaosOps-RC with TRL integration.

This script demonstrates how to train an LLM agent on the ChaosOps environment
using HuggingFace TRL (GRPO or PPO trainer).

For full integration, this would use:
- transformers.AutoModelForCausalLM
- trl.GRPOTrainer or trl.PPOTrainer
- Unsloth for efficient inference

Current version provides a training template.
"""

import json
import argparse
import random
from typing import Optional, Dict, Any

from .utils import EpisodeCollector, batch_generator, format_observation_for_llm


def train(
    num_episodes: int = 100,
    curriculum_level: int = 1,
    batch_size: int = 16,
    seed: Optional[int] = None,
    output_dir: str = "chaosops-rc-checkpoint",
) -> Dict[str, Any]:
    """Train an agent on ChaosOps-RC.

    Args:
        num_episodes: Total episodes to collect
        curriculum_level: Difficulty level (1-4)
        batch_size: Training batch size
        seed: Random seed
        output_dir: Where to save checkpoints

    Returns:
        Training results dict
    """
    print(f"Training ChaosOps-RC Agent")
    print(f"  Curriculum Level: {curriculum_level}")
    print(f"  Episodes: {num_episodes}")
    print(f"  Batch Size: {batch_size}")

    # Initialize episode collector
    collector = EpisodeCollector(
        curriculum_level=curriculum_level,
        seed=seed,
    )

    # Collect episodes
    print("\nCollecting episodes...")
    episodes = collector.collect_episodes(num_episodes)

    # Print statistics
    stats = collector.get_buffer_stats()
    print(f"Collected {stats['num_episodes']} episodes")
    print(f"  Avg reward: {stats['avg_reward']:.3f}")
    print(f"  Avg length: {stats['avg_length']:.1f}")
    print(f"  Total steps: {stats['total_steps']}")

    # Batch episodes for training
    print("\nPreparing batches...")
    batches = list(batch_generator(episodes, batch_size))
    print(f"  {len(batches)} batches of size {batch_size}")

    # In a full TRL implementation, you would:
    # 1. Create a model (using transformers.AutoModelForCausalLM)
    # 2. Create a trainer (GRPOTrainer or PPOTrainer from trl)
    # 3. Implement a reward function callback
    # 4. Train with trainer.train()

    # For now, just demonstrate the data format
    print("\nSample episode trajectory (first step):")
    first_episode = episodes[0]
    first_step = first_episode["trajectory"][0]
    print(f"  Observation keys: {list(first_step['observation'].keys())}")
    print(f"  Action: {first_step['action']}")
    print(f"  Reward: {first_step['reward']:.4f}")

    # Save results
    results = {
        "num_episodes_collected": stats["num_episodes"],
        "curriculum_level": curriculum_level,
        "mean_reward": stats["avg_reward"],
        "max_reward": stats["max_reward"],
        "mean_episode_length": stats["avg_length"],
        "total_steps": stats["total_steps"],
        "batch_size": batch_size,
        "output_dir": output_dir,
    }

    # Save results to file
    with open(f"{output_dir}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}_results.json")

    return results


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train ChaosOps-RC agent with TRL")
    parser.add_argument("--num-episodes", type=int, default=100, help="Number of episodes to collect")
    parser.add_argument("--curriculum-level", type=int, default=1, help="Difficulty level (1-4)")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="chaosops-rc-checkpoint", help="Output directory")

    args = parser.parse_args()

    # Run training
    results = train(
        num_episodes=args.num_episodes,
        curriculum_level=args.curriculum_level,
        batch_size=args.batch_size,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    print("\nTraining complete!")
    return results


if __name__ == "__main__":
    main()
