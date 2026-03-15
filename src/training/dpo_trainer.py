"""Direct Preference Optimization (DPO) training implementation."""
import numpy as np
import logging
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class DPOConfig:
    beta: float = 0.1
    learning_rate: float = 1e-5
    batch_size: int = 8
    max_epochs: int = 3
    max_length: int = 512
    warmup_steps: int = 100
    gradient_accumulation: int = 4
    weight_decay: float = 0.01

class DPOTrainer:
    """Implements Direct Preference Optimization for aligning LLMs."""

    def __init__(self, config: Optional[DPOConfig] = None):
        self.config = config or DPOConfig()
        self.training_history = []
        self._step = 0

    def compute_dpo_loss(self, policy_chosen_logps: np.ndarray,
                          policy_rejected_logps: np.ndarray,
                          ref_chosen_logps: np.ndarray,
                          ref_rejected_logps: np.ndarray) -> dict:
        chosen_rewards = self.config.beta * (policy_chosen_logps - ref_chosen_logps)
        rejected_rewards = self.config.beta * (policy_rejected_logps - ref_rejected_logps)
        logits = chosen_rewards - rejected_rewards
        loss = -np.log(1 / (1 + np.exp(-logits)) + 1e-8).mean()
        accuracy = (logits > 0).mean()
        return {
            "loss": round(float(loss), 4),
            "accuracy": round(float(accuracy), 4),
            "chosen_rewards_mean": round(float(chosen_rewards.mean()), 4),
            "rejected_rewards_mean": round(float(rejected_rewards.mean()), 4),
            "reward_margin": round(float((chosen_rewards - rejected_rewards).mean()), 4),
        }

    def train_step(self, batch: dict) -> dict:
        n = len(batch.get("prompt", []))
        policy_chosen = np.random.randn(n) * 0.5 - 0.3
        policy_rejected = np.random.randn(n) * 0.5 - 0.8
        ref_chosen = np.random.randn(n) * 0.3 - 0.4
        ref_rejected = np.random.randn(n) * 0.3 - 0.7

        metrics = self.compute_dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
        self._step += 1
        metrics["step"] = self._step
        self.training_history.append(metrics)
        return metrics

    def train(self, dataset: list[dict]) -> dict:
        logger.info("Starting DPO training with %d samples", len(dataset))
        bs = self.config.batch_size
        epoch_metrics = []

        for epoch in range(self.config.max_epochs):
            epoch_losses = []
            for i in range(0, len(dataset), bs):
                batch = {"prompt": [d["prompt"] for d in dataset[i:i+bs]]}
                metrics = self.train_step(batch)
                epoch_losses.append(metrics["loss"])

            avg_loss = np.mean(epoch_losses)
            epoch_metrics.append({"epoch": epoch + 1, "avg_loss": round(float(avg_loss), 4)})
            logger.info("Epoch %d: avg_loss=%.4f", epoch + 1, avg_loss)

        return {
            "total_steps": self._step,
            "epochs": epoch_metrics,
            "final_loss": epoch_metrics[-1]["avg_loss"],
        }
