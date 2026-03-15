import pytest
import numpy as np
from src.training.dpo_trainer import DPOTrainer, DPOConfig
from src.training.reward_model import ReasoningRewardModel
from src.data.loader import ReasoningDataset

def test_dpo_loss_computation():
    trainer = DPOTrainer()
    metrics = trainer.compute_dpo_loss(
        np.array([-0.3, -0.2]), np.array([-0.8, -0.9]),
        np.array([-0.4, -0.3]), np.array([-0.7, -0.8]),
    )
    assert "loss" in metrics
    assert "accuracy" in metrics
    assert metrics["reward_margin"] > 0

def test_dpo_training():
    dataset = ReasoningDataset().generate_dataset(50)
    trainer = DPOTrainer(DPOConfig(max_epochs=1, batch_size=10))
    result = trainer.train(dataset)
    assert result["total_steps"] > 0
    assert result["final_loss"] > 0

def test_reward_model():
    rm = ReasoningRewardModel()
    trajectory = "Step 1: 5 + 3 = 8 -> Step 2: 8 * 2 = 16 -> Final: 16"
    reward = rm.compute_reward(trajectory, reference="16")
    assert reward["total_reward"] > 0
    assert reward["correctness"] > 0.5

def test_dataset_generation():
    ds = ReasoningDataset()
    data = ds.generate_dataset(20)
    assert len(data) == 20
    assert all("prompt" in d and "chosen" in d for d in data)
