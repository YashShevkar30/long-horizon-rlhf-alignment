"""Comprehensive evaluation metrics for reasoning tasks."""
import numpy as np
from typing import Optional

def calculate_accuracy(predictions: list, labels: list) -> float:
    return sum(p == l for p, l in zip(predictions, labels)) / max(len(labels), 1)

def calculate_reasoning_metrics(predicted_chains: list[list[str]], 
                                 reference_chains: list[list[str]]) -> dict:
    step_accuracies = []
    chain_accuracies = []
    for pred, ref in zip(predicted_chains, reference_chains):
        min_len = min(len(pred), len(ref))
        step_correct = sum(p == r for p, r in zip(pred[:min_len], ref[:min_len]))
        step_accuracies.append(step_correct / max(len(ref), 1))
        chain_accuracies.append(1.0 if pred == ref else 0.0)

    return {
        "step_accuracy": round(float(np.mean(step_accuracies)), 4),
        "chain_accuracy": round(float(np.mean(chain_accuracies)), 4),
        "avg_steps_predicted": round(float(np.mean([len(p) for p in predicted_chains])), 1),
        "avg_steps_reference": round(float(np.mean([len(r) for r in reference_chains])), 1),
    }

def pass_at_k(n_samples: int, n_correct: int, k: int) -> float:
    if n_samples - n_correct < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n_samples - n_correct + 1, n_samples + 1))
