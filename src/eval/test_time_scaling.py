"""Test-time scaling evaluation across compute budgets."""
import numpy as np
from typing import Optional

class TestTimeScaler:
    """Evaluate model accuracy across varying compute budgets."""

    def __init__(self, base_accuracy: float = 0.6):
        self.base_accuracy = base_accuracy

    def evaluate_compute_budget(self, budget: int, problem_difficulty: str = "medium") -> dict:
        difficulty_multiplier = {"easy": 1.3, "medium": 1.0, "hard": 0.7}
        mult = difficulty_multiplier.get(problem_difficulty, 1.0)
        accuracy = self.base_accuracy * mult * (1 - np.exp(-budget / 5))
        accuracy = min(accuracy, 0.98)

        return {
            "compute_budget": budget,
            "difficulty": problem_difficulty,
            "accuracy": round(float(accuracy), 4),
            "tokens_per_problem": budget * 50,
            "cost_efficiency": round(float(accuracy / (budget + 1)), 4),
        }

    def scaling_curve(self, budgets: list[int], difficulty: str = "medium") -> list[dict]:
        return [self.evaluate_compute_budget(b, difficulty) for b in budgets]

    def find_optimal_budget(self, target_accuracy: float = 0.85, difficulty: str = "medium") -> dict:
        for budget in range(1, 50):
            result = self.evaluate_compute_budget(budget, difficulty)
            if result["accuracy"] >= target_accuracy:
                return {"optimal_budget": budget, **result}
        return {"optimal_budget": 50, "note": "Target accuracy not achievable"}
