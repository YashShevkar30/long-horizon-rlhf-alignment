"""Reward model for step-by-step reasoning verification."""
import numpy as np
import re
from typing import Optional

class ReasoningRewardModel:
    """Scores reasoning trajectories for correctness and quality."""

    QUALITY_WEIGHTS = {
        "step_count_match": 0.25,
        "logical_coherence": 0.25,
        "answer_correctness": 0.30,
        "explanation_depth": 0.20,
    }

    def compute_reward(self, trajectory: str, reference: Optional[str] = None) -> dict:
        steps = self._parse_steps(trajectory)
        step_score = self._score_step_quality(steps)
        coherence = self._check_coherence(steps)
        depth = self._score_depth(trajectory)

        correctness = 1.0
        if reference:
            correctness = self._check_correctness(trajectory, reference)

        total = (
            self.QUALITY_WEIGHTS["step_count_match"] * step_score +
            self.QUALITY_WEIGHTS["logical_coherence"] * coherence +
            self.QUALITY_WEIGHTS["answer_correctness"] * correctness +
            self.QUALITY_WEIGHTS["explanation_depth"] * depth
        )

        return {
            "total_reward": round(total, 4),
            "step_quality": round(step_score, 4),
            "coherence": round(coherence, 4),
            "correctness": round(correctness, 4),
            "depth": round(depth, 4),
            "num_steps": len(steps),
        }

    def _parse_steps(self, trajectory: str) -> list[str]:
        steps = re.split(r"Step \d+:|->|\n", trajectory)
        return [s.strip() for s in steps if s.strip()]

    def _score_step_quality(self, steps: list) -> float:
        if not steps:
            return 0.0
        quality = min(len(steps) / 5.0, 1.0)
        avg_length = np.mean([len(s.split()) for s in steps])
        length_bonus = min(avg_length / 10.0, 1.0)
        return (quality + length_bonus) / 2

    def _check_coherence(self, steps: list) -> float:
        if len(steps) < 2:
            return 0.5
        has_numbers = sum(1 for s in steps if re.search(r"\d+", s))
        return min(has_numbers / len(steps), 1.0)

    def _check_correctness(self, trajectory: str, reference: str) -> float:
        traj_numbers = re.findall(r"\d+", trajectory)
        ref_numbers = re.findall(r"\d+", reference)
        if not ref_numbers:
            return 0.5
        if traj_numbers and traj_numbers[-1] == ref_numbers[-1]:
            return 1.0
        return 0.2

    def _score_depth(self, trajectory: str) -> float:
        words = len(trajectory.split())
        return min(words / 100, 1.0)
