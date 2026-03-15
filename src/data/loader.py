"""Dataset loaders for long-horizon reasoning tasks."""
import json
import random
import numpy as np
from pathlib import Path
from typing import Optional

class ReasoningDataset:
    """Generates and manages reasoning task data for RLHF training."""

    TASK_TEMPLATES = {
        "arithmetic_chain": {
            "description": "Multi-step arithmetic reasoning",
            "steps_range": (3, 8),
        },
        "logic_deduction": {
            "description": "Logical deduction with multiple premises",
            "steps_range": (4, 10),
        },
        "code_debugging": {
            "description": "Step-by-step code debugging",
            "steps_range": (3, 7),
        },
    }

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

    def generate_arithmetic_chain(self, steps: int = 5) -> dict:
        numbers = [self.rng.randint(1, 100) for _ in range(steps + 1)]
        operations = [self.rng.choice(["+", "-", "*"]) for _ in range(steps)]
        reasoning_chain = []
        result = numbers[0]
        for i, (op, num) in enumerate(zip(operations, numbers[1:])):
            prev = result
            if op == "+": result += num
            elif op == "-": result -= num
            elif op == "*": result *= num
            reasoning_chain.append(f"Step {i+1}: {prev} {op} {num} = {result}")

        return {
            "question": f"Compute: {numbers[0]} " + " ".join(f"{op} {n}" for op, n in zip(operations, numbers[1:])),
            "answer": str(result),
            "reasoning_chain": reasoning_chain,
            "num_steps": steps,
            "task_type": "arithmetic_chain",
        }

    def generate_dataset(self, n_samples: int = 1000, task_type: str = "arithmetic_chain") -> list[dict]:
        dataset = []
        for _ in range(n_samples):
            steps = self.rng.randint(3, 8)
            sample = self.generate_arithmetic_chain(steps)
            # Generate chosen (correct) and rejected (incorrect) pairs for DPO
            chosen = " -> ".join(sample["reasoning_chain"]) + f" -> Final: {sample['answer']}"
            wrong_answer = int(sample["answer"]) + self.rng.randint(-50, 50)
            rejected = " -> ".join(sample["reasoning_chain"][:2]) + f" -> Final: {wrong_answer}"
            dataset.append({
                "prompt": sample["question"],
                "chosen": chosen,
                "rejected": rejected,
                "num_steps": sample["num_steps"],
            })
        return dataset

    def split_dataset(self, data: list, train_ratio: float = 0.8) -> dict:
        self.rng.shuffle(data)
        split_idx = int(len(data) * train_ratio)
        return {"train": data[:split_idx], "val": data[split_idx:]}


def load_reasoning_dataset(n_samples: int = 1000) -> list[dict]:
    ds = ReasoningDataset()
    return ds.generate_dataset(n_samples)
