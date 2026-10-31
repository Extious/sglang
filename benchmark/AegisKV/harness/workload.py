"""Synthetic agentic Workflow: dispatcher A, parallel B-E, joiner F.

Every Agent owns a distinct random-token history of `input_len` tokens; B-E
also read A's output and F reads B-E's outputs. Decoding is greedy with
`ignore_eos`, so a rerun reproduces the same tokens.
"""

from __future__ import annotations

import random
from typing import Dict, List

STAGES: List[List[str]] = [["A"], ["B", "C", "D", "E"], ["F"]]
PREDECESSORS: Dict[str, List[str]] = {
    "A": [],
    "B": ["A"],
    "C": ["A"],
    "D": ["A"],
    "E": ["A"],
    "F": ["B", "C", "D", "E"],
}
SUCCESSORS: Dict[str, List[str]] = {
    "A": ["B", "C", "D", "E"],
    "B": ["F"],
    "C": ["F"],
    "D": ["F"],
    "E": ["F"],
    "F": [],
}
ROLE_INDEX = {role: i for i, role in enumerate("ABCDEF")}

# Avoid special and control tokens at both ends of MiniCPM's vocabulary.
_TOKEN_LO, _TOKEN_HI = 1000, 70000


def own_history(*, seed: int, workflow: int, role: str, length: int) -> List[int]:
    rng = random.Random(f"{seed}/{workflow}/{role}")
    return [rng.randrange(_TOKEN_LO, _TOKEN_HI) for _ in range(length)]


def build_prompt(
    *, own: List[int], role: str, outputs: Dict[str, List[int]]
) -> List[int]:
    prompt = list(own)
    for pred in PREDECESSORS[role]:
        prompt.extend(outputs[pred])
    return prompt
