from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import numpy as np


@dataclass
class DomainFingerprint:
    struct_stats: Dict[str, float]
    schema_stats: Dict[str, float]
    task_meta: Dict[str, float]
    vector: np.ndarray # 1D


@dataclass
class ToolChoice:
    name: str
    cfg: Dict[str, Any]


@dataclass
class ToolOutput:
    tool: ToolChoice
    logits: np.ndarray # [N, C]
    probs: np.ndarray # [N, C]
    stats: Dict[str, float]