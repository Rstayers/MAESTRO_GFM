import os
import numpy as np
import torch

from ..types import ToolChoice, ToolOutput
from ..utils.metrics import entropy, softmax



class GFM_GOFA:
    """
    GOFA-backed tool for MAESTRO.

    encode(G_batch)  -> [N, d] node embeddings
    predict(G_batch) -> ToolOutput with pseudo logits/probs for entropy-based routing
    """

    def __init__(self):
        pass
    def encode(self, G_batch):
        """
        Try GOFA's graph-utils path first; on failure, fall back to text-only embeddings.
        """
        pass

    def predict(self, G_batch, with_tta: bool = False):
       pass
