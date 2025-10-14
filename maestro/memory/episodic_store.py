from typing import List
import numpy as np
from ..types import DomainFingerprint


class EpisodicStore:
    def __init__(self):
        self._vecs = [] # list[np.ndarray]
        self._meta = [] # list[dict]


    def add(self, phi: DomainFingerprint, meta: dict) -> int:
        self._vecs.append(phi.vector.astype(float))
        self._meta.append(meta)
        return len(self._meta) - 1


    def retrieve(self, phi: DomainFingerprint, k: int = 8) -> List[dict]:
        if not self._vecs: return []
        X = np.vstack(self._vecs)
        q = phi.vector.reshape(1, -1)
        sims = (X @ q.T).ravel() / ((np.linalg.norm(X, axis=1) + 1e-8) * (np.linalg.norm(q) + 1e-8))
        idx = sims.argsort()[::-1][:k]
        return [self._meta[i] for i in idx]