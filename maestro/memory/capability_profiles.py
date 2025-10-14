import numpy as np
from sklearn.linear_model import Ridge


class CapabilityProfiles:
    def __init__(self, alpha: float = 1.0):
        self.models = {}
        self.alpha = alpha


    def fit(self, tool_name: str, X: np.ndarray, y: np.ndarray):
        m = Ridge(alpha=self.alpha)
        m.fit(X, y)
        self.models[tool_name] = m


    def predict(self, tool_name: str, phi_vec: np.ndarray):
        m = self.models.get(tool_name, None)
        if m is None:
            return 0.0, 0.1
        mu = float(m.predict(phi_vec.reshape(1, -1))[0])
        return mu, 0.1 # MVP: fixed small uncertainty