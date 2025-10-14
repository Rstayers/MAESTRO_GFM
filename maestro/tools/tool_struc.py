import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from ..types import ToolChoice, ToolOutput
from ..utils.metrics import softmax, entropy

class GCN(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, out_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class ToolSTRUC:
    def __init__(self, name='tool_struc', hidden=64, device='auto'):
        self.name = name
        self.hidden = hidden
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = None
        self.num_classes = None
        self.in_dim = None  # <-- store expected input dim

    def _align_features(self, X: np.ndarray) -> np.ndarray:
        """Align target features to the training in_dim by truncate/pad."""
        if self.in_dim is None:
            # Fallback: infer from current X if model isn't set yet
            return X
        d = X.shape[1]
        if d == self.in_dim:
            return X
        if d > self.in_dim:
            # truncate extra dims
            return X[:, : self.in_dim]
        else:
            # pad missing dims with zeros
            pad = np.zeros((X.shape[0], self.in_dim - d), dtype=X.dtype)
            return np.concatenate([X, pad], axis=1)

    def fit(self, G_train, y, epochs: int = 200, lr: float = 0.01, weight_decay: float = 5e-4):
        x = torch.tensor(G_train['x'], dtype=torch.float32, device=self.device)
        edge_index = torch.tensor(G_train['edge_index'], dtype=torch.long, device=self.device)
        y = torch.tensor(y, dtype=torch.long, device=self.device)
        in_dim = x.size(1)
        out_dim = int(y.max().item() + 1)
        self.in_dim = in_dim  # <-- remember train dim
        self.num_classes = out_dim
        self.model = GCN(in_dim, self.hidden, out_dim).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.model.train()
        for _ in range(epochs):
            opt.zero_grad()
            logits = self.model(x, edge_index)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()

    def predict(self, G_batch, with_tta=False):
        assert self.model is not None, "ToolSTRUC model is not initialized"
        self.model.eval()
        # --- NEW: align features to training dim
        X = self._align_features(G_batch['x'])
        x = torch.tensor(X, dtype=torch.float32, device=self.device)
        edge_index = torch.tensor(G_batch['edge_index'], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(x, edge_index).cpu().numpy()
        probs = softmax(logits)
        ent = float(entropy(probs).mean())
        return ToolOutput(tool=ToolChoice(self.name, {}), logits=logits, probs=probs, stats={'mean_entropy': ent})
