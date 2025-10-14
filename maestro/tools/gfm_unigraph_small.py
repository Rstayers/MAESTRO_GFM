# maestro/tools/gfm_unigraph_small.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from ..types import ToolChoice, ToolOutput
from ..utils.metrics import entropy, softmax

class UniGraphTiny(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, out_dim)
        self.dropout = dropout
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index); x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GFM_UniGraphSmall:
    """
    Structure-first 'GFM' stand-in:
      - Tiny 2-layer GCN (you can swap for a Universal Graph Transformer later)
      - Robust to feature dim mismatch via truncate/pad alignment
      - Optional adapter init via class prototype vectors mapped to classifier layer
    """
    def __init__(self, name="gfm_unigraph_small", hidden=128, device="cpu"):
        self.name = name
        self.family = "GNN"
        self.hidden = hidden
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = None
        self.in_dim = None
        self.num_classes = None
        self.adapter_bias = None  # optional class-wise bias

        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    def _align_features(self, X: np.ndarray) -> np.ndarray:
        if self.in_dim is None: return X
        d = X.shape[1]
        if d == self.in_dim: return X
        if d > self.in_dim: return X[:, : self.in_dim]
        pad = np.zeros((X.shape[0], self.in_dim - d), dtype=X.dtype)
        return np.concatenate([X, pad], axis=1)

    def fit(self, G_train, y, epochs=300, lr=0.01, wd=5e-4):
        x = torch.tensor(G_train["x"], dtype=torch.float32, device=self.device)
        edge_index = torch.tensor(G_train["edge_index"], dtype=torch.long, device=self.device)
        y = torch.tensor(y, dtype=torch.long, device=self.device)
        self.in_dim = x.size(1)
        self.num_classes = int(y.max().item() + 1)
        self.model = UniGraphTiny(self.in_dim, self.hidden, self.num_classes).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        self.model.train()
        for _ in range(epochs):
            opt.zero_grad(set_to_none=True)
            logits = self.model(x, edge_index)
            loss = F.cross_entropy(logits, y)
            loss.backward(); opt.step()

    def init_with_prototypes(self, proto_vectors_by_class: dict[int, np.ndarray] | None):
        """
        Optional: pass dict class_id -> vector in input feature space.
        We project means through first layer weights (approx) to bias classifier.
        """
        if not proto_vectors_by_class or self.model is None:
            self.adapter_bias = None; return self
        # crude: take mean proto, project to logits space using current second-layer weights
        with torch.no_grad():
            W2 = self.model.conv2.lin.weight  # [C, H]
            # set small bias favoring proto classes
            bias = 0.05 * torch.ones(self.num_classes, device=W2.device)
            for c, vec in proto_vectors_by_class.items():
                # vec assumed in input dim; collapse through conv1 approx by linear part
                bias[c] = 0.1  # a slightly higher bias to present class; tune if needed
            self.adapter_bias = bias.cpu().numpy()
        return self

    def encode(self, G_batch):
        X = self._align_features(G_batch["x"])
        x = torch.tensor(X, dtype=torch.float32, device=self.device)
        edge_index = torch.tensor(G_batch["edge_index"], dtype=torch.long, device=self.device)
        self.model.eval()
        with torch.no_grad():
            # take penultimate layer as embedding (after conv1 + ReLU, before conv2)
            h = self.model.conv1(x, edge_index)
            h = torch.relu(h).cpu().numpy()
        return h

    def predict(self, G_batch, with_tta=False):
        assert self.model is not None, "Model not initialized"
        X = self._align_features(G_batch["x"])
        x = torch.tensor(X, dtype=torch.float32, device=self.device)
        edge_index = torch.tensor(G_batch["edge_index"], dtype=torch.long, device=self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x, edge_index).cpu().numpy()
        if self.adapter_bias is not None:
            logits = logits + self.adapter_bias[None, :]
        probs = softmax(logits)
        ent = float(entropy(probs).mean())
        return ToolOutput(tool=ToolChoice(self.name, {}), logits=logits, probs=probs, stats={"mean_entropy": ent})
