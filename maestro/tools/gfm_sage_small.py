# maestro/tools/gfm_sage_small.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from ..types import ToolChoice, ToolOutput
from ..utils.metrics import softmax, entropy


class SageTiny(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GFM_SAGESmall:
    """
    Structure-first 'GFM' stand-in using GraphSAGE:
      - 2-layer SAGEConv
      - Same interface as GFM_UniGraphSmall
    """
    def __init__(self, name: str = "gfm_sage_small", hidden: int = 128, device: str = "cpu"):
        self.name = name
        self.family = "GNN"
        self.hidden = hidden
        self.device = torch.device(
            device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model: SageTiny | None = None
        self.in_dim: int | None = None
        self.num_classes: int | None = None
        self.adapter_bias: np.ndarray | None = None

        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    def _align_features(self, X: np.ndarray) -> np.ndarray:
        """
        Match incoming feature dimension to the model's expected in_dim.
        Truncate or zero-pad as needed.
        """
        if self.in_dim is None:
            return X
        d = X.shape[1]
        if d == self.in_dim:
            return X
        if d > self.in_dim:
            return X[:, : self.in_dim]
        pad = np.zeros((X.shape[0], self.in_dim - d), dtype=X.dtype)
        return np.concatenate([X, pad], axis=1)

    def fit(self, G_train, y, epochs: int = 300, lr: float = 0.01, wd: float = 5e-4):
        x = torch.tensor(G_train["x"], dtype=torch.float32, device=self.device)
        edge_index = torch.tensor(G_train["edge_index"], dtype=torch.long, device=self.device)
        y = torch.tensor(y, dtype=torch.long, device=self.device)

        self.in_dim = x.size(1)
        self.num_classes = int(y.max().item() + 1)

        self.model = SageTiny(self.in_dim, self.hidden, self.num_classes).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)

        self.model.train()
        for _ in range(epochs):
            opt.zero_grad(set_to_none=True)
            logits = self.model(x, edge_index)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()

    def init_with_prototypes(self, *_args, **_kwargs):
        """
        Placeholder for proto-based initialization to mirror GFM_UniGraphSmall.
        Not used in current experiments.
        """
        self.adapter_bias = None
        return self

    def encode(self, G_batch) -> np.ndarray:
        """
        Return node embeddings from the penultimate layer (after conv1 + ReLU).
        """
        assert self.model is not None, "Model not initialized"
        X = self._align_features(G_batch["x"])
        x = torch.tensor(X, dtype=torch.float32, device=self.device)
        edge_index = torch.tensor(G_batch["edge_index"], dtype=torch.long, device=self.device)
        self.model.eval()
        with torch.no_grad():
            h = self.model.conv1(x, edge_index)
            h = F.relu(h).cpu().numpy()
        return h

    def predict(self, G_batch, with_tta: bool = False) -> ToolOutput:
        assert self.model is not None, "Model not initialized"

        X = self._align_features(G_batch["x"])
        x = torch.tensor(X, dtype=torch.float32, device=self.device)
        edge_index = torch.tensor(G_batch["edge_index"], dtype=torch.long, device=self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(x, edge_index).cpu().numpy()

        if self.adapter_bias is not None:
            logits = logits + self.adapter_bias

        probs = softmax(logits)
        mean_ent = float(entropy(probs).mean())

        return ToolOutput(
            tool=ToolChoice(self.name, {"family": self.family}),
            logits=logits,
            probs=probs,
            stats={"mean_entropy": mean_ent},
        )
