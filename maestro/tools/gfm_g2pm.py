# maestro/tools/gfm_g2pm.py
import os, numpy as np, torch
from typing import Dict
from sklearn.cluster import KMeans
from ..types import ToolChoice, ToolOutput

# --- G2PM imports (repo must be on PYTHONPATH) ---
from G2PM.G2PM.model.model import PretrainModel  # owns encoder/decoder; returns pred, instance_emb
from G2PM.G2PM.task.node import preprocess_node   # builds pattern_set from a PyG Data / dict
from G2PM.G2PM.data.pyg_data_loader import load_data  # only to mirror shapes when needed

class GFM_G2PM:
    """
    Minimal wrapper to use G2PM as a GFM inside MAESTRO.

    Modes:
      - encode(): produce node embeddings for MAESTRO's zero-label pipeline
      - predict(): unsupervised soft clustering over embeddings to provide probs/entropy
                   (works with label-free utility; not a supervised classifier)
    """
    def __init__(self, ckpt_path: str, params: dict, name: str = "gfm_g2pm", device: str = "cuda"):
        self.name = name
        self.family = "GNN"  # sequencing-based GFM; set a distinct family from TAG tools
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.params = params.copy()
        self.params['device'] = self.device

        # Build model & load weights
        self.model = PretrainModel(params=self.params).to(self.device)
        state = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        # cache for speed
        self._pattern_cache = None  # (patterns, nids, eids, node_pe, feat, e_feat)

    # ---- helpers ----
    def _to_pyg(self, G: Dict):
        # Accept MAESTRO G dict: {'x': np.ndarray [N,D], 'edge_index': np.ndarray [2,E]}
        import torch_geometric.data as tg
        x = torch.tensor(G["x"], dtype=torch.float)
        ei = torch.tensor(G["edge_index"], dtype=torch.long)
        data = tg.Data(x=x, edge_index=ei)
        return data

    def _build_patterns(self, data):
        # Mirror the G2PM preprocessing that pretrain.py does: preprocess -> pattern_set
        # The preprocess fills pattern_set (patterns, nids, eids) given data & params
        pattern_set = preprocess_node(data, self.params)
        return pattern_set

    # ---- public API expected by MAESTRO ----
    def encode(self, G: Dict) -> np.ndarray:
        data = self._to_pyg(G)
        with torch.no_grad():
            pattern_set = self._build_patterns(data)
            self.params['pattern_set'] = pattern_set  # same contract used in training
            # Forward once to grab instance embeddings (node-level)
            pred, instance_emb, _, _ = self.model(self.params, mode="eval")  # model forward returns (pred, emb, ...)
            Z = instance_emb.detach().cpu().numpy()
        return Z

    def predict(self, G: Dict) -> ToolOutput:
        # Unsupervised: cluster embeddings → distances → softmax = probs
        Z = self.encode(G)  # [N, d]
        K = max(2, int(np.clip(np.sqrt(Z.shape[0] / 50), 2, 50)))  # a safe default if #classes unknown
        km = KMeans(n_clusters=K, n_init=10, random_state=42).fit(Z)
        centers = km.cluster_centers_  # [K, d]
        # Convert to logits by negative squared distance to cluster centers
        d2 = ((Z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)  # [N, K]
        logits = -d2
        # stable softmax
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits); probs /= probs.sum(axis=1, keepdims=True)

        return ToolOutput(
            tool=ToolChoice(self.name, cfg={"mode": "unsup-kmeans", "K": K}),
            logits=logits,
            probs=probs,
            stats={}
        )
