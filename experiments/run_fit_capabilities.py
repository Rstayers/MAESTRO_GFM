#!/usr/bin/env python3
"""
Create many source episodes on Cora/Citeseer/PubMed and fit capability profiles
for multiple GFMs:

  - gfm_ofa_small       (TAG)
  - gfm_text_mlp_small  (TAG)
  - gfm_unigraph_small  (GCN)
  - gfm_sage_small      (GraphSAGE)
  - gfm_gat_small       (GAT)

Artifacts:
  - artifacts/episodes_gfms.jsonl   (for inspection)
  - artifacts/capabilities.pkl      (CapabilityProfiles)
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from maestro.profiler.domain_profiler import compute_fingerprint
from maestro.memory.capability_profiles import CapabilityProfiles
from maestro.utils.metrics import label_free_utility


def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def load_planetoid(name: str):
    from torch_geometric.datasets import Planetoid

    ds = Planetoid(root="data/Planetoid", name=name)
    d = ds[0]
    return {"x": d.x.numpy(), "edge_index": d.edge_index.numpy()}, d.y.numpy()


def synth_texts_from_features(X: np.ndarray, topk: int = 20):
    idxs = np.argsort(-X, axis=1)[:, :topk]
    return [" ".join([f"f{j}" for j in row]) for row in idxs]


def edge_dropout(EI: np.ndarray, p: float, rng: np.random.Generator):
    if p <= 0:
        return EI
    m = EI.shape[1]
    keep = rng.random(m) > p
    kept = EI[:, keep]
    return kept if kept.size else EI


def feature_mask(X: np.ndarray, p: float, rng: np.random.Generator):
    if p <= 0:
        return X
    Xc = X.copy()
    N, D = Xc.shape
    nodes = np.where(rng.random(N) < p)[0]
    for i in nodes:
        k = max(1, int(0.02 * D))
        cols = rng.choice(D, size=k, replace=False)
        Xc[i, cols] = 0.0
    return Xc


def rebuild_tools(device: str = "cpu"):
    """
    Load all GFMs from artifacts/ and return as a list.
    """
    from maestro.tools.gfm_ofa_small import GFM_OFASmall  # noqa: F401
    from maestro.tools.gfm_text_mlp_small import GFM_TextMLPSmall  # noqa: F401
    from maestro.tools.gfm_unigraph_small import GFM_UniGraphSmall, UniGraphTiny
    from maestro.tools.gfm_sage_small import GFM_SAGESmall, SageTiny
    from maestro.tools.gfm_gat_small import GFM_GATSmall, GATTiny

    with open("artifacts/gfm_ofa_small.pkl", "rb") as f:
        ofa = pickle.load(f)
    with open("artifacts/gfm_text_mlp_small.pkl", "rb") as f:
        text_mlp = pickle.load(f)

    device_torch = torch.device(
        device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # UniGraphSmall
    ckpt_uni = torch.load("artifacts/gfm_unigraph_small.pt", map_location=device_torch)
    uni = GFM_UniGraphSmall(device=device)
    uni.in_dim = ckpt_uni["meta"]["in"]
    uni.num_classes = ckpt_uni["meta"]["out"]
    uni.hidden = ckpt_uni["meta"]["hidden"]
    uni.model = UniGraphTiny(
        ckpt_uni["meta"]["in"], ckpt_uni["meta"]["hidden"], ckpt_uni["meta"]["out"]
    ).to(uni.device)
    uni.model.load_state_dict(ckpt_uni["state_dict"])

    # SAGESmall
    ckpt_sage = torch.load("artifacts/gfm_sage_small.pt", map_location=device_torch)
    sage = GFM_SAGESmall(device=device)
    sage.in_dim = ckpt_sage["meta"]["in"]
    sage.num_classes = ckpt_sage["meta"]["out"]
    sage.hidden = ckpt_sage["meta"]["hidden"]
    sage.model = SageTiny(
        ckpt_sage["meta"]["in"], ckpt_sage["meta"]["hidden"], ckpt_sage["meta"]["out"]
    ).to(sage.device)
    sage.model.load_state_dict(ckpt_sage["state_dict"])

    # GATSmall
    ckpt_gat = torch.load("artifacts/gfm_gat_small.pt", map_location=device_torch)
    gat = GFM_GATSmall(device=device)
    gat.in_dim = ckpt_gat["meta"]["in"]
    gat.num_classes = ckpt_gat["meta"]["out"]
    gat.hidden = ckpt_gat["meta"]["hidden"]
    gat.heads = ckpt_gat["meta"].get("heads", gat.heads)
    gat.model = GATTiny(
        ckpt_gat["meta"]["in"],
        ckpt_gat["meta"]["hidden"],
        ckpt_gat["meta"]["out"],
        heads=gat.heads,
    ).to(gat.device)
    gat.model.load_state_dict(ckpt_gat["state_dict"])

    tools = [ofa, text_mlp, uni, sage, gat]
    return tools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=36)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--edge_drop", type=float, default=0.05)
    ap.add_argument("--feat_mask", type=float, default=0.10)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ensure_dir("artifacts")

    tools = rebuild_tools(device=args.device)
    tool_names = [t.name for t in tools]

    sources = ["Cora", "CiteSeer", "PubMed"]
    Xs, Ys, EIs = {}, {}, {}
    for s in sources:
        G, y = load_planetoid(s)
        Xs[s], Ys[s], EIs[s] = G["x"], y, G["edge_index"]

    Xcap = []
    U = {name: [] for name in tool_names}

    ep_path = Path("artifacts/episodes_gfms.jsonl")
    with ep_path.open("w", encoding="utf-8") as fout:
        for s in sources:
            X0, EI0 = Xs[s], EIs[s]
            for seed in range(args.seeds):
                rng = np.random.default_rng(100 + 1000 * seed)
                for e in range(args.episodes // args.seeds):
                    X = feature_mask(X0, args.feat_mask, rng)
                    EI = edge_dropout(EI0, args.edge_drop, rng)
                    phi = compute_fingerprint({"x": X, "edge_index": EI, "text": None})

                    # TAG texts from features (stand-in)
                    texts = synth_texts_from_features(X, topk=25)
                    G_tag = {"x": X, "edge_index": EI, "text": texts}
                    G_plain = {"x": X, "edge_index": EI, "text": None}

                    u_dict = {}
                    for tool in tools:
                        if getattr(tool, "family", None) == "TAG":
                            out = tool.predict(G_tag)
                        else:
                            out = tool.predict(G_plain)
                        u_val = label_free_utility(out.probs)
                        u_dict[tool.name] = float(u_val)
                        U[tool.name].append(u_val)

                    fout.write(
                        json.dumps(
                            {
                                "src": s,
                                "phi": phi.vector.tolist(),
                                "u": u_dict,
                            }
                        )
                        + "\n"
                    )
                    Xcap.append(phi.vector)

    Xcap = np.vstack(Xcap)
    cap = CapabilityProfiles(alpha=1.0)

    for name in tool_names:
        U_arr = np.array(U[name])
        cap.fit(name, Xcap, U_arr)

    with open("artifacts/capabilities.pkl", "wb") as f:
        pickle.dump(cap, f)

    print("✔ saved artifacts/capabilities.pkl")
    print(f"✔ episodes written to {ep_path} (count={Xcap.shape[0]})")


if __name__ == "__main__":
    main()
