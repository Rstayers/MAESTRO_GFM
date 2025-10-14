#!/usr/bin/env python3
"""
Train two GFMs on source domains: Cora, Citeseer, PubMed.
- TAG-aware: GFM_OFASmall (uses node texts; we synthesize if needed)
- Structure-first: GFM_UniGraphSmall (tiny GCN)

Artifacts:
  artifacts/gfm_ofa_small.pkl
  artifacts/gfm_unigraph_small.pt (torch)  + meta
"""
import argparse, os, pickle, numpy as np, torch
from pathlib import Path

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

# --- data loaders (PyG) ---
def load_planetoid(name):
    from torch_geometric.datasets import Planetoid
    ds = Planetoid(root=f"data/Planetoid", name=name)
    d = ds[0]
    return {"x": d.x.numpy(), "edge_index": d.edge_index.numpy()}, d.y.numpy()

def synth_texts_from_features(X: np.ndarray, topk=20):
    idxs = np.argsort(-X, axis=1)[:, :topk]
    return [" ".join([f"f{j}" for j in row]) for row in idxs]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()
    ensure_dir("artifacts")

    # aggregate sources by simple concatenation for training each GFM
    sources = ["Cora", "CiteSeer", "PubMed"]
    X_all, EI_all, y_all, texts_all = [], [], [], []

    for name in sources:
        G, y = load_planetoid(name)
        X_all.append(G["x"]); EI_all.append(G["edge_index"]); y_all.append(y)
        # we don't have raw text in these; synth tokens to train TAG-aware model
        texts_all += synth_texts_from_features(G["x"])
    # stack features for TAG; structure model will use Cora only to keep it simple/fast
    y_tag = np.concatenate(y_all, axis=0)

    # --- TAG-aware GFM ---
    from maestro.tools.gfm_ofa_small import GFM_OFASmall
    ofa = GFM_OFASmall()
    ofa.fit(texts_all, y_tag)
    with open("artifacts/gfm_ofa_small.pkl", "wb") as f: pickle.dump(ofa, f)
    print("✔ saved artifacts/gfm_ofa_small.pkl")

    # --- UniGraph-small on Cora only (fast) ---
    from maestro.tools.gfm_unigraph_small import GFM_UniGraphSmall
    G_cora, y_cora = load_planetoid("Cora")
    uni = GFM_UniGraphSmall(device=args.device)
    uni.fit({"x": G_cora["x"], "edge_index": G_cora["edge_index"]}, y_cora, epochs=args.epochs)

    ckpt = {"state_dict": uni.model.state_dict(), "meta": {"in": uni.in_dim, "out": uni.num_classes, "hidden": uni.hidden}}
    torch.save(ckpt, "artifacts/gfm_unigraph_small.pt")
    print("✔ saved artifacts/gfm_unigraph_small.pt")

if __name__ == "__main__":
    main()
