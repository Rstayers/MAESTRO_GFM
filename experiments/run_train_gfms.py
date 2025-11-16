#!/usr/bin/env python3
"""
Train multiple stand-in GFMs on source domains (Planetoid datasets).

Models:
  - TAG-aware:
      * GFM_OFASmall       (TF-IDF + SVD + LogisticRegression)
      * GFM_TextMLPSmall   (TF-IDF + PCA + MLP)
  - Structure-first:
      * GFM_UniGraphSmall  (GCN)
      * GFM_SAGESmall      (GraphSAGE)
      * GFM_GATSmall       (GAT)

Artifacts written to:
  artifacts/gfm_ofa_small.pkl
  artifacts/gfm_text_mlp_small.pkl
  artifacts/gfm_unigraph_small.pt
  artifacts/gfm_sage_small.pt
  artifacts/gfm_gat_small.pt
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_planetoid(name: str):
    from torch_geometric.datasets import Planetoid

    ds = Planetoid(root="data/Planetoid", name=name)
    d = ds[0]
    return {"x": d.x.numpy(), "edge_index": d.edge_index.numpy()}, d.y.numpy()


def synth_texts_from_features(X: np.ndarray, topk: int = 20) -> list[str]:
    """
    Synthetic node texts: for each node, take top-k feature indices as "tokens".
    """
    idxs = np.argsort(-X, axis=1)[:, :topk]
    return [" ".join([f"f{j}" for j in row]) for row in idxs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()

    ensure_dir("artifacts")

    device = args.device

    # -----------------------------------------------------------
    # 1) Aggregate Planetoid sources for TAG models
    # -----------------------------------------------------------
    sources = ["Cora", "CiteSeer", "PubMed"]
    X_all, y_all, texts_all = [], [], []

    for name in sources:
        G, y = load_planetoid(name)
        X = G["x"]
        X_all.append(X)
        y_all.append(y)
        texts_all.extend(synth_texts_from_features(X, topk=25))

    y_tag = np.concatenate(y_all, axis=0)

    # TAG-aware GFMs
    from maestro.tools.gfm_ofa_small import GFM_OFASmall
    from maestro.tools.gfm_text_mlp_small import GFM_TextMLPSmall

    print("▶ Training GFM_OFASmall (TAG)...")
    ofa = GFM_OFASmall()
    ofa.fit(texts_all, y_tag)
    with open("artifacts/gfm_ofa_small.pkl", "wb") as f:
        pickle.dump(ofa, f)
    print("✔ Saved artifacts/gfm_ofa_small.pkl")

    print("▶ Training GFM_TextMLPSmall (TAG)...")
    text_mlp = GFM_TextMLPSmall()
    text_mlp.fit(texts_all, y_tag)
    with open("artifacts/gfm_text_mlp_small.pkl", "wb") as f:
        pickle.dump(text_mlp, f)
    print("✔ Saved artifacts/gfm_text_mlp_small.pkl")

    # -----------------------------------------------------------
    # 2) Train structure-first GFMs on a representative source
    #    (Cora, for speed & simplicity)
    # -----------------------------------------------------------
    G_cora, y_cora = load_planetoid("Cora")

    from maestro.tools.gfm_unigraph_small import GFM_UniGraphSmall, UniGraphTiny
    from maestro.tools.gfm_sage_small import GFM_SAGESmall, SageTiny
    from maestro.tools.gfm_gat_small import GFM_GATSmall, GATTiny

    print("▶ Training GFM_UniGraphSmall (GCN)...")
    uni = GFM_UniGraphSmall(device=device)
    uni.fit(G_cora, y_cora, epochs=args.epochs)
    torch.save(
        {
            "state_dict": uni.model.state_dict(),
            "meta": {"in": uni.in_dim, "out": uni.num_classes, "hidden": uni.hidden},
        },
        "artifacts/gfm_unigraph_small.pt",
    )
    print("✔ Saved artifacts/gfm_unigraph_small.pt")

    print("▶ Training GFM_SAGESmall (GraphSAGE)...")
    sage = GFM_SAGESmall(device=device)
    sage.fit(G_cora, y_cora, epochs=args.epochs)
    torch.save(
        {
            "state_dict": sage.model.state_dict(),
            "meta": {"in": sage.in_dim, "out": sage.num_classes, "hidden": sage.hidden},
        },
        "artifacts/gfm_sage_small.pt",
    )
    print("✔ Saved artifacts/gfm_sage_small.pt")

    print("▶ Training GFM_GATSmall (GAT)...")
    gat = GFM_GATSmall(device=device)
    gat.fit(G_cora, y_cora, epochs=args.epochs)
    torch.save(
        {
            "state_dict": gat.model.state_dict(),
            "meta": {"in": gat.in_dim, "out": gat.num_classes, "hidden": gat.hidden, "heads": gat.heads},
        },
        "artifacts/gfm_gat_small.pt",
    )
    print("✔ Saved artifacts/gfm_gat_small.pt")

    print("✅ All GFMs trained and saved into artifacts/")


if __name__ == "__main__":
    main()
