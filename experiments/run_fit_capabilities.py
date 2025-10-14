#!/usr/bin/env python3
"""
Create many source episodes on Cora/Citeseer/PubMed and fit capability profiles
for the two GFMs: gfm_ofa_small (TAG) and gfm_unigraph_small (GNN).
"""
import argparse, os, pickle, json, numpy as np, torch
from pathlib import Path
from maestro.profiler.domain_profiler import compute_fingerprint
from maestro.memory.capability_profiles import CapabilityProfiles
from maestro.utils.metrics import label_free_utility

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def load_planetoid(name):
    from torch_geometric.datasets import Planetoid
    ds = Planetoid(root="data/Planetoid", name=name)
    d = ds[0]
    return {"x": d.x.numpy(), "edge_index": d.edge_index.numpy()}, d.y.numpy()

def synth_texts_from_features(X: np.ndarray, topk=20):
    idxs = np.argsort(-X, axis=1)[:, :topk]
    return [" ".join([f"f{j}" for j in row]) for row in idxs]

def edge_dropout(EI, p, rng):
    if p <= 0: return EI
    m = EI.shape[1]
    keep = rng.random(m) > p
    kept = EI[:, keep]
    return kept if kept.size else EI

def feature_mask(X, p, rng):
    if p <= 0: return X
    Xc = X.copy()
    N, D = Xc.shape
    nodes = np.where(rng.random(N) < p)[0]
    for i in nodes:
        k = max(1, int(0.02 * D))
        cols = rng.choice(D, size=k, replace=False)
        Xc[i, cols] = 0.0
    return Xc

def rebuild_tools(device="cpu"):
    with open("artifacts/gfm_ofa_small.pkl", "rb") as f:
        ofa = pickle.load(f)

    from maestro.tools.gfm_unigraph_small import GFM_UniGraphSmall, UniGraphTiny
    ckpt = torch.load("artifacts/gfm_unigraph_small.pt", map_location=device)
    uni = GFM_UniGraphSmall(device=device)
    uni.in_dim = ckpt["meta"]["in"]; uni.num_classes = ckpt["meta"]["out"]
    uni.model = UniGraphTiny(ckpt["meta"]["in"], ckpt["meta"]["hidden"], ckpt["meta"]["out"]).to(uni.device)
    uni.model.load_state_dict(ckpt["state_dict"])
    return ofa, uni

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=36)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--edge_drop", type=float, default=0.05)
    ap.add_argument("--feat_mask", type=float, default=0.10)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    ensure_dir("artifacts")

    ofa, uni = rebuild_tools(device=args.device)
    tools = [ofa, uni]

    sources = ["Cora", "CiteSeer", "PubMed"]
    Xs, Ys, EIs = {}, {}, {}
    for s in sources:
        G, y = load_planetoid(s)
        Xs[s], Ys[s], EIs[s] = G["x"], y, G["edge_index"]

    Xcap, U_ofa, U_uni = [], [], []
    ep_path = Path("artifacts/episodes_gfms.jsonl")
    with ep_path.open("w", encoding="utf-8") as fout:
        for s in sources:
            X0, EI0 = Xs[s], EIs[s]
            for seed in range(args.seeds):
                rng = np.random.default_rng(100 + 1000*seed)
                for e in range(args.episodes // args.seeds):
                    X = feature_mask(X0, args.feat_mask, rng)
                    EI = edge_dropout(EI0, args.edge_drop, rng)
                    phi = compute_fingerprint({"x": X, "edge_index": EI, "text": None})
                    # TAG texts from features (stand-in)
                    texts = synth_texts_from_features(X, topk=25)
                    out_ofa = ofa.predict({"x": X, "edge_index": EI, "text": texts})
                    out_uni = uni.predict({"x": X, "edge_index": EI})
                    u_ofa = label_free_utility(out_ofa.probs)
                    u_uni = label_free_utility(out_uni.probs)
                    fout.write(json.dumps({
                        "src": s, "phi": phi.vector.tolist(),
                        "u": {"gfm_ofa_small": u_ofa, "gfm_unigraph_small": u_uni}
                    }) + "\n")
                    Xcap.append(phi.vector); U_ofa.append(u_ofa); U_uni.append(u_uni)

    Xcap = np.vstack(Xcap); U_ofa = np.array(U_ofa); U_uni = np.array(U_uni)
    cap = CapabilityProfiles(alpha=1.0)
    cap.fit("gfm_ofa_small", Xcap, U_ofa)
    cap.fit("gfm_unigraph_small", Xcap, U_uni)
    with open("artifacts/capabilities.pkl", "wb") as f:
        pickle.dump(cap, f)

    print("✔ saved artifacts/capabilities.pkl")
    print(f"✔ episodes written to {ep_path} (count={Xcap.shape[0]})")

if __name__ == "__main__":
    main()
