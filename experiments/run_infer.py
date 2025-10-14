#!/usr/bin/env python3
"""
Zero-label (unsupervised) inference on WikiCS:
- Routes between gfm_ofa_small (TAG) and gfm_unigraph_small (GNN)
- Uses embeddings + k-means on target, then Hungarian match for evaluation
"""
import argparse, pickle, numpy as np, torch
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from maestro.orchestrator.infer import Orchestrator
from maestro.profiler.domain_profiler import compute_fingerprint

def load_wikics():
    from torch_geometric.datasets import WikiCS
    ds = WikiCS(root="data/WikiCS")
    d = ds[0]
    return {"x": d.x.numpy(), "edge_index": d.edge_index.numpy()}, d.y.numpy()


# Make sure TAG model always gets pseudo-texts
def synth_texts_from_features(X, topk=25):
    import numpy as np
    idxs = np.argsort(-X, axis=1)[:, :topk]
    return [" ".join([f"f{j}" for j in row]) for row in idxs]

def rebuild_tools(device="cpu"):
    with open("artifacts/gfm_ofa_small.pkl", "rb") as f: ofa = pickle.load(f)
    from maestro.tools.gfm_unigraph_small import GFM_UniGraphSmall, UniGraphTiny
    ckpt = torch.load("artifacts/gfm_unigraph_small.pt", map_location=device)
    uni = GFM_UniGraphSmall(device=device)
    uni.in_dim = ckpt["meta"]["in"]; uni.num_classes = ckpt["meta"]["out"]
    uni.model = UniGraphTiny(ckpt["meta"]["in"], ckpt["meta"]["hidden"], ckpt["meta"]["out"]).to(uni.device)
    uni.model.load_state_dict(ckpt["state_dict"])
    return ofa, uni

def hungarian_map(pred_clusters, y_true):
    # build confusion matrix [K x K]; K = number of classes
    K = int(y_true.max() + 1)
    C = np.zeros((K, K), dtype=int)
    for p, t in zip(pred_clusters, y_true):
        C[p, t] += 1
    r, c = linear_sum_assignment(C.max() - C)
    mapping = {r_i: c_i for r_i, c_i in zip(r, c)}
    y_pred = np.array([mapping[p] for p in pred_clusters], dtype=int)
    return y_pred

def macro_f1(y_true, y_pred):
    classes = np.unique(y_true)
    f1s = []
    for c in classes:
        tp = np.sum((y_true==c) & (y_pred==c))
        fp = np.sum((y_true!=c) & (y_pred==c))
        fn = np.sum((y_true==c) & (y_pred!=c))
        prec = tp / (tp+fp+1e-8); rec = tp / (tp+fn+1e-8)
        f1s.append(2*prec*rec/(prec+rec+1e-8))
    return float(np.mean(f1s))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--committee_k", type=int, default=2)
    ap.add_argument("--k", type=int, default=None, help="num clusters; default = #classes in WikiCS")
    args = ap.parse_args()

    G, y = load_wikics()
    K = int(y.max() + 1) if args.k is None else args.k
    texts = synth_texts_from_features(G["x"], topk=25)
    G_tag = {"x": G["x"], "edge_index": G["edge_index"], "text": texts}

    # tools + capabilities
    ofa, uni = rebuild_tools(device=args.device)
    # --- after you load or rebuild the tools ---
    ofa.family = "TAG"
    uni.family = "GNN"


    texts = synth_texts_from_features(G["x"], topk=25)
    G_tag = {"x": G["x"], "edge_index": G["edge_index"], "text": texts}

    orig_predict = ofa.predict

    def predict_with_texts(_):
        return orig_predict(G_tag)

    ofa.predict = predict_with_texts
    with open("artifacts/capabilities.pkl", "rb") as f: cap = pickle.load(f)

    # Orchestrate (same planner); we’ll call tool.encode() after selection
    orch = Orchestrator(tools=[ofa, uni], cap_profiles=cap, committee_k=args.committee_k, diversity=True, debug=True)
    # Just to get the routing decision; final aggregation will be ignored here
    _, meta = orch.infer({"x": G["x"], "edge_index": G["edge_index"], "text": None})
    chosen = meta["chosen"]

    # build embeddings from chosen tools
    Z_list = []
    for name in chosen:
        if name == "gfm_ofa_small":
            Z_list.append(ofa.encode(texts))                       # [N, d1]
        elif name == "gfm_unigraph_small":
            Z_list.append(uni.encode(G))                           # [N, d2]
    Z = np.concatenate(Z_list, axis=1) if len(Z_list) > 1 else Z_list[0]

    # k-means on target
    km = KMeans(n_clusters=K, n_init=10, random_state=42)
    clusters = km.fit_predict(Z)

    # Evaluation by Hungarian alignment (still zero-label for training)
    y_hat = hungarian_map(clusters, y)
    f1 = macro_f1(y, y_hat)

    print("\n=== Zero-label (unsupervised) Result — WikiCS ===")
    print(f"Chosen tools: {chosen}")
    print(f"KMeans K={K}, Macro-F1 (Hungarian-mapped): {f1:.4f}")

if __name__ == "__main__":
    main()
