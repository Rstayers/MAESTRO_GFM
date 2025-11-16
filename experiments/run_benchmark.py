#!/usr/bin/env python3
"""
End-to-end benchmark:

For each dataset:
  - Build synthetic TAG texts from features (if needed)
  - Evaluate each GFM alone:
      * Use tool.encode() -> embeddings
      * KMeans clustering -> Hungarian alignment -> accuracy + macro-F1
  - Evaluate Maestro:
      * Use capability profiles + UCB planner to select a committee
      * Concatenate embeddings from chosen tools
      * Same KMeans + Hungarian evaluation

Outputs:
  - artifacts/benchmark_results.json   (for replication)
  - Pretty-printed summary to stdout
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

import torch

from maestro.orchestrator.infer import Orchestrator
from maestro.utils.metrics import macro_f1


def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


# ---------- dataset loaders ----------


def load_planetoid(name: str):
    from torch_geometric.datasets import Planetoid

    ds = Planetoid(root="data/Planetoid", name=name)
    d = ds[0]
    return {"x": d.x.numpy(), "edge_index": d.edge_index.numpy()}, d.y.numpy()


def load_wikics():
    from torch_geometric.datasets import WikiCS

    ds = WikiCS(root="data/WikiCS")
    d = ds[0]
    return {"x": d.x.numpy(), "edge_index": d.edge_index.numpy()}, d.y.numpy()


def load_amazon_computers():
    from torch_geometric.datasets import Amazon

    ds = Amazon(root="data/Amazon", name="Computers")
    d = ds[0]
    return {"x": d.x.numpy(), "edge_index": d.edge_index.numpy()}, d.y.numpy()


def synth_texts_from_features(X: np.ndarray, topk: int = 25):
    idxs = np.argsort(-X, axis=1)[:, :topk]
    return [" ".join([f"f{j}" for j in row]) for row in idxs]


# ---------- evaluation helpers ----------


def hungarian_map(pred_clusters: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Map unsupervised clusters to ground truth labels via Hungarian matching.
    """
    K = int(y_true.max() + 1)
    C = np.zeros((K, K), dtype=int)
    for p, t in zip(pred_clusters, y_true):
        C[p, t] += 1
    r, c = linear_sum_assignment(C.max() - C)
    mapping = {r_i: c_i for r_i, c_i in zip(r, c)}
    y_pred = np.array([mapping[p] for p in pred_clusters], dtype=int)
    return y_pred


def evaluate_embeddings_kmeans(Z: np.ndarray, y_true: np.ndarray, K: int | None = None, seed: int = 0):
    if K is None:
        K = int(y_true.max() + 1)
    km = KMeans(n_clusters=K, random_state=seed, n_init=10)
    clusters = km.fit_predict(Z)
    y_pred = hungarian_map(clusters, y_true)
    acc = float((y_pred == y_true).mean())
    f1 = float(macro_f1(y_true, y_pred))
    return acc, f1


# ---------- tool loader (same GFMs as training/capabilities) ----------


def rebuild_tools(device: str = "cpu"):
    import pickle
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


# ---------- main ----------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--datasets",
        nargs="*",
        default=["Cora", "CiteSeer", "PubMed", "WikiCS", "Amazon-Computers"],
        help="Datasets to evaluate",
    )
    ap.add_argument("--device", default="auto")
    ap.add_argument("--committee_k", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    np.random.seed(args.seed)

    # Dataset registry
    dataset_loaders = {
        "Cora": lambda: load_planetoid("Cora"),
        "CiteSeer": lambda: load_planetoid("CiteSeer"),
        "PubMed": lambda: load_planetoid("PubMed"),
        "WikiCS": load_wikics,
        "Amazon-Computers": load_amazon_computers,
    }

    # Load tools + capabilities
    tools = rebuild_tools(device=args.device)
    tool_names = [t.name for t in tools]

    import pickle

    with open("artifacts/capabilities.pkl", "rb") as f:
        cap = pickle.load(f)

    orch = Orchestrator(
        tools=tools,
        cap_profiles=cap,
        committee_k=args.committee_k,
        diversity=True,
        debug=args.debug,
    )

    results = {
        "config": {
            "datasets": args.datasets,
            "committee_k": args.committee_k,
            "seed": args.seed,
        },
        "per_dataset": {},
    }

    for ds_name in args.datasets:
        if ds_name not in dataset_loaders:
            print(f"[warn] Unknown dataset '{ds_name}', skipping.")
            continue

        print(f"\n===== Dataset: {ds_name} =====")
        G_raw, y = dataset_loaders[ds_name]()
        K = int(y.max() + 1)

        texts = synth_texts_from_features(G_raw["x"], topk=25)
        G = {"x": G_raw["x"], "edge_index": G_raw["edge_index"], "text": texts}

        # ---------- Evaluate each tool alone ----------
        tool_scores = {}
        for t in tools:
            if not hasattr(t, "encode"):
                print(f"[warn] Tool {t.name} has no encode(); skipping in benchmark.")
                continue

            if getattr(t, "family", None) == "TAG":
                Z = t.encode(G["text"])
            else:
                Z = t.encode(G)

            acc, f1 = evaluate_embeddings_kmeans(Z, y, K=K, seed=args.seed)
            tool_scores[t.name] = {"acc": acc, "macro_f1": f1}
            print(f"  [tool] {t.name:18s} acc={acc:.4f}  macroF1={f1:.4f}")

        # ---------- Maestro committee ----------
        final, meta = orch.infer(G)
        chosen_names = meta.get("chosen", [])
        print(f"  [maestro] chosen tools: {chosen_names}")

        Z_list = []
        for t in tools:
            if t.name in chosen_names and hasattr(t, "encode"):
                if getattr(t, "family", None) == "TAG":
                    Z_list.append(t.encode(G["text"]))
                else:
                    Z_list.append(t.encode(G))

        if not Z_list:
            print("  [maestro] No tools with embeddings selected! Skipping Maestro eval.")
            maestro_result = {"acc": None, "macro_f1": None, "chosen_tools": chosen_names}
        else:
            Z_cat = np.concatenate(Z_list, axis=1) if len(Z_list) > 1 else Z_list[0]
            acc_m, f1_m = evaluate_embeddings_kmeans(Z_cat, y, K=K, seed=args.seed)
            print(f"  [maestro] acc={acc_m:.4f}  macroF1={f1_m:.4f}")
            maestro_result = {
                "acc": acc_m,
                "macro_f1": f1_m,
                "chosen_tools": chosen_names,
                "utility": meta.get("utility", None),
            }

        results["per_dataset"][ds_name] = {
            "tools": tool_scores,
            "maestro": maestro_result,
        }

    # Save JSON results
    ensure_dir("artifacts")
    out_path = Path("artifacts/benchmark_results.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Benchmark results saved to {out_path}")


if __name__ == "__main__":
    main()
