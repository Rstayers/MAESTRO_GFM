# MAESTRO-GFM
**Memory-Augmented Orchestrator for Graph Foundation Models (node-level)**

MAESTRO-GFM is an **agent** that profiles an unseen graph, retrieves prior experience, **chooses the best GFM(s)**, runs inference (zero-label or few-shot), aggregates outputs, and logs episodes to improve future routing.

---

## What it does 
- **Zero-label node classification:** encode nodes → **k-means** (k = #classes) → **Hungarian** matching to evaluate.  
- **Routing:** predicts which GFM will be most confident on the target graph using **capability profiles** learned from many source “episodes.”  
- **Committee (optional):** runs up to 2 diverse tools (e.g., **TAG** + **GNN**) with **aggregation** (confidence gate; winner-takes-lowest-entropy or entropy-weighted averaging).  
- **Memory:** logs each episode (fingerprint, chosen tools, utilities) to improve later decisions.

---

## Architecture (tiny map)
```
Target Graph G
   └─► Profiler → φ(G)                # structure+schema fingerprint
         └─► Capability Profiles û_t(φ)   # tool utility predictors
               └─► Planner (diversity guard, UCB/greedy) → pick tool(s)
                     └─► Executors (GFMs) → logits/embeddings
                           └─► Aggregator → predictions
                                 └─► Critic (entropy) + Episode Logger
```
**Modules**
- `profiler/domain_profiler.py` → φ(G) (degree/clustering/assort/homophily/spectral, TAG flag)  
- `memory/capability_profiles.py` → per-tool û_t(φ) (learned from episodes)  
- `planner/ucb_planner.py` → select 1–2 tools with family diversity  
- `orchestrator/infer.py` → glue + committee aggregation  
- `tools/` → **gfm_ofa_small (TAG)**, **gfm_unigraph_small (GNN)** (+ v0 toy tools)



---

## Run it (end-to-end, unseen zero-label on WikiCS)

### 1) Train two GFMs on sources (Cora/CiteSeer/PubMed)
```bash
python experiments/run_train_gfms.py --device cuda --epochs 300
```

### 2) Fit capability profiles from many source “episodes”
```bash
python experiments/run_fit_capabilities_gfms.py --episodes 48 --seeds 4 --edge_drop 0.05 --feat_mask 0.10
```

### 3) Route on **unseen WikiCS** (zero-label; embeddings → k-means → Hungarian)
```bash
# Single tool (headline)
python experiments/run_infer_wikics_unsup.py --committee_k 1 --device cuda

# Committee of 2 (concat embeddings before k-means)
python experiments/run_infer_wikics_unsup.py --committee_k 2 --device cuda
```

You’ll see:
```
[debug] planner chose: ['gfm_ofa_small']
=== Zero-label (unsupervised) Result — WikiCS ===
Chosen tools: ['gfm_ofa_small']
KMeans K=10, Macro-F1 (Hungarian-mapped): x.xxx
```

> **Note:** The diagnostic “supervised logits” path (`experiments/run_infer.py`) is for sanity only—label spaces differ across datasets, so F1 can be near-chance. Use the **unsupervised** script above for correct zero-label evaluation.
---
**Happy routing!**
