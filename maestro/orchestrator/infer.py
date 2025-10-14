# maestro/orchestrator/infer.py
import numpy as np
from typing import List, Dict, Tuple

from ..profiler.domain_profiler import compute_fingerprint
from ..memory.episodic_store import EpisodicStore
from ..memory.capability_profiles import CapabilityProfiles
from ..planner.ucb_planner import UCBPlanner
from ..utils.metrics import label_free_utility, softmax, entropy
from ..types import ToolOutput, ToolChoice


# ---------- Aggregation helpers ----------

def _aggregate_outputs(outputs: List[ToolOutput],
                       disagree_thr: float = 0.05,
                       conf_max_entropy: float = 1.10,
                       debug: bool = True) -> ToolOutput:
    """
    Aggregate multiple ToolOutput predictions robustly:
      1) Drop tools that are clearly over-uncertain (mean entropy >= conf_max_entropy).
      2) If remaining tools strongly disagree, pick the single most confident (lowest entropy).
      3) Else do entropy-weighted logit averaging.
    """
    if debug:
        for o in outputs:
            print(f"[debug] {o.tool.name} mean_entropy={o.stats.get('mean_entropy', float('nan')):.4f}")

    if len(outputs) == 1:
        return outputs[0]

    # Keep a raw copy for fallback
    outputs_raw = outputs[:]

    # 1) Confidence gate
    gated = [o for o in outputs if o.stats.get('mean_entropy', 9e9) < conf_max_entropy]
    outputs = gated if len(gated) > 0 else outputs_raw

    if len(outputs) == 1:
        # Only one survived gating → return it
        outputs[0].tool = ToolChoice(f"{outputs[0].tool.name}_winner", {})
        return outputs[0]

    # 2) Disagreement check (use std of mean entropies as a simple proxy)
    ents = np.array([entropy(o.probs).mean() for o in outputs], dtype=float)
    disagree = float(ents.std())

    if debug:
        print(f"[debug] committee: ents={ents.round(4)}, std(disagree)={disagree:.4f}")

    if disagree > disagree_thr:
        # Winner-takes-all: choose single most confident
        best_idx = int(ents.argmin())
        best = outputs[best_idx]
        best.tool = ToolChoice(f"{best.tool.name}_winner", {})
        if debug:
            print(f"[debug] disagreement>{disagree_thr:.3f} → pick {best.tool.name}")
        return best

    # 3) Entropy-weighted logit averaging
    weights = 1.0 / (ents + 1e-6)
    weights = weights / weights.sum()
    logits_stack = np.stack([o.logits for o in outputs], axis=0)         # [T, N, C]
    logits_wmean = (weights[:, None, None] * logits_stack).sum(axis=0)   # [N, C]
    probs = softmax(logits_wmean)
    agg = ToolOutput(
        tool=ToolChoice("committee_wmean", {}),
        logits=logits_wmean,
        probs=probs,
        stats={}
    )
    if debug:
        print(f"[debug] weighted-avg weights={weights.round(3)}")
    return agg


# ---------- Orchestrator ----------

class Orchestrator:
    """
    End-to-end routing & inference.

    - Profiles target graph → fingerprint φ(G)
    - Uses capability profiles + UCB planner to pick tools
    - Runs selected tools
    - Aggregates outputs robustly
    - Computes label-free utility and logs an episode
    """

    def __init__(self,
                 tools: List,
                 episodic_store: EpisodicStore | None = None,
                 cap_profiles: CapabilityProfiles | None = None,
                 kappa: float = 0.0,
                 committee_k: int = 1,
                 diversity: bool = True,
                 disagree_thr: float = 0.05,
                 conf_max_entropy: float = 1.10,
                 debug: bool = True):
        self.tools = tools
        self.store = episodic_store or EpisodicStore()
        self.cap = cap_profiles or CapabilityProfiles()
        self.planner = UCBPlanner(self.cap, k=committee_k, kappa=kappa, diversity=diversity)
        self.disagree_thr = disagree_thr
        self.conf_max_entropy = conf_max_entropy
        self.debug = debug

    def _tool_meta(self) -> List[Dict]:
        meta = []
        for t in self.tools:
            fam = getattr(t, "family", None)
            meta.append({
                "name": getattr(t, "name", t.__class__.__name__),
                "family": fam if fam is not None else "NA",
            })
        return meta

    def _lookup_tool(self, name: str):
        for t in self.tools:
            if getattr(t, "name", None) == name:
                return t
        # fallback by class name
        for t in self.tools:
            if t.__class__.__name__ == name:
                return t
        raise KeyError(f"Selected tool '{name}' not found among { [getattr(x,'name',x.__class__.__name__) for x in self.tools] }")

    def infer(self, G_test: Dict) -> Tuple[ToolOutput, Dict]:
        # 1) Profile
        phi = compute_fingerprint(G_test)

        # 2) Plan
        tools_meta = self._tool_meta()
        chosen_meta = self.planner.select(phi.vector, tools_meta)  # list of dicts with 'name'
        chosen_names = [m["name"] for m in chosen_meta]
        if self.debug:
            print(f"[debug] planner chose: {chosen_names}")

        # 3) Execute
        outputs: List[ToolOutput] = []
        for m in chosen_meta:
            tool = self._lookup_tool(m["name"])
            out = tool.predict(G_test)          # ToolOutput
            outputs.append(out)

        # 4) Aggregate (robust)
        final = _aggregate_outputs(
            outputs,
            disagree_thr=self.disagree_thr,
            conf_max_entropy=self.conf_max_entropy,
            debug=self.debug
        )

        # 5) Utility (label-free)
        util = label_free_utility(final.probs)

        # 6) Log minimal episode (fingerprint + selection + utility)
        self.store.add(phi, meta={"chosen": chosen_names, "utility": util})

        meta = {
            "chosen": chosen_names if final.tool.name != "committee_wmean" else chosen_names + ["(wmean)"],
            "utility": util,
            "phi": phi,
        }
        return final, meta
