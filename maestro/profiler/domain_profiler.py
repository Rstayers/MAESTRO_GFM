import numpy as np
import networkx as nx
from ..types import DomainFingerprint


# Expect G dict: {'edge_index': np.ndarray shape [2,E], 'x': np.ndarray [N,D] or None, 'text': Optional[List[str]]}


def compute_fingerprint(G) -> DomainFingerprint:
    edge_index = G['edge_index']
    N = G['x'].shape[0] if G.get('x') is not None else int(edge_index.max()) + 1
    g_nx = nx.Graph()
    g_nx.add_nodes_from(range(N))
    g_nx.add_edges_from(edge_index.T.tolist())


    degs = np.array([d for _, d in g_nx.degree()])
    deg_mean = float(degs.mean())
    deg_std = float(degs.std())
    deg_gini = _gini(degs.astype(float))


    clustering = float(nx.average_clustering(g_nx)) if N < 100000 else 0.0
    assort = float(nx.degree_assortativity_coefficient(g_nx)) if g_nx.number_of_edges() > 0 else 0.0


    # homophily proxy: cosine similarity between neighbors' feature vectors (label-free)
    homoph = 0.0
    X = G.get('x')
    if X is not None:
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        sims = []
        for u, v in edge_index.T:
            sims.append((Xn[u] * Xn[v]).sum())
        if sims:
            homoph = float(np.mean(sims))


    # quick spectral sketch: largest eigenvalue of normalized Laplacian via power iteration on adjacency
    spec1 = float(_spectral_sketch_largest(g_nx))


    is_TAG = 1.0 if G.get('text') is not None else 0.0


    struct_stats = dict(deg_mean=deg_mean, deg_std=deg_std, deg_gini=deg_gini,
    clustering=clustering, assort=assort, homophily=homoph, spec_1=spec1)
    schema_stats = dict(is_TAG=is_TAG)
    task_meta = {}


    vec = np.array([deg_mean, deg_std, deg_gini, clustering, assort, homoph, spec1, is_TAG], dtype=float)
    vec = (vec - vec.mean()) / (vec.std() + 1e-8) # simple standardization per‑graph for MVP


    return DomainFingerprint(struct_stats=struct_stats, schema_stats=schema_stats, task_meta=task_meta, vector=vec)




def _gini(x: np.ndarray) -> float:
    if len(x) == 0: return 0.0
    xs = np.sort(np.abs(x))
    n = len(xs)
    cumx = np.cumsum(xs)
    g = (n + 1 - 2 * (cumx.sum() / cumx[-1])) / n if cumx[-1] > 0 else 0.0
    return float(g)




def _spectral_sketch_largest(g: nx.Graph, iters: int = 20) -> float:
    # power iteration on adjacency for a rough largest eigenvalue sketch
    N = g.number_of_nodes()
    if N == 0: return 0.0
    v = np.random.randn(N)
    A = nx.to_scipy_sparse_array(g, format='csr', dtype=float)
    for _ in range(iters):
        v = A @ v
        nrm = np.linalg.norm(v) + 1e-8
        v = v / nrm
    lam = float(np.linalg.norm(A @ v))
    return lam