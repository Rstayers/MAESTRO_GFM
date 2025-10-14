# maestro/tools/gfm_ofa_small.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from ..types import ToolChoice, ToolOutput
from ..utils.metrics import entropy

class GFM_OFASmall:
    """
    TAG-aware 'GFM' stand-in:
      - TF-IDF on node text -> SVD(256) -> multinomial logistic regression
      - Accepts optional prototype vectors to init/shift the classifier (proto-as-prior)
    """
    def __init__(self, name="gfm_ofa_small", svd_dim=256, C=2.0, random_state=42):
        self.name = name
        self.family = "TAG"
        self.vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
        self.svd = TruncatedSVD(n_components=svd_dim, random_state=random_state)
        self.clf = LogisticRegression(max_iter=200, C=C, multi_class="multinomial", random_state=random_state)
        self.num_classes = None
        self.proto_bias = None  # optional class-wise bias from prototypes
        self._fitted = False

    def encode(self, texts):
        X = self.vectorizer.transform(texts)
        Z = self.svd.transform(X)  # [N, d]
        return Z
    def fit(self, texts, y):
        X = self.vectorizer.fit_transform(texts)
        Z = self.svd.fit_transform(X)
        self.num_classes = int(np.max(y) + 1)
        self.clf.fit(Z, y)
        self._fitted = True

    def init_with_prototypes(self, class_texts_by_id: dict[int, list[str]] | None):
        """
        Optional: pass dict class_id -> list of prototype texts.
        We compute their SVD means and store a small bias toward those classes.
        """
        if not class_texts_by_id:
            self.proto_bias = None
            return self
        means = np.zeros((self.num_classes, self.svd.n_components), dtype=np.float32)
        for c, texts in class_texts_by_id.items():
            if not texts: continue
            X = self.vectorizer.transform(texts)
            Z = self.svd.transform(X)
            means[c] = Z.mean(axis=0)
        # turn means into a simple bias by dot with class weight direction
        W = getattr(self.clf, "coef_", None)
        if W is not None and W.shape[0] == self.num_classes:
            self.proto_bias = means @ W.T  # [C, C] small bias matrix
        return self

    def predict(self, G_batch, with_tta=False):
        texts = G_batch.get("text")
        assert texts is not None, f"{self.name} requires node texts (TAG)."
        X = self.vectorizer.transform(texts)
        Z = self.svd.transform(X)
        logits = self.clf.decision_function(Z)  # [N, C]
        if self.proto_bias is not None:
            # add diagonal bias (class preference); small scale
            bias = 0.05 * np.diag(self.proto_bias)  # [C]
            logits = logits + bias[None, :]
        # softmax
        z = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(z); probs = probs / probs.sum(axis=1, keepdims=True)
        ent = float(entropy(probs).mean())
        return ToolOutput(tool=ToolChoice(self.name, {}), logits=logits, probs=probs, stats={"mean_entropy": ent})
