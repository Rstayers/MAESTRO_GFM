# maestro/tools/gfm_text_mlp_small.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

from ..types import ToolChoice, ToolOutput
from ..utils.metrics import softmax, entropy


class GFM_TextMLPSmall:
    """
    TAG-aware 'GFM' stand-in:
      - TF-IDF (word n-grams) -> PCA -> MLP classifier
      - Provides encode() for embeddings
    """
    def __init__(
        self,
        name: str = "gfm_text_mlp_small",
        pca_dim: int = 256,
        hidden_sizes: tuple[int, ...] = (512, 256),
        max_features: int = 20000,
        random_state: int = 42,
    ):
        self.name = name
        self.family = "TAG"

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
        )
        self.pca = PCA(n_components=pca_dim, random_state=random_state)
        self.clf = MLPClassifier(
            hidden_layer_sizes=hidden_sizes,
            activation="relu",
            max_iter=800,
            random_state=random_state,
        )

        self._fitted = False
        self.num_classes: int | None = None

    # ---------- core training ----------

    def fit(self, texts: list[str], y: np.ndarray):
        X_tfidf = self.vectorizer.fit_transform(texts)
        X_red = self.pca.fit_transform(X_tfidf.toarray())
        self.num_classes = int(np.max(y) + 1)
        self.clf.fit(X_red, y)
        self._fitted = True
        return self

    # ---------- prototypes (optional; no-op for now) ----------

    def init_with_prototypes(self, class_texts_by_id: dict[int, list[str]] | None):
        # You could implement class-wise bias here if desired.
        return self

    # ---------- embeddings ----------

    def encode(self, texts: list[str]) -> np.ndarray:
        assert self._fitted, "GFM_TextMLPSmall not fitted"
        X_tfidf = self.vectorizer.transform(texts)
        X_red = self.pca.transform(X_tfidf.toarray())
        return X_red

    # ---------- prediction ----------

    def predict(self, G_batch, with_tta: bool = False) -> ToolOutput:
        assert self._fitted, "GFM_TextMLPSmall not fitted"

        texts = G_batch.get("text", None)
        if texts is None:
            raise ValueError(
                f"{self.name} requires G_batch['text']; "
                "pass synthetic texts from features if raw text is unavailable."
            )

        X_tfidf = self.vectorizer.transform(texts)
        X_red = self.pca.transform(X_tfidf.toarray())

        # MLPClassifier already exposes predict_proba
        probs = self.clf.predict_proba(X_red)
        logits = np.log(probs + 1e-8)
        mean_ent = float(entropy(probs).mean())

        return ToolOutput(
            tool=ToolChoice(self.name, {"family": self.family}),
            logits=logits,
            probs=probs,
            stats={"mean_entropy": mean_ent},
        )
