import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from ..types import ToolChoice, ToolOutput
from ..utils.metrics import softmax, entropy



class ToolTAG:
    def __init__(self, name='tool_tag', pca_dim=128, random_state=42):
        self.name = name
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.pca = PCA(n_components=pca_dim, random_state=random_state)
        self.clf = MLPClassifier(hidden_layer_sizes=(pca_dim,), max_iter=800, random_state=random_state)
        self.fitted = False
        self.num_classes = None


    def fit(self, texts, y):
        X_tfidf = self.vectorizer.fit_transform(texts)
        X_pca = self.pca.fit_transform(X_tfidf.toarray())
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        X_pca = self.scaler.fit_transform(X_pca)
        self.num_classes = int(np.unique(y).shape[0])
        self.clf.fit(X_pca, y)
        self.fitted = True


    def predict(self, G_batch, with_tta=False):
        texts = G_batch.get('text')
        X = G_batch.get('x')
        if texts is None:
            # fallback to numeric features
            feats = X
        else:
            X_tfidf = self.vectorizer.transform(texts)
            feats = self.pca.transform(X_tfidf.toarray())
        logits = self.clf.predict_proba(feats)
        probs = logits # sklearn returns probs already
        ent = float(entropy(probs).mean())
        return ToolOutput(tool=ToolChoice(self.name, {}), logits=np.log(probs + 1e-8), probs=probs, stats={'mean_entropy': ent})