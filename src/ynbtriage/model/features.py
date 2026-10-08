"""Turn excerpt text into a fixed-length float vector the MLP can take.

Three featurizers, all with the same fit/transform interface. Each is fit on the
TRAINING split only and then applied to val/test, so nothing about val/test
leaks into the features.

  tfidf  word 1-2 grams + character 3-5 grams, reduced to `dim` dense dimensions
         with truncated SVD, then standardized. Captures exact error tokens.
  embed  a frozen pretrained sentence encoder (default BAAI/bge-small-en-v1.5,
         384-d). Nothing is trained here; it is a fixed feature extractor.
         Captures that differently worded errors can mean the same thing.
  both   the two concatenated.

The encoder is downloaded from the Hugging Face Hub on first use. Embeddings are
cached on disk keyed by text hash, so retraining the MLP does not re-encode.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

DEFAULT_ENCODER = "BAAI/bge-small-en-v1.5"
DEFAULT_CACHE = Path(".cache/embeddings")


class TfidfFeaturizer:
    def __init__(self, dim: int = 256, seed: int = 7400):
        self.dim, self.seed = dim, seed

    def fit(self, texts):
        from scipy.sparse import hstack
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import StandardScaler

        self.word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                    token_pattern=r"(?u)\b\w[\w.+-]*\b")
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                    sublinear_tf=True, max_features=50000)
        X = hstack([self.word.fit_transform(texts), self.char.fit_transform(texts)]).tocsr()
        k = max(2, min(self.dim, X.shape[0] - 1, X.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=self.seed)
        Z = self.svd.fit_transform(X)
        self.scaler = StandardScaler().fit(Z)
        return self

    def transform(self, texts) -> np.ndarray:
        from scipy.sparse import hstack
        X = hstack([self.word.transform(texts), self.char.transform(texts)]).tocsr()
        return self.scaler.transform(self.svd.transform(X)).astype(np.float32)


class EmbeddingFeaturizer:
    def __init__(self, model_name: str = DEFAULT_ENCODER, cache_dir: Path | str = DEFAULT_CACHE,
                 batch_size: int = 32):
        self.model_name, self.cache_dir, self.batch_size = model_name, Path(cache_dir), batch_size
        self._model = None

    def __getstate__(self):  # don't pickle the encoder weights into the artifact
        d = self.__dict__.copy()
        d["_model"] = None
        return d

    def _encoder(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def _cache_path(self) -> Path:
        return self.cache_dir / (self.model_name.replace("/", "__") + ".npz")

    def fit(self, texts):
        return self  # frozen: nothing to fit

    def transform(self, texts) -> np.ndarray:
        texts = list(texts)
        keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
        path = self._cache_path()
        cache = dict(np.load(path)) if path.exists() else {}
        missing = [i for i, k in enumerate(keys) if k not in cache]
        if missing:
            vecs = self._encoder().encode([texts[i] for i in missing], batch_size=self.batch_size,
                                          normalize_embeddings=True, show_progress_bar=False)
            for i, v in zip(missing, vecs):
                cache[keys[i]] = v.astype(np.float32)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, **cache)
        return np.stack([cache[k] for k in keys]).astype(np.float32)


class ConcatFeaturizer:
    def __init__(self, parts):
        self.parts = parts

    def fit(self, texts):
        for p in self.parts:
            p.fit(texts)
        return self

    def transform(self, texts) -> np.ndarray:
        return np.hstack([p.transform(texts) for p in self.parts])


def make_featurizer(kind: str, tfidf_dim: int = 256, encoder: str = DEFAULT_ENCODER):
    if kind == "tfidf":
        return TfidfFeaturizer(dim=tfidf_dim)
    if kind == "embed":
        return EmbeddingFeaturizer(model_name=encoder)
    if kind == "both":
        return ConcatFeaturizer([EmbeddingFeaturizer(model_name=encoder), TfidfFeaturizer(dim=tfidf_dim)])
    raise ValueError(f"unknown featurizer '{kind}' (tfidf | embed | both)")
