"""Shallow benchmarks the MLP has to beat. All use the same excerpts and splits.

  majority      always predict the most common training label. A floor.
  tfidf_logreg  TF-IDF (word + char n-grams, full sparse vocabulary) into a
                class-balanced logistic regression. The honest shallow benchmark:
                linear, no hidden layer, strong on lexical log signals.
  linear_probe  logistic regression on exactly the features the MLP sees. The
                gap between this and the MLP is what the hidden layer adds.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class Majority:
    def fit(self, X, y):
        self.label = int(np.bincount(y).argmax())
        return self

    def predict(self, X):
        return np.full(len(X), self.label)


class TfidfLogReg:
    def __init__(self, C: float = 4.0, seed: int = 7400):
        self.C, self.seed = C, seed

    ## this is data pre-processing to vectorize the text inputs
    ## called in the fit() funciton on-the-fly to create the matrix of transformed text
    def _X(self, texts, fit=False):
        if fit:
            self.word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                        token_pattern=r"(?u)\b\w[\w.+-]*\b")
            self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                        sublinear_tf=True, max_features=50000)
            return hstack([self.word.fit_transform(texts), self.char.fit_transform(texts)]).tocsr()
        return hstack([self.word.transform(texts), self.char.transform(texts)]).tocsr()

    def fit(self, texts, y):
        self.clf = LogisticRegression(C=self.C, class_weight="balanced", max_iter=5000,
                                      random_state=self.seed)
        self.clf.fit(self._X(texts, fit=True), y)
        return self

    def predict(self, texts):
        return self.clf.predict(self._X(texts))


def linear_probe(X_train, y_train, seed: int = 7400):
    return LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000,
                              random_state=seed).fit(X_train, y_train)
