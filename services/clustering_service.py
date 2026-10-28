"""Feature construction (TF-IDF + genre one-hot) and K-Means clustering of books.

Representation: TF-IDF of description/subgenre words (what a book is *about*) plus a down-weighted one-hot
genre vector, L2-normalised. Limitation: short descriptions give sparse, noisy vectors, so clusters reflect
shared vocabulary and genre, not guaranteed reader interest.
"""
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from services.text_utils import is_blank

GENRE_WEIGHT = 0.6


@dataclass
class FeatureSpace:
    X_text: csr_matrix      # TF-IDF only, L2-normalised (used for content similarity)
    X_full: csr_matrix      # TF-IDF + genre one-hot, L2-normalised (used for clustering)
    terms: np.ndarray       # vocabulary of the TF-IDF part
    n_text: int


@dataclass
class ClusterResult:
    assignments: pd.DataFrame            # book_id, cluster, x, y
    summaries: list = field(default_factory=list)
    k: int = 1
    silhouette: float = None
    notes: list = field(default_factory=list)


def book_text(df: pd.DataFrame) -> pd.Series:
    desc = df["description"].where(~df["description"].map(is_blank), df["title"]).fillna("")
    sub = df["subgenre"].fillna("")
    return (desc + " " + sub).str.strip()  # genre enters separately via one-hot, so it is not double counted


def build_features(df: pd.DataFrame) -> FeatureSpace:
    n = len(df)
    if n == 0:
        z = csr_matrix((0, 0))
        return FeatureSpace(z, z, np.array([]), 0)
    try:
        vec = TfidfVectorizer(stop_words="english", max_features=500, ngram_range=(1, 2), sublinear_tf=True)
        T = vec.fit_transform(book_text(df))
        terms = vec.get_feature_names_out()
    except ValueError:  # empty vocabulary (e.g. only stop-words)
        T, terms = csr_matrix((n, 0)), np.array([])
    genres = df["genre"].fillna("Unknown")
    G = pd.get_dummies(genres).drop(columns=["Unknown"], errors="ignore")
    G = csr_matrix(G.values.astype(float) * GENRE_WEIGHT) if G.shape[1] else csr_matrix((n, 0))
    X_text = normalize(T) if T.shape[1] else T
    X_full = normalize(hstack([T, G]).tocsr()) if (T.shape[1] + G.shape[1]) else csr_matrix((n, 0))
    return FeatureSpace(X_text.tocsr(), X_full.tocsr(), terms, T.shape[1])


def _distinct_rows(X) -> int:
    if X.shape[1] == 0:
        return 0
    return len(np.unique(np.round(X.toarray(), 6), axis=0))


def cluster_books(df: pd.DataFrame, k: int = None, features: FeatureSpace = None, random_state: int = 42) -> ClusterResult:
    """Cluster books. k=None picks k in [2, min(8, n-1)] by best silhouette score. Never raises on tiny data."""
    n = len(df)
    df = df.reset_index(drop=True)
    notes = []
    if n == 0:
        return ClusterResult(pd.DataFrame(columns=["book_id", "cluster", "x", "y"]), [], 0, None,
                             ["The catalogue is empty - add or import books first."])
    fs = features or build_features(df)
    X = fs.X_full
    labels = np.zeros(n, dtype=int)
    sil = None
    distinct = _distinct_rows(X)
    if n < 4:
        notes.append("Fewer than 4 books: all books are placed in one cluster.")
    elif distinct < 2:
        notes.append("Books are not distinguishable by genre or description: one cluster only.")
    else:
        max_k = min(8, n - 1, distinct)
        candidates = [int(min(max(2, k), max(2, min(n - 1, distinct))))] if k else list(range(2, max_k + 1))
        best = None
        for kk in candidates:
            km = KMeans(n_clusters=kk, n_init=10, random_state=random_state).fit(X)
            if len(set(km.labels_)) < 2 or len(set(km.labels_)) >= n:
                continue
            s = silhouette_score(X, km.labels_)
            if best is None or s > best[0]:
                best = (s, km.labels_)
        if best:
            sil, labels = float(best[0]), best[1]
            if not k:
                notes.append("Number of clusters chosen automatically by the best silhouette score.")
            if n < 15:
                notes.append("Small catalogue: treat clusters as illustrative, not statistically stable.")
        else:
            notes.append("Clustering could not separate the books: one cluster only.")
    xy = np.zeros((n, 2))
    if X.shape[1] >= 2 and n >= 2:
        with warnings.catch_warnings():  # degenerate (near-constant) data triggers a harmless divide warning
            warnings.simplefilter("ignore", RuntimeWarning)
            xy = TruncatedSVD(n_components=2, random_state=random_state).fit_transform(X)
    elif X.shape[1] == 1:
        xy[:, 0] = X.toarray()[:, 0]
    assign = pd.DataFrame({"book_id": df["book_id"], "cluster": labels, "x": xy[:, 0], "y": xy[:, 1]})
    return ClusterResult(assign, _summaries(df, assign, fs), int(len(set(labels))), sil, notes)


def _summaries(df, assign, fs: FeatureSpace) -> list:
    out = []
    for c in sorted(assign["cluster"].unique()):
        idx = np.where(assign["cluster"].values == c)[0]
        sub = df.iloc[idx]
        top_genres = [(g, round(float(s) * 100)) for g, s in sub["genre"].value_counts(normalize=True).head(3).items()]
        terms = []
        if fs.n_text:
            centroid = np.asarray(fs.X_text[idx].mean(axis=0)).ravel()
            terms = [fs.terms[i] for i in centroid.argsort()[::-1][:4] if centroid[i] > 0]
        dom, pct = top_genres[0]
        label = dom + (f" ({pct}%)" if pct < 100 else "") + (f": {', '.join(terms[:2])}" if terms else "")
        expl = (f"{len(idx)} book(s), mostly {dom}. Books in this group share vocabulary such as "
                f"{', '.join(terms) if terms else 'n/a'}; e.g. {', '.join(sub['title'].head(3))}.")
        out.append({"cluster": int(c), "size": len(idx), "top_genres": top_genres, "top_terms": terms,
                    "label": label, "explanation": expl})
    return out
