"""Explainable hybrid recommender.

score = sum_c w_c * component_c, with components in [0,1]:
  content  - cosine similarity (TF-IDF of description/subgenre) to the user's weighted taste vector
  genre    - normalised genre-preference score from reading history (or onboarding choices)
  cluster  - share of the user's liked books that fall in the candidate's K-Means cluster
  author   - 1 if the user liked another book by the same author
  collab   - user-based CF; only used when >= MIN_CF_USERS users have enough signals and overlap exists
Weights are renormalised, so unavailable components (e.g. collab) do not distort the score.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config
from services.clustering_service import build_features, cluster_books
from services.text_utils import author_surname, is_blank

REC_COLUMNS = ["rank", "book_id", "title", "author", "genre", "description", "score", "content", "genre_pref",
               "cluster_affinity", "author_affinity", "collab", "explanation", "availability", "info_source"]


def history_weight(status, rating, liked) -> float:
    """Signed preference signal of one history row (documented, transparent)."""
    if not is_blank(liked) and int(liked) != 0 and is_blank(rating):
        return 1.0 if int(liked) > 0 else -1.0
    if status == "interested":
        return 0.5
    if status == "reading":
        return 0.7
    if is_blank(rating):
        return 1.0 if is_blank(liked) or int(liked) >= 0 else -1.0
    return {5: 1.0, 4: 0.8, 3: 0.5, 2: 0.0, 1: -0.5}[int(rating)]


@dataclass
class UserProfile:
    weights: dict = field(default_factory=dict)         # book_id -> signed weight
    genre_scores: dict = field(default_factory=dict)    # genre -> 0..1
    genre_counts: dict = field(default_factory=dict)    # genre -> number of books read
    liked_surnames: dict = field(default_factory=dict)  # surname -> author display name
    excluded: set = field(default_factory=set)          # read / reading / disliked
    pref_genres: list = field(default_factory=list)
    source: str = "none"                                # history | preferences | none


@dataclass
class RecResult:
    table: pd.DataFrame
    notes: list
    profile: UserProfile
    weights_used: dict


def build_profile(history: pd.DataFrame, books: pd.DataFrame, user: dict = None) -> UserProfile:
    user = user or {}
    prefs_g, prefs_a = list(user.get("pref_genres", [])), list(user.get("pref_authors", []))
    p = UserProfile(pref_genres=prefs_g)
    bmap = books.set_index("book_id") if len(books) else books
    raw_genre = {}
    if history is not None and len(history):
        for _, h in history.iterrows():
            if h["book_id"] not in bmap.index:
                continue
            b = bmap.loc[h["book_id"]]
            w = history_weight(h["status"], h["rating"], h["liked"])
            p.weights[h["book_id"]] = w
            if h["status"] in ("read", "reading") or w < 0:
                p.excluded.add(h["book_id"])
            if h["status"] == "read":
                p.genre_counts[b["genre"]] = p.genre_counts.get(b["genre"], 0) + 1
            if b["genre"] != "Unknown":
                raw_genre[b["genre"]] = raw_genre.get(b["genre"], 0.0) + w
            if w >= 0.8 and not is_blank(b["author"]):
                p.liked_surnames[author_surname(b["author"])] = b["author"]
    if p.weights:
        p.source = "history"
    prior = 0.5 if p.weights else 1.0  # onboarding choices: a prior for everyone, the only signal for new users
    for g in prefs_g:
        raw_genre[g] = raw_genre.get(g, 0.0) + prior
    for a in prefs_a:
        p.liked_surnames.setdefault(author_surname(a), a)
    if not p.weights and (prefs_g or prefs_a):
        p.source = "preferences"
        pa = [author_surname(a) for a in prefs_a]
        for _, b in books.iterrows():
            if b["genre"] in prefs_g or author_surname(b["author"]) in pa:
                p.weights[b["book_id"]] = 0.6  # pseudo-signal from stated preferences
    mx = max([v for v in raw_genre.values() if v > 0], default=0)
    p.genre_scores = {g: max(v, 0) / mx for g, v in raw_genre.items()} if mx > 0 else {}
    return p


def collaborative_scores(all_history: pd.DataFrame, target_user: str, min_users=config.MIN_CF_USERS,
                         min_signals=config.MIN_CF_SIGNALS_PER_USER):
    """User-based CF. Returns (Series(book_id -> score) | None, reason_if_none)."""
    if all_history is None or not len(all_history):
        return None, "no history data"
    h = all_history.copy()
    h["w"] = [history_weight(s, r, l) for s, r, l in zip(h["status"], h["rating"], h["liked"])]
    counts = h.groupby("user_id").size()
    eligible = counts[counts >= min_signals].index
    if len(eligible) < min_users:
        return None, f"only {len(eligible)} user(s) have >= {min_signals} rated/read books (need {min_users})"
    if target_user not in eligible:
        return None, "this user has too few rated/read books"
    M = h[h["user_id"].isin(eligible)].pivot_table(index="user_id", columns="book_id", values="w", fill_value=0.0)
    t = M.loc[target_user].values
    if not np.any(t):
        return None, "this user has no usable signals"
    sims = {}
    for u in M.index:
        if u == target_user:
            continue
        v = M.loc[u].values
        d = np.linalg.norm(t) * np.linalg.norm(v)
        s = float(t @ v / d) if d else 0.0
        if s > 0:
            sims[u] = s
    if not sims:
        return None, "no other user shares liked books with this user"
    score = sum(M.loc[u] * s for u, s in sims.items())
    return score.clip(lower=0), ""


class Recommender:
    def __init__(self, books: pd.DataFrame, k_clusters: int = None):
        self.books = books.reset_index(drop=True)
        self.features = build_features(self.books)
        self.clusters = cluster_books(self.books, k=k_clusters, features=self.features)
        self.cluster_of = dict(zip(self.clusters.assignments["book_id"], self.clusters.assignments["cluster"]))
        self.cluster_label = {s["cluster"]: s["label"] for s in self.clusters.summaries}

    def recommend(self, history: pd.DataFrame, user: dict = None, top_n: int = 10, candidate_ids=None,
                  exclude_read: bool = True, genres_include=None, weights: dict = None,
                  all_history: pd.DataFrame = None) -> RecResult:
        notes = []
        if not len(self.books):
            return RecResult(pd.DataFrame(columns=REC_COLUMNS), ["The catalogue is empty."], UserProfile(), {})
        profile = build_profile(history, self.books, user)
        ids = self.books["book_id"].values
        mask = np.ones(len(self.books), dtype=bool)
        if candidate_ids is not None:
            mask &= np.isin(ids, list(candidate_ids))
        if exclude_read:
            mask &= ~np.isin(ids, list(profile.excluded))
        if genres_include:
            mask &= self.books["genre"].isin(genres_include).values
        cand = np.where(mask)[0]
        if len(cand) == 0:
            return RecResult(pd.DataFrame(columns=REC_COLUMNS), ["No candidate books remain after filtering."], profile, {})
        if profile.source == "none":
            return self._starter_picks(cand, top_n, profile)

        Xt = self.features.X_text
        content = np.zeros(len(self.books))
        if Xt.shape[1]:
            vec = np.zeros(Xt.shape[1])
            for i, b in enumerate(ids):
                w = profile.weights.get(b)
                if w:
                    vec += (w if w > 0 else 0.5 * w) * Xt[i].toarray().ravel()
            nrm = np.linalg.norm(vec)
            if nrm > 1e-9:
                content = np.asarray(np.clip(Xt @ (vec / nrm), 0, None)).ravel()
        cmax = content[cand].max()
        content_n = content / cmax if cmax > 0 else content

        genre = np.array([profile.genre_scores.get(g, 0.0) for g in self.books["genre"]])
        mass = {}
        for b, w in profile.weights.items():
            if w > 0 and b in self.cluster_of:
                mass[self.cluster_of[b]] = mass.get(self.cluster_of[b], 0) + w
        mmax = max(mass.values(), default=0)
        cluster = np.array([mass.get(self.cluster_of[b], 0) / mmax if mmax else 0.0 for b in ids])
        author = np.array([1.0 if author_surname(a) in profile.liked_surnames else 0.0 for a in self.books["author"]])

        collab = np.zeros(len(self.books))
        cf, cf_reason = None, "no user context"
        if all_history is not None and user and user.get("user_id"):
            cf, cf_reason = collaborative_scores(all_history, user["user_id"])
        if cf is not None:
            collab = np.array([cf.get(b, 0.0) for b in ids])
            cm = collab[cand].max()
            collab = collab / cm if cm > 0 else collab
        else:
            notes.append(f"Collaborative filtering skipped: {cf_reason}.")

        w = dict(config.DEFAULT_WEIGHTS)
        w.update(weights or {})
        if cf is None:
            w["collab"] = 0.0
        tot = sum(w.values()) or 1.0
        w = {k: v / tot for k, v in w.items()}
        total = (w["content"] * content_n + w["genre"] * genre + w["cluster"] * cluster
                 + w["author"] * author + w["collab"] * collab)

        order = cand[np.argsort(-total[cand], kind="stable")][:top_n]
        rows = []
        for rank, i in enumerate(order, 1):
            b = self.books.iloc[i]
            comp = {"content": content_n[i], "genre": genre[i], "cluster": cluster[i], "author": author[i], "collab": collab[i]}
            rows.append({
                "rank": rank, "book_id": b["book_id"], "title": b["title"], "author": b["author"], "genre": b["genre"],
                "description": b["description"], "score": round(float(total[i]), 4),
                "content": round(float(comp["content"]), 3), "genre_pref": round(float(comp["genre"]), 3),
                "cluster_affinity": round(float(comp["cluster"]), 3), "author_affinity": round(float(comp["author"]), 3),
                "collab": round(float(comp["collab"]), 3),
                "explanation": self._explain(i, comp, w, profile),
                "availability": b["availability"], "info_source": b["source"],
            })
        if profile.source == "preferences":
            notes.append("New user: ranking is based on the genres/authors chosen during onboarding.")
        return RecResult(pd.DataFrame(rows, columns=REC_COLUMNS), notes, profile, w)

    def _explain(self, i, comp, w, profile: UserProfile) -> str:
        b = self.books.iloc[i]
        reasons = []
        for val, k in sorted(((w[k] * v, k) for k, v in comp.items()), reverse=True):
            if val <= 0 or len(reasons) == 2:
                continue
            g = b["genre"]
            if k == "genre":
                n = profile.genre_counts.get(g, 0)
                reasons.append(f"you frequently read {g} books ({n} read)" if n >= 2 else
                               f"you read a {g} book and rated it positively" if n == 1 else
                               f"you selected {g} as a preferred genre")
            elif k == "content":
                sim = self._most_similar_liked(i, profile)
                reasons.append(f"its description is similar to '{sim}' (which you liked)" if sim
                               else "its description matches your taste profile")
            elif k == "author":
                reasons.append(f"you liked other books by {b['author']}")
            elif k == "cluster":
                reasons.append("it belongs to the same theme group "
                               f"('{self.cluster_label.get(self.cluster_of[b['book_id']], '')}') as books you liked")
            elif k == "collab":
                reasons.append("readers with similar ratings also liked it")
        return ("Recommended because " + ", and ".join(reasons) + ".") if reasons else "Low-confidence match."

    def _most_similar_liked(self, i, profile):
        Xt = self.features.X_text
        if Xt.shape[1] == 0:
            return None
        best, name = 0.0, None
        for j, bid in enumerate(self.books["book_id"]):
            if profile.weights.get(bid, 0) >= 0.5 and j != i:
                s = float(Xt[i].multiply(Xt[j]).sum())
                if s > best:
                    best, name = s, self.books.iloc[j]["title"]
        return name

    def _starter_picks(self, cand, top_n, profile) -> RecResult:
        picks, seen = [], set()
        for i in sorted(cand, key=lambda i: (self.books.iloc[i]["genre"], self.books.iloc[i]["title"])):
            g = self.books.iloc[i]["genre"]
            if g not in seen:
                seen.add(g)
                picks.append(i)
        rows = []
        for rank, i in enumerate(picks[:top_n], 1):
            b = self.books.iloc[i]
            rows.append({"rank": rank, "book_id": b["book_id"], "title": b["title"], "author": b["author"],
                         "genre": b["genre"], "description": b["description"], "score": 0.0, "content": 0.0,
                         "genre_pref": 0.0, "cluster_affinity": 0.0, "author_affinity": 0.0, "collab": 0.0,
                         "explanation": f"Starter pick from {b['genre']}: no history or preferences yet, so this is a "
                                        "sample from each genre to help you choose.",
                         "availability": b["availability"], "info_source": b["source"]})
        return RecResult(pd.DataFrame(rows, columns=REC_COLUMNS),
                         ["No reading history or preferences: showing one starter pick per genre."], profile, {})


def evaluate_leave_one_out(books: pd.DataFrame, all_history: pd.DataFrame, users: pd.DataFrame, k: int = 5,
                           min_positive: int = 3, max_holdouts_per_user: int = 5) -> dict:
    """Leave-one-out on liked 'read' items (rating >= 4). Reports HitRate@k and MRR.

    Valid only as a smoke test on small demo data - NOT an experimental result. Real evaluation needs a
    held-out interaction log from real readers, many users, and comparison against baselines.
    """
    rec = Recommender(books)
    cases, ranks = 0, []
    for u in users.itertuples():
        h = all_history[all_history["user_id"] == u.user_id]
        pos = h[(h["status"] == "read") & (h["rating"].fillna(0) >= 4)]
        if len(pos) < min_positive:
            continue
        user = {"user_id": u.user_id, "pref_genres": [], "pref_authors": []}  # no onboarding prior: avoids leakage
        for _, held in pos.head(max_holdouts_per_user).iterrows():
            h_train = h[h["book_id"] != held["book_id"]]
            all_train = all_history[~((all_history["user_id"] == u.user_id) & (all_history["book_id"] == held["book_id"]))]
            res = rec.recommend(h_train, user, top_n=len(books), all_history=all_train)
            order = list(res.table["book_id"])
            cases += 1
            ranks.append(order.index(held["book_id"]) + 1 if held["book_id"] in order else None)
    valid = [r for r in ranks if r]
    return {"cases": cases, "k": k,
            "hit_rate_at_k": (sum(1 for r in valid if r <= k) / cases) if cases else None,
            "mrr": (sum(1 / r for r in valid) / cases) if cases else None,
            "candidate_pool": len(books)}
