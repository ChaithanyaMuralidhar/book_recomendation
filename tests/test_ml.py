import unittest

import pandas as pd

from services.clustering_service import cluster_books
from services.recommendation_service import (Recommender, collaborative_scores, evaluate_leave_one_out,
                                             history_weight)
from tests.helpers import make_services


def books_df(rows):
    return pd.DataFrame([{"book_id": f"B{i + 1:03d}", "title": t, "author": a, "genre": g, "subgenre": "",
                          "description": d, "source": "test", "availability": "Unknown"}
                         for i, (t, a, g, d) in enumerate(rows)])


class RecommenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = make_services(sample=True)
        cls.books = cls.s.cat.list_books()
        cls.rec = Recommender(cls.books)

    @classmethod
    def tearDownClass(cls):
        cls.s.cleanup()

    def _recs(self, uid, **kw):
        return self.rec.recommend(self.s.users.get_history(uid), self.s.users.get_user(uid), all_history=self.s.users.get_history(), **kw)

    def test_mystery_reader_gets_mysteries_and_not_read_books(self):
        res = self._recs("U001", top_n=5)
        self.assertIn(res.table.iloc[0]["genre"], ("Mystery", "Thriller"))
        read = set(self.s.users.get_history("U001").query("status == 'read'")["book_id"])
        self.assertFalse(read & set(res.table["book_id"]))
        self.assertTrue(res.table["score"].is_monotonic_decreasing)

    def test_disliked_books_are_excluded_and_negative_signal_used(self):
        self.assertEqual(history_weight("read", 1, None), -0.5)
        self.assertEqual(history_weight("read", None, -1), -1.0)
        hobbit = self.books[self.books.title == "The Hobbit"].book_id.iloc[0]
        res = self._recs("U001", top_n=50)  # U001 rated The Hobbit 2 (read)
        self.assertNotIn(hobbit, set(res.table["book_id"]))

    def test_explanations_present_and_reference_user_data(self):
        res = self._recs("U002", top_n=3)
        self.assertTrue(all(e.startswith("Recommended because") for e in res.table["explanation"]))
        self.assertTrue(res.table["explanation"].str.contains("Science Fiction|similar|similar ratings", regex=True).all())

    def test_genre_filter_and_candidate_restriction(self):
        res = self._recs("U001", top_n=10, genres_include=["Romance"])
        self.assertTrue((res.table["genre"] == "Romance").all())
        ids = list(self.books.book_id[:3])
        res = self._recs("U003", top_n=10, candidate_ids=ids)
        self.assertTrue(set(res.table["book_id"]) <= set(ids))

    def test_weights_change_ranking_inputs(self):
        a = self._recs("U001", top_n=3, weights={"genre": 1, "content": 0, "cluster": 0, "author": 0, "collab": 0})
        self.assertEqual(round(sum(a.weights_used.values()), 6), 1.0)
        self.assertEqual(a.weights_used["genre"], 1.0)

    def test_new_user_with_preferences(self):
        u = {"user_id": "NEW", "pref_genres": ["Fantasy"], "pref_authors": []}
        res = self.rec.recommend(pd.DataFrame(columns=["user_id", "book_id", "status", "rating", "liked"]), u)
        self.assertEqual(res.table.iloc[0]["genre"], "Fantasy")
        self.assertIn("preferred genre", res.table.iloc[0]["explanation"])

    def test_new_user_without_anything_gets_starter_picks(self):
        res = self.rec.recommend(self.s.users.get_history("NOPE"), {"user_id": "NOPE"})
        self.assertGreater(len(res.table), 5)
        self.assertEqual(len(set(res.table["genre"])), len(res.table))  # one per genre
        self.assertIn("starter", res.table.iloc[0]["explanation"].lower())

    def test_empty_catalogue(self):
        res = Recommender(books_df([]).reindex(columns=self.books.columns)).recommend(pd.DataFrame(), {"user_id": "x"})
        self.assertTrue(res.table.empty)
        self.assertTrue(res.notes)

    def test_collaborative_filtering_gating(self):
        sub = self.s.users.get_history()
        sub3 = sub[sub.user_id.isin(["U001", "U002", "U003"])]
        scores, reason = collaborative_scores(sub3, "U001")
        self.assertIsNone(scores)
        self.assertIn("need 5", reason)
        scores, _ = collaborative_scores(sub, "U001")
        self.assertIsNotNone(scores)
        self.assertIsNone(collaborative_scores(pd.DataFrame(), "U001")[0])

    def test_evaluation_runs_and_reports_cases(self):
        out = evaluate_leave_one_out(self.books, self.s.users.get_history(), self.s.users.list_users())
        self.assertGreater(out["cases"], 0)
        self.assertTrue(0 <= out["hit_rate_at_k"] <= 1 and 0 <= out["mrr"] <= 1)
        empty = evaluate_leave_one_out(self.books, self.s.users.get_history().iloc[0:0], self.s.users.list_users())
        self.assertEqual(empty["cases"], 0)


class ClusteringTests(unittest.TestCase):
    def test_empty_and_tiny_catalogues(self):
        self.assertEqual(cluster_books(books_df([]).reindex(columns=["book_id", "title", "genre", "subgenre", "description"])).k, 0)
        one = cluster_books(books_df([("A", "x", "Mystery", "a murder story")]))
        self.assertEqual(one.k, 1)
        three = cluster_books(books_df([("A", "x", "Mystery", "murder"), ("B", "y", "Romance", "love"), ("C", "z", "Fantasy", "dragon")]))
        self.assertEqual((three.k, len(three.assignments)), (1, 3))
        self.assertTrue(three.notes)

    def test_identical_books_do_not_crash(self):
        r = cluster_books(books_df([(f"T{i}", "a", "Mystery", "same text") for i in range(6)]))
        self.assertEqual(r.k, 1)

    def test_missing_descriptions_and_unknown_genre(self):
        r = cluster_books(books_df([(f"Book {i}", "a", "Unknown", "") for i in range(8)]))
        self.assertGreaterEqual(r.k, 1)
        self.assertEqual(len(r.assignments), 8)

    def test_sample_dataset_gets_multiple_clusters_with_summaries(self):
        s = make_services(sample=True)
        self.addCleanup(s.cleanup)
        r = cluster_books(s.cat.list_books())
        self.assertGreaterEqual(r.k, 2)
        self.assertEqual(sum(x["size"] for x in r.summaries), 52)
        self.assertIsNotNone(r.silhouette)
        self.assertEqual(cluster_books(s.cat.list_books(), k=4).k, 4)


if __name__ == "__main__":
    unittest.main()
