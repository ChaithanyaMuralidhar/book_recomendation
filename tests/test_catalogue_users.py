import unittest

from tests.helpers import make_services


class CatalogueTests(unittest.TestCase):
    def setUp(self):
        self.s = make_services()
        self.addCleanup(self.s.cleanup)

    def test_empty_catalogue(self):
        df = self.s.cat.list_books()
        self.assertEqual(len(df), 0)
        self.assertIn("title", df.columns)
        self.assertEqual(self.s.users.get_history().shape[0], 0)

    def test_add_and_clean(self):
        r = self.s.cat.add_book({"title": "  Dune ", "author": "Frank Herbert", "genre": "sci-fi",
                                 "isbn": "bad", "publication_year": "1965.0", "scan_confidence": 7})
        self.assertEqual(r.status, "added")
        b = self.s.cat.get_book(r.book_id)
        self.assertEqual((b["title"], b["genre"], b["isbn"], b["publication_year"]), ("Dune", "Science Fiction", "", 1965))
        self.assertEqual(b["scan_confidence"], 1.0)  # clamped

    def test_missing_title_rejected(self):
        for bad in ({"title": ""}, {"title": None}, {"author": "x"}, {"title": float("nan")}):
            with self.assertRaises(ValueError):
                self.s.cat.add_book(bad)
        self.assertEqual(self.s.cat.add_many([{"title": ""}, {"title": "Ok"}])["invalid"], 1)

    def test_duplicate_detection_variants(self):
        a = self.s.cat.add_book({"title": "The Hobbit", "author": "J. R. R. Tolkien"})
        for t, au in [("hobbit", "Tolkien, J.R.R."), ("THE HOBBIT!", ""), ("The  Hobbit", "J.R.R. Tolkien")]:
            r = self.s.cat.add_book({"title": t, "author": au})
            self.assertIn(r.status, ("duplicate", "merged"))
            self.assertEqual(r.book_id, a.book_id)
        self.assertEqual(self.s.cat.count(), 1)
        self.assertEqual(self.s.cat.add_book({"title": "The Hobbit", "author": "Someone Else"}).status, "added")

    def test_isbn_duplicate(self):
        a = self.s.cat.add_book({"title": "Book A", "isbn": "978-0-306-40615-7"})
        r = self.s.cat.add_book({"title": "Totally different title", "isbn": "9780306406157"})
        self.assertEqual(r.book_id, a.book_id)

    def test_merge_never_overwrites_existing_values(self):
        a = self.s.cat.add_book({"title": "Dune", "author": "Frank Herbert", "genre": "Science Fiction",
                                 "description": "My own note"})
        r = self.s.cat.add_book({"title": "Dune", "author": "Frank Herbert", "genre": "Fantasy",
                                 "description": "Imported text", "subgenre": "Space Opera", "publication_year": 1965})
        self.assertEqual(r.status, "merged")
        b = self.s.cat.get_book(a.book_id)
        self.assertEqual((b["genre"], b["description"]), ("Science Fiction", "My own note"))
        self.assertEqual((b["subgenre"], b["publication_year"]), ("Space Opera", 1965))  # blanks were filled
        self.assertEqual(self.s.cat.add_book({"title": "Dune", "author": "Frank Herbert"}, "skip").status, "duplicate")

    def test_update_and_conflict(self):
        a = self.s.cat.add_book({"title": "A"}).book_id
        self.s.cat.add_book({"title": "B"})
        self.s.cat.update_book(a, {"genre": "Mystery"})
        self.assertEqual(self.s.cat.get_book(a)["genre"], "Mystery")
        with self.assertRaises(ValueError):
            self.s.cat.update_book(a, {"title": "B"})
        with self.assertRaises(KeyError):
            self.s.cat.update_book("B999", {"title": "X"})


class UserTests(unittest.TestCase):
    def setUp(self):
        self.s = make_services()
        self.addCleanup(self.s.cleanup)
        self.bid = self.s.cat.add_book({"title": "Dune", "genre": "Science Fiction"}).book_id
        self.uid = self.s.users.create_user("Tester")

    def test_scanning_does_not_mark_as_read(self):
        self.assertEqual(len(self.s.users.get_history(self.uid)), 0)  # in catalogue, but not read by the user

    def test_status_upsert_and_validation(self):
        self.s.users.set_status(self.uid, self.bid, "interested")
        self.s.users.set_status(self.uid, self.bid, "read", 5)
        h = self.s.users.get_history(self.uid)
        self.assertEqual((len(h), h.iloc[0]["status"], h.iloc[0]["rating"]), (1, "read", 5))
        for kw in ({"status": "owned"}, {"status": "read", "rating": 9}, {"status": "read", "liked": 5}):
            with self.assertRaises(ValueError):
                self.s.users.set_status(self.uid, self.bid, **kw)
        with self.assertRaises(KeyError):
            self.s.users.set_status("U999", self.bid, "read")
        with self.assertRaises(KeyError):
            self.s.users.set_status(self.uid, "B999", "read")

    def test_genre_frequency_and_delete_cascade(self):
        self.assertEqual(len(self.s.users.genre_frequency(self.uid)), 0)
        self.s.users.set_status(self.uid, self.bid, "read", 4)
        self.assertEqual(self.s.users.genre_frequency(self.uid)["Science Fiction"], 1)
        self.s.cat.delete_book(self.bid)
        self.assertEqual(len(self.s.users.get_history(self.uid)), 0)

    def test_user_requires_name(self):
        with self.assertRaises(ValueError):
            self.s.users.create_user("  ")


if __name__ == "__main__":
    unittest.main()
