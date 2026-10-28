import unittest

from services.text_utils import (author_surname, extract_isbns, is_valid_isbn, normalize_genre, normalize_title,
                                 similarity, to_year)


class TextUtilsTests(unittest.TestCase):
    def test_isbn_validation(self):
        self.assertTrue(is_valid_isbn("978-0-306-40615-7"))
        self.assertTrue(is_valid_isbn("0-306-40615-2"))
        self.assertFalse(is_valid_isbn("978-0-306-40615-8"))
        self.assertFalse(is_valid_isbn("12345"))
        self.assertFalse(is_valid_isbn(None))

    def test_isbn_extraction_from_ocr_text(self):
        self.assertEqual(extract_isbns("Price 9.99 | ISBN 978-0-306-40615-7 | made in"), ["9780306406157"])
        self.assertEqual(extract_isbns("ISBN 978-0-306-40615-8"), [])  # bad checksum is ignored
        self.assertEqual(extract_isbns(""), [])

    def test_normalisation(self):
        self.assertEqual(normalize_title("The Hobbit"), normalize_title("hobbit!"))
        self.assertEqual(author_surname("J. K. Rowling"), author_surname("Rowling, J.K."))
        self.assertEqual(author_surname("Stuart Russell and Peter Norvig"), "russell")
        self.assertEqual(author_surname(None), "")

    def test_genre_and_year(self):
        self.assertEqual(normalize_genre("sci-fi"), "Science Fiction")
        self.assertEqual(normalize_genre(""), "Unknown")
        self.assertEqual(normalize_genre(float("nan")), "Unknown")
        self.assertEqual(to_year("1999.0"), 1999)
        self.assertIsNone(to_year("abc"))
        self.assertIsNone(to_year(12))

    def test_similarity(self):
        self.assertGreater(similarity("Harry Potter Philosophers Stone", "Harry Potter and the Philosopher's Stone"), 0.8)
        self.assertLess(similarity("Dune", "Frankenstein"), 0.5)
        self.assertEqual(similarity("", "x"), 0.0)


if __name__ == "__main__":
    unittest.main()
