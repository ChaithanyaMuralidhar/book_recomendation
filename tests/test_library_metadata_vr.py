import base64
import io
import json
import unittest
from unittest import mock

import requests

from services import vr_service
from services.metadata_service import MetadataService, genre_from_subjects
from services.recommendation_service import Recommender
from services import ocr_service
from tests.helpers import make_cover, make_services, table_image, to_bgr


class LibraryMatchingTests(unittest.TestCase):
    def setUp(self):
        self.s = make_services(sample=True)
        self.addCleanup(self.s.cleanup)

    def test_noisy_titles_match_and_unknown_do_not(self):
        recs = [{"title": "Harry Potter Philosophers Stone", "author": "J K Rowling"}, {"title": "dune", "author": ""},
                {"title": "Quantum Gardening", "author": "Elena Brooks"}, {"title": "Dunee", "author": "Frank Herbert"}]
        out = self.s.cat.match_records(recs)
        st = [o["match_status"] for o in out]
        self.assertEqual(st[0], "matched")
        self.assertEqual(st[1], "matched")
        self.assertEqual(st[2], "unmatched")
        self.assertIsNone(out[2]["matched_book_id"])
        self.assertEqual(st[3], "matched")

    def test_isbn_is_exact_match(self):
        bid = self.s.cat.add_book({"title": "Odd Book", "isbn": "9780306406157"}).book_id
        out = self.s.cat.match_records([{"title": "garbled ocr", "isbn": "978-0-306-40615-7"}])[0]
        self.assertEqual((out["matched_book_id"], out["match_score"]), (bid, 1.0))

    def test_recommend_only_from_photographed_subset(self):
        recs = self.s.cat.match_records([{"title": "The Big Sleep"}, {"title": "Dune"}, {"title": "Persuasion"}])
        ids = {r["matched_book_id"] for r in recs if r["match_status"] == "matched"}
        self.assertEqual(len(ids), 3)
        h = self.s.users.get_history("U001")
        res = Recommender(self.s.cat.list_books()).recommend(h, self.s.users.get_user("U001"), 5, ids)
        self.assertTrue(set(res.table["book_id"]) <= ids)
        self.assertEqual(res.table.iloc[0]["title"], "The Big Sleep")  # best fit for a mystery reader

    @unittest.skipUnless(ocr_service.is_available(), "Tesseract not installed")
    def test_end_to_end_photo(self):
        img = table_image([make_cover(["DUNE"], "Frank Herbert"), make_cover(["THE", "HOBBIT"], "J. R. R. Tolkien"),
                           make_cover(["Quantum", "Gardening"], "Elena Brooks")])
        rep = self.s.scanner.scan(to_bgr(img))
        m = self.s.cat.match_records(rep.records)
        got = {r["title"]: r["match_status"] for r in m}
        self.assertEqual(got.get("Dune"), "matched")
        self.assertEqual(got.get("The Hobbit"), "matched")
        self.assertEqual(got.get("Quantum Gardening"), "unmatched")


class MetadataTests(unittest.TestCase):
    def test_offline_fallback_and_unknown(self):
        m = MetadataService(online=False)
        r = m.identify("Dune", "Frank Herbert")
        self.assertTrue(r.found)
        self.assertEqual((r.source, r.record["genre"]), ("local", "Science Fiction"))
        self.assertFalse(m.identify("Zzzz Qqqq Unknown Book", "Nobody").found)

    def test_noisy_ocr_text_match(self):
        row, score = MetadataService(online=False).match_text("THE | HOBITT | J R R TOLKIEN | 5 99")
        self.assertEqual(row["title"], "The Hobbit")

    def test_online_lookup_parsed_and_marked(self):
        sess = mock.Mock()
        search = {"docs": [{"key": "/works/OL1W", "title": "Quantum Gardening", "author_name": ["E. Brooks"],
                            "first_publish_year": 2001, "subject": ["Gardening", "Popular science"], "language": ["eng"]}]}
        work = {"description": {"value": "A book about plants."}}
        sess.get.side_effect = [mock.Mock(json=lambda: search, raise_for_status=lambda: None),
                                mock.Mock(json=lambda: work, raise_for_status=lambda: None)]
        r = MetadataService(online=True, session=sess, sample_csv="missing.csv").identify("Quantum Gardening", "Elena Brooks")
        self.assertEqual((r.found, r.source, r.record["genre"], r.record["description"]),
                         (True, "openlibrary", "Science", "A book about plants."))

    def test_network_failures_trip_circuit_breaker(self):
        sess = mock.Mock()
        sess.get.side_effect = requests.ConnectionError("offline")
        m = MetadataService(online=True, session=sess, sample_csv="missing.csv")
        for _ in range(5):
            self.assertFalse(m.identify("Some Title").found)
        self.assertEqual(sess.get.call_count, 2)  # stopped trying after ONLINE_MAX_FAILURES

    def test_genre_mapping(self):
        self.assertEqual(genre_from_subjects(["Detective and mystery stories"]), "Mystery")
        self.assertEqual(genre_from_subjects([]), "Unknown")


class VRTests(unittest.TestCase):
    def setUp(self):
        self.s = make_services()
        self.addCleanup(self.s.cleanup)

    def _png_b64(self):
        b = io.BytesIO()
        make_cover(["X"], "Y", size=(200, 300)).save(b, "PNG")
        return base64.b64encode(b.getvalue()).decode()

    def test_submit_list_load_process(self):
        iid = vr_service.submit({"device": "sim", "image_b64": "data:image/png;base64," + self._png_b64()}, self.s.tmp)
        vr_service.submit({"title": "Dune", "author": "Frank Herbert"}, self.s.tmp)
        pend = vr_service.list_pending(self.s.tmp)
        self.assertEqual(len(pend), 2)
        item = vr_service.load_item(iid, self.s.tmp)
        self.assertIsNotNone(vr_service.image_of(item))
        vr_service.mark_processed(iid, self.s.tmp)
        self.assertEqual(len(vr_service.list_pending(self.s.tmp)), 1)

    def test_invalid_payloads(self):
        for bad in ({}, {"device": "x"}, {"image_b64": "!!!notbase64!!!"}, {"image_b64": base64.b64encode(b"not an image").decode()}, "str"):
            with self.assertRaises(vr_service.VRPayloadError):
                vr_service.submit(bad, self.s.tmp)
        with self.assertRaises(KeyError):
            vr_service.load_item("nope", self.s.tmp)

    def test_http_bridge(self):
        import threading
        import urllib.error
        import urllib.request
        from http.server import ThreadingHTTPServer
        from unittest.mock import patch
        import config
        from pathlib import Path
        from vr import bridge_server
        with patch.object(config, "VR_INBOX_DIR", Path(self.s.tmp)):
            srv = ThreadingHTTPServer(("127.0.0.1", 0), bridge_server.Handler)
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            base = f"http://127.0.0.1:{srv.server_port}"
            try:
                req = urllib.request.Request(base + "/api/vr/submit", json.dumps({"title": "Dune"}).encode(),
                                             {"Content-Type": "application/json"})
                self.assertEqual(urllib.request.urlopen(req).status, 201)
                with self.assertRaises(urllib.error.HTTPError) as cm:
                    urllib.request.urlopen(urllib.request.Request(base + "/api/vr/submit", b"{}", {"Content-Type": "application/json"}))
                self.assertEqual(cm.exception.code, 400)
                self.assertEqual(urllib.request.urlopen(base + "/health").status, 200)
            finally:
                srv.shutdown()
            self.assertEqual(len(vr_service.list_pending()), 1)


if __name__ == "__main__":
    unittest.main()
