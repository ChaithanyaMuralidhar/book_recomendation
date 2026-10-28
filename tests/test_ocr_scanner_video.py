import os
import unittest
from unittest import mock

import cv2
import numpy as np

from services import ocr_service, video_scanner as vs
from services.image_scanner import ImageScanner, InvalidImageError, detect_book_regions, load_image, parse_lines
from services.ocr_service import OCRLine, OCRUnavailableError
from tests.helpers import make_cover, make_services, to_bgr

OCR = ocr_service.is_available()


class ParsingAndDetectionTests(unittest.TestCase):
    def test_parse_lines_title_author_isbn(self):
        lines = [OCRLine("THE HOBBIT", 0.95, 80, 100, 10), OCRLine("by J. R. R. Tolkien", 0.9, 30, 400, 10),
                 OCRLine("ISBN 978-0-306-40615-7", 0.8, 15, 900, 10), OCRLine("@@", 0.2, 10, 950, 10)]
        p = parse_lines(lines)
        self.assertEqual((p["title"], p["author"], p["isbn"]), ("The Hobbit", "J. R. R. Tolkien", "9780306406157"))
        self.assertGreater(p["ocr_conf"], 0.9)

    def test_parse_lines_garbage_gives_empty(self):
        p = parse_lines([OCRLine("~~", 0.1, 10, 0, 0), OCRLine("a", 0.9, 10, 20, 0)])
        self.assertEqual((p["title"], p["isbn"]), ("", ""))
        self.assertEqual(parse_lines([])["title"], "")

    def test_invalid_image_inputs(self):
        for bad in (b"not an image", b"", b"\x89PNG broken"):
            with self.assertRaises(InvalidImageError):
                load_image(bad)

    def test_region_detection(self):
        img = np.full((600, 900, 3), 220, np.uint8)
        self.assertEqual(detect_book_regions(img, "single"), [(0, 0, 900, 600)])
        self.assertGreaterEqual(len(detect_book_regions(img, "auto")), 1)  # blank image: safe fallback
        shelf = np.full((600, 1200, 3), 200, np.uint8)
        x = 30
        for c in [(120, 40, 40), (30, 80, 130), (40, 110, 60), (130, 100, 30), (90, 40, 120)]:
            shelf[40:560, x:x + 180] = c
            x += 200
        self.assertGreaterEqual(len(detect_book_regions(shelf, "shelf")), 4)


class OCRFailureTests(unittest.TestCase):
    def setUp(self):
        self.s = make_services()
        self.addCleanup(self.s.cleanup)

    def test_ocr_unavailable_is_handled(self):
        with mock.patch.object(ocr_service, "_configure", side_effect=OCRUnavailableError("no tesseract")):
            rep = self.s.scanner.scan(np.full((300, 200, 3), 255, np.uint8))
        self.assertFalse(rep.ocr_available)
        self.assertEqual(rep.records, [])
        self.assertIn("no tesseract", rep.warnings[0])

    def test_ocr_engine_crash_per_region_is_a_warning(self):
        fake = mock.Mock()
        fake.read_best.side_effect = ocr_service.OCRError("boom")
        rep = ImageScanner(None, ocr=fake).scan(np.full((300, 200, 3), 255, np.uint8))
        self.assertEqual(rep.records, [])
        self.assertTrue(any("boom" in w for w in rep.warnings))

    @unittest.skipUnless(OCR, "Tesseract not installed")
    def test_blank_image_yields_no_books(self):
        rep = self.s.scanner.scan(np.full((500, 400, 3), 255, np.uint8))
        self.assertEqual(rep.records, [])
        self.assertTrue(rep.warnings)


@unittest.skipUnless(OCR, "Tesseract not installed")
class OCRPipelineTests(unittest.TestCase):
    def setUp(self):
        self.s = make_services()
        self.addCleanup(self.s.cleanup)

    def test_known_cover_is_identified_with_metadata(self):
        rep = self.s.scanner.scan(to_bgr(make_cover(["DUNE"], "Frank Herbert")))
        self.assertEqual(len(rep.records), 1)
        r = rep.records[0]
        self.assertEqual((r["title"], r["genre"]), ("Dune", "Science Fiction"))
        self.assertTrue(r["source"].startswith("image_scan+local"))
        self.assertGreater(r["scan_confidence"], 0.75)

    def test_unknown_cover_has_low_confidence_and_unknown_genre(self):
        r = self.s.scanner.scan(to_bgr(make_cover(["Quantum", "Gardening"], "Elena Brooks"))).records[0]
        self.assertEqual(r["genre"], "Unknown")
        self.assertLess(r["scan_confidence"], 0.75)  # never presented as certain
        self.assertEqual(r["source"], "image_scan")

    def test_video_pipeline_skips_similar_and_dedupes(self):
        a = cv2.resize(to_bgr(make_cover(["DUNE"], "Frank Herbert")), (480, 680))
        b = cv2.resize(to_bgr(make_cover(["THE", "HOBBIT"], "J. R. R. Tolkien")), (480, 680))
        path = os.path.join(self.s.tmp, "t.avi")
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (480, 680))
        for i in range(60):
            w.write(a if i < 30 else b)
        w.release()
        rep = vs.scan_video(path, self.s.scanner, interval_s=0.5, max_frames=20)
        self.assertEqual(sorted(r["title"] for r in rep.records), ["Dune", "The Hobbit"])
        self.assertGreater(rep.frames_skipped_similar, 0)
        self.assertLess(rep.frames_processed, rep.frames_read)


class VideoLogicTests(unittest.TestCase):
    def test_dedupe_keeps_best_and_counts(self):
        recs = [{"title": "Dune", "isbn": "", "scan_confidence": 0.5}, {"title": "dune", "isbn": "", "scan_confidence": 0.9},
                {"title": "Other Book", "isbn": "", "scan_confidence": 0.7}, {"title": "", "isbn": ""}]
        out = vs.dedupe_records(recs)
        self.assertEqual(len(out), 2)
        d = next(r for r in out if r["title"].lower() == "dune")
        self.assertEqual((d["sightings"], d["scan_confidence"]), (2, 0.9))

    def test_scan_frames_skips_identical_frames_without_ocr(self):
        frame = np.full((200, 200, 3), 128, np.uint8)
        scanner = mock.Mock()
        scanner.scan.return_value = mock.Mock(records=[], ocr_available=True, warnings=[])
        rep = vs.scan_frames([(0, frame), (1, frame.copy()), (2, frame.copy())], scanner)
        self.assertEqual((rep.frames_processed, rep.frames_skipped_similar), (1, 2))
        self.assertTrue(rep.warnings)  # tells the user to use the manual frame picker

    def test_unsupported_and_corrupt_video(self):
        with self.assertRaises(vs.InvalidVideoError):
            vs.probe_video("clip.gif")
        d = make_services()
        self.addCleanup(d.cleanup)
        bad = os.path.join(d.tmp, "bad.mp4")
        open(bad, "wb").write(b"junk")
        with self.assertRaises(vs.InvalidVideoError):
            vs.probe_video(bad)
        with self.assertRaises(vs.InvalidVideoError):
            vs.save_upload_to_temp("x.exe", b"1")


if __name__ == "__main__":
    unittest.main()
