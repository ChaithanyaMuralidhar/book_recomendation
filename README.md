# Multimodal Intelligent Book Recommendation System
*Computer Vision + OCR + Machine Learning — final-year AIML project (Python / Streamlit)*

One shared catalogue (SQLite) is fed by **images, video/webcam frames, Excel/CSV, manual entry and an optional VR inbox**.
An explainable hybrid recommender (content similarity + genre preference + K-Means theme groups + optional collaborative
filtering) ranks books for a user, including from a **photo of a library/store shelf**.

## Setup (Windows + VS Code, no GPU)
1. Install **Python 3.10+** and add it to PATH.
2. Install **Tesseract OCR** (the UB-Mannheim Windows installer). Default path
   `C:\Program Files\Tesseract-OCR\tesseract.exe` is auto-detected; otherwise set `TESSERACT_CMD`.
   Without it everything still works except automatic text extraction (use Excel/manual entry).
3. In the project folder:
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
Run the app: `streamlit run app.py`  ·  Run tests: `python -m unittest discover -s tests -t . -v` (pytest also works)

First launch: **Home → Load demo dataset** (52 real books, 5 users with reading histories), then try **AI Recommendations**.

## Pages
Home Dashboard · Scan Books (image) · Video/Webcam/VR · Excel Import & Export · Book Catalogue · Profile & Reading
History · AI Recommendations · Scan a Library/Store · ML Analytics & Clustering · Settings & Help.

## Layout
```
app.py  config.py  requirements.txt
services/  db, text_utils, catalogue_service, user_service, excel_service, metadata_service, ocr_service,
           image_scanner, video_scanner, clustering_service, recommendation_service, vr_service, sample_data
ui/        one module per page (+ components, state)      vr/  bridge_server.py, webxr_client.html
data/      sample_books.csv, sample_users.csv, sample_history.csv     tests/   unittest suite     docs/
```
(`ui/` is used instead of Streamlit's `pages/` folder because `pages/` would add an automatic second navigation menu.)

## Design decisions & assumptions
- **OCR = Tesseract** (light, CPU-only, simple install) rather than EasyOCR (pulls in PyTorch, ~GBs).
- **Book detection = classical CV** (edge/contour boxes; vertical-edge projection to split shelf spines), not a trained
  detector. A pretrained detector (e.g. a small YOLO model) could replace `detect_book_regions` later; it was not made
  mandatory to keep the install light. Overlapping books, glare, curved/tiny/stylised text will fail — results are always
  shown in an editable review table with a confidence value.
- **Metadata**: local catalogue + bundled sample first (offline), **Open Library** (no key) optional; online-derived rows are
  tagged `source=…+openlibrary`. After 2 network failures the app stays offline for the session. Nothing is fabricated.
- **SQLite** is the live store; Excel is import/export (4-sheet workbook). Imports **never overwrite**: duplicates only get
  *blank* fields filled; user edits happen explicitly in the Catalogue page.
- **Scanned ≠ read**: reading history is a separate table; the app never marks a scanned book as read.
- **Sample data**: real titles/authors/years with short *original* descriptions written for this project. ISBNs are left
  blank on purpose (not invented); ISBN scanning/validation works with real ISBNs (checksum-validated).
- Scan confidence = `0.4·OCR confidence + 0.6·match score` when a known book is matched (≥0.9 if a valid ISBN was read),
  else `0.6·OCR confidence`. Bands: High ≥ 0.75, Medium ≥ 0.5, else “Low (verify)”.

## Recommendation scoring (transparent)
`score = w·content + w·genre + w·cluster + w·author + w·collab` (weights adjustable in the UI, renormalised; defaults
0.35/0.30/0.10/0.10/0.15). Collaborative filtering is **only active when ≥5 users have ≥3 rated/read books and the target
user shares liked books with someone**; otherwise its weight is redistributed and the UI says why. History signal weights:
read 5★=1.0, 4★=0.8, 3★=0.5, 2★=0, 1★=−0.5, unrated read=1.0, reading=0.7, interested=0.5, liked/disliked=±1.
New users: onboarding genres/authors; with nothing at all, one starter pick per genre.

## VR / Meta Quest (optional, separate)
Not needed for the core app. Python **cannot** read a headset camera directly. Provided: an HTTP bridge
(`python -m vr.bridge_server --port 8765`) → inbox → processed in the **Video/Webcam/VR → VR input** tab by the same
pipeline; plus a simulator (upload an image as a “headset capture”) so it is testable without hardware.
Real-device path (**not tested here, verify against current Meta docs**): open `http://<pc>:8765/client` in Quest Browser;
camera APIs need a secure context (HTTPS, or `adb reverse tcp:8765 tcp:8765` and use `http://localhost:8765/client`).
Quest restricts raw passthrough-camera access for web content; a native/Unity app using Meta’s passthrough-camera API
(requires the appropriate headset OS version and runtime permission) can POST frames to `/api/vr/submit`.
Optional shared secret: set `VR_BRIDGE_TOKEN` and send `X-Bridge-Token`.

## Known limitations
- Heuristic region detection and OCR on real shelves are much harder than the synthetic images used in tests; expect to
  correct many rows. No claim of perfect identification of overlapping/unreadable spines is made.
- Streamlit UI was smoke-tested only through a stub harness in the build environment; `tests/test_ui_smoke.py` runs the real
  app with `streamlit.testing` on your machine (skipped if Streamlit is absent). Webcam capture, file-upload widgets and
  layout need a manual check in a browser.
- Demo data is tiny and synthetic in its user behaviour: recommendation quality figures on it are **not** evidence of
  accuracy (see docs/PROJECT_REPORT.md §Evaluation).
- Availability is unknown unless typed in or the book was seen in a photo (“recognised in photo — verify on the shelf”).
