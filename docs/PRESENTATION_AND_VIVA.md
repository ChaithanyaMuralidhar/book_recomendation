# Presentation outline (12–14 slides, ~12 min)
1. Title, team, guide
2. Problem: readers can’t see which shelf books fit them; catalogues are manual and siloed
3. Objectives (multimodal capture → one catalogue → explainable recommendations)
4. Existing systems & gaps (rating-only recommenders, no shelf photo input, black boxes)
5. Proposed system & architecture diagram
6. Input pipeline: image → regions → OCR → parse → metadata → review → save (show screenshot)
7. Video/webcam: frame sampling, near-duplicate skipping, cross-frame de-duplication
8. Data model: books / users / reading_history / recommendation_log / scan_logs; Excel 4-sheet workbook
9. ML: TF-IDF + genre one-hot, K-Means (silhouette-chosen k), user profile
10. Hybrid scoring formula + explanation example
11. Library/store photo mode: recognised → matched → recommended
12. Testing: 64 tests, what they cover, honest evaluation caveat
13. Limitations & future work (trained book detector, larger real dataset, VR hardware)
14. Conclusion + live demo order: Home→load demo→Scan cover→Recommendations→Library photo

# Likely viva questions
**Q1. Why Tesseract and not a deep OCR model?** CPU-only, small install, good on printed text; EasyOCR/PaddleOCR need PyTorch/Paddle and are heavier. The OCR sits behind one function (`read_best`) so it can be swapped.
**Q2. How are books found in a shelf photo?** Classical CV: Sobel vertical edges + morphological opening → column profile → peaks split spines; for covers, Canny + contours + non-maximum suppression. It is a heuristic; a trained detector (YOLO-family) would be the upgrade. Failures are surfaced as warnings and a manual crop/edit fallback.
**Q3. How is the title chosen from OCR text?** Tesseract returns word boxes; the tallest-font line group is taken as the title, a “by …” or name-shaped line as the author, and checksum-valid ISBNs are extracted by regex. Then it is matched to known metadata.
**Q4. How do you avoid duplicates?** Normalised title (lower-case, no punctuation/leading article) + author surname, or identical valid ISBN. Merge mode fills only blank fields, so user data is never overwritten.
**Q5. Why is a scanned book not “read”?** Scanning means “seen/available”; reading is a user fact stored in `reading_history`. Mixing them would corrupt preferences.
**Q6. Explain the recommendation score.** Weighted sum of content similarity (TF-IDF cosine to the user’s weighted taste vector), genre preference, cluster affinity, author affinity and optional collaborative score; weights are user-adjustable and renormalised. Each result states its top two reasons.
**Q7. Why K-Means and how is k chosen?** Simple, fast, explainable on TF-IDF vectors; k∈[2,min(8,n−1)] by best silhouette score; tiny/degenerate data falls back to one cluster. Limitation: spherical clusters, sparse short texts; clusters ≠ user interest.
**Q8. When is collaborative filtering used?** Only with ≥5 users having ≥3 signals and overlapping liked books; otherwise skipped and the UI says why (cold-start/sparsity).
**Q9. How do you handle a new user?** Onboarding genres/authors act as pseudo-signals; with nothing, one starter pick per genre.
**Q10. How did you evaluate?** Unit/integration tests for correctness. For quality, a leave-one-out hit-rate/MRR is implemented but run only on 5 synthetic demo users, so it is a smoke test, **not** a valid result; a proper study needs real interaction logs, many users, a held-out split and baselines (popularity/random).
**Q11. What does the VR part really do?** A bridge/inbox so a headset client can submit images or recognised text; a simulator for testing. Python cannot access a Quest camera directly; real passthrough capture needs a native/WebXR client and Meta permissions — not tested without hardware.
**Q12. Privacy?** Everything is local (SQLite); online lookup sends only title/author/ISBN to Open Library and can be turned off.
**Q13. Biggest limitations?** Real-world shelf photos (glare, curved spines, tiny text), small demo dataset, heuristic detection, metadata coverage offline.
**Q14. Future work?** Trained spine/book detector, ISBN barcode decoding, learned embeddings (sentence transformers), implicit-feedback models, mobile capture, library-catalogue API integration.
