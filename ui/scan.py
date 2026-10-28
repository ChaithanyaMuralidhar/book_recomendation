import streamlit as st

import config
from services.image_scanner import InvalidImageError, load_image
from ui.components import ocr_banner, review_and_save, show_flash

KEY = "scan_records"
MODES = {"auto": "Auto (shelf or separate covers)", "single": "Single book / cover", "shelf": "Shelf of spines"}


def render(svc):
    st.title("📷 Scan Books from Images")
    show_flash()
    ocr_banner()
    files = st.file_uploader("Upload photo(s) of book covers, spines or shelves",
                             type=[e.lstrip(".") for e in sorted(config.SUPPORTED_IMAGE_EXT)],
                             accept_multiple_files=True, key="scan_files")
    c1, c2 = st.columns([2, 3])
    mode = c1.selectbox("Detection mode", list(MODES), format_func=MODES.get,
                        help="Use 'single' for one cover, 'shelf' for spines on a shelf.")
    with c2.expander("Optional: crop before scanning (fallback if detection fails)"):
        cl, cr = st.slider("Left / right (%)", 0, 100, (0, 100))
        ct, cb = st.slider("Top / bottom (%)", 0, 100, (0, 100))
    crop = (cl / 100, cr / 100, ct / 100, cb / 100) if (cl, cr, ct, cb) != (0, 100, 0, 100) else None
    st.caption("Region detection is a classical computer-vision heuristic. Overlapping books, glare and unreadable "
               "spines will not be identified - correct or add them manually below.")

    if st.button("Scan images", type="primary", disabled=not files):
        records, shots, warns = [], [], []
        bar = st.progress(0.0)
        for i, f in enumerate(files):
            try:
                img = load_image(f)
            except InvalidImageError as exc:
                warns.append(f"{f.name}: {exc}")
                continue
            rep = svc.scanner.scan(img, mode=mode, crop=crop)
            for r in rep.records:
                r["file"] = f.name
            records += rep.records
            shots.append((f.name, rep.annotated))
            warns += [f"{f.name}: {w}" for w in rep.warnings]
            bar.progress((i + 1) / len(files))
        st.session_state[KEY], st.session_state["scan_shots"] = records, shots
        st.session_state["scan_warns"], st.session_state["scan_names"] = warns, ", ".join(f.name for f in files)

    for w in st.session_state.get("scan_warns", []):
        st.warning(w)
    if st.session_state.get("scan_shots"):
        with st.expander("Detected regions", expanded=False):
            for name, arr in st.session_state["scan_shots"]:
                st.image(arr, caption=name)
    if KEY in st.session_state:
        if not st.session_state[KEY]:
            st.info("No books identified. Try another mode, crop closer, or use manual entry in **Book Catalogue**.")
        review_and_save(svc, st.session_state[KEY], "image_scan", st.session_state.get("scan_names", ""), KEY)
