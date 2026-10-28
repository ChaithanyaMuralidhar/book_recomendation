import streamlit as st

import config
from services import ocr_service
from services.sample_data import load_sample
from ui.components import flash, show_flash


def render(svc):
    st.title("⚙️ Settings & Help")
    show_flash()
    t1, t2, t3 = st.tabs(["Settings", "Help", "VR guide"])
    with t1:
        svc.metadata.online = st.checkbox("Allow online metadata lookup (Open Library)", svc.metadata.online,
                                          help="Off = fully offline using the catalogue + bundled sample data.")
        if svc.metadata.failures >= config.ONLINE_MAX_FAILURES:
            st.warning("Online lookup was switched off for this session after repeated network failures.")
            if st.button("Retry online"):
                svc.metadata.failures = 0
        st.write("OCR engine:", "✅ Tesseract found" if ocr_service.is_available() else "❌ not found (see README)")
        st.write("Database file:", f"`{config.DB_PATH}`")
        st.divider()
        if st.button("Load demo dataset"):
            flash(f"Demo data: {load_sample(svc.catalogue, svc.users)}")
            st.rerun()
        with st.expander("Danger zone"):
            ok = st.checkbox("Delete ALL books, users, history and logs")
            if st.button("Reset database", disabled=not ok):
                svc.db.reset()
                flash("Database cleared.")
                st.rerun()
    with t2:
        st.markdown("""
**Typical workflow:** load demo data or add books → create/select a user → add reading history →
**AI Recommendations**. Try **Scan Books** with a cover photo and review the table before saving.

**Good photos:** even light, no glare, text at least ~25 px tall, one cover or a straight-on shelf shot.

**Honest limits:** OCR + heuristic region detection can miss overlapping, curved, tiny or stylised text; every
result is editable and carries a confidence value. Metadata marked *openlibrary* was retrieved online and may be
imperfect; availability is only known if you enter it or the book was seen in a photo.
        """)
    with t3:
        st.markdown("""
1. Run the bridge on the PC: `python -m vr.bridge_server --port 8765`
2. Test without hardware: **Video/Webcam/VR → VR input** tab (simulated headset).
3. Real Quest: open the bridge's `/client` page in the Quest Browser (camera needs HTTPS or localhost - e.g.
   `adb reverse tcp:8765 tcp:8765` then browse to `http://localhost:8765/client`) and use the capture buttons.
4. Raw passthrough-camera frames are restricted on Quest; a native/Unity app using Meta's passthrough camera API
   could POST frames to `/api/vr/submit` instead. **Verify current Meta documentation and permissions**; this
   project does not claim headset access from Python.
        """)
