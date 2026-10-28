import base64
import io

import cv2
import streamlit as st

import config
from services import vr_service
from services.image_scanner import InvalidImageError, load_image
from services.video_scanner import (InvalidVideoError, get_frame_at, probe_video, save_upload_to_temp, scan_frames,
                                    scan_video)
from ui.components import ocr_banner, review_and_save, show_flash


def render(svc):
    st.title("🎥 Video, Webcam & VR Scanner")
    show_flash()
    ocr_banner()
    t_vid, t_cam, t_vr = st.tabs(["Upload video", "Webcam capture", "VR input (optional)"])
    with t_vid:
        _video_tab(svc)
    with t_cam:
        _webcam_tab(svc)
    with t_vr:
        _vr_tab(svc)
    if "video_records" in st.session_state:
        review_and_save(svc, st.session_state["video_records"], "video_scan",
                        st.session_state.get("video_name", ""), "video_records")
    if "vr_records" in st.session_state:
        review_and_save(svc, st.session_state["vr_records"], "vr_input", "vr_inbox", "vr_records")


def _video_tab(svc):
    f = st.file_uploader("Upload a video panning across books",
                         type=[e.lstrip(".") for e in sorted(config.SUPPORTED_VIDEO_EXT)], key="video_file")
    c = st.columns(3)
    interval = c[0].slider("Sample every (seconds)", 0.5, 5.0, 1.0, 0.5)
    max_frames = c[1].slider("Max frames", 5, 60, 20)
    diff = c[2].slider("Skip-similar threshold", 1.0, 20.0, 6.0, help="Higher = skip more near-identical frames.")
    if not f:
        return
    sig = f"{f.name}-{f.size}"
    if st.session_state.get("video_sig") != sig:
        try:
            st.session_state["video_path"] = save_upload_to_temp(f.name, f.getvalue())
            st.session_state["video_info"] = probe_video(st.session_state["video_path"])
            st.session_state["video_sig"] = sig
        except InvalidVideoError as exc:
            st.error(str(exc))
            return
    info = st.session_state["video_info"]
    st.caption(f"{info['duration']:.1f}s, {info['fps']:.0f} fps, {info['frames']} frames")
    if st.button("Scan video", type="primary"):
        bar = st.progress(0.0)
        rep = scan_video(st.session_state["video_path"], svc.scanner, interval, max_frames, diff_threshold=diff,
                         progress=bar.progress)
        st.session_state["video_records"], st.session_state["video_name"] = rep.records, f.name
        st.info(f"Frames read: {rep.frames_read}, processed: {rep.frames_processed}, "
                f"skipped as near-identical: {rep.frames_skipped_similar}.")
        for w in rep.warnings:
            st.warning(w)
    with st.expander("Fallback: pick a frame manually"):
        t = st.slider("Time (s)", 0.0, max(float(info["duration"]), 0.1), 0.0, 0.1)
        try:
            frame = get_frame_at(st.session_state["video_path"], t)
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if st.button("Scan this frame"):
                rep = svc.scanner.scan(frame)
                st.session_state["video_records"] = rep.records
                st.session_state["video_name"] = f"{f.name}@{t:.1f}s"
                for w in rep.warnings:
                    st.warning(w)
                st.rerun()
        except InvalidVideoError as exc:
            st.warning(str(exc))


def _webcam_tab(svc):
    st.caption("Uses your browser's camera. Capture several angles/shelves, then scan the batch; "
               "near-identical captures are skipped and duplicate books merged.")
    shot = st.camera_input("Take a photo", key="cam")
    batch = st.session_state.setdefault("cam_batch", [])
    c = st.columns(3)
    if c[0].button("Add capture to batch", disabled=shot is None):
        batch.append(shot.getvalue())
    if c[1].button("Clear batch", disabled=not batch):
        batch.clear()
        st.rerun()
    c[2].metric("Captures in batch", len(batch))
    if st.button("Scan batch", type="primary", disabled=not batch):
        frames = []
        for i, b in enumerate(batch):
            try:
                frames.append((float(i), load_image(io.BytesIO(b))))
            except InvalidImageError as exc:
                st.warning(str(exc))
        rep = scan_frames(frames, svc.scanner)
        st.session_state["video_records"], st.session_state["video_name"] = rep.records, "webcam batch"
        st.rerun()


def _vr_tab(svc):
    st.markdown("**Optional extension.** A VR/AR client sends captures to a small HTTP bridge "
                "(`python -m vr.bridge_server`); they land in an inbox processed here by the same pipeline. "
                "Python cannot read a Quest headset camera directly - see README > VR for real-device steps.")
    st.markdown("##### Simulated headset capture")
    sim = st.file_uploader("Pretend this image came from the headset", type=["png", "jpg", "jpeg"], key="vr_sim")
    if st.button("Send to VR inbox (simulated)", disabled=sim is None):
        try:
            vr_service.submit({"device": "simulated-quest", "image_b64": base64.b64encode(sim.getvalue()).decode()})
            st.success("Capture stored in the VR inbox.")
        except vr_service.VRPayloadError as exc:
            st.error(str(exc))
    with st.expander("Or simulate a headset that already recognised a book (typed/voice)"):
        c = st.columns(3)
        t, a, i = c[0].text_input("Title", key="vr_t"), c[1].text_input("Author", key="vr_a"), c[2].text_input("ISBN", key="vr_i")
        if st.button("Send typed book to inbox"):
            try:
                vr_service.submit({"device": "simulated-quest", "title": t, "author": a, "isbn": i})
                st.success("Stored.")
            except vr_service.VRPayloadError as exc:
                st.error(str(exc))
    pending = vr_service.list_pending()
    st.markdown(f"##### Inbox ({len(pending)} pending)")
    if not pending:
        st.caption("Nothing waiting. Start the bridge or use the simulator above.")
        return
    st.dataframe(pending, hide_index=True)
    sel = st.selectbox("Item to process", [p["id"] for p in pending])
    if st.button("Process selected item"):
        item = vr_service.load_item(sel)
        recs = []
        try:
            img = vr_service.image_of(item)
            if img is not None:
                recs = svc.scanner.scan(img).records
            if not recs and (item.get("title") or item.get("isbn")):
                parsed = {"title": item["title"], "author": item["author"], "isbn": item["isbn"],
                          "raw_text": f"{item['title']} {item['author']}", "ocr_conf": 1.0}
                recs = [svc.scanner.build_record(parsed)]
        except InvalidImageError as exc:
            st.error(str(exc))
        vr_service.mark_processed(sel)
        st.session_state["vr_records"] = recs
        st.rerun()
