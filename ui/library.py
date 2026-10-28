import pandas as pd
import streamlit as st

import config
from services.image_scanner import InvalidImageError, load_image
from services.recommendation_service import Recommender
from ui.components import book_card, empty_catalogue_notice, flash, ocr_banner, show_flash
from ui.state import current_user


def render(svc):
    st.title("🏛️ Scan a Library / Store")
    show_flash()
    ocr_banner()
    st.caption("Photograph a shelf. The app (1) recognises what it can read, (2) matches it to the catalogue, "
               "(3) recommends the best matches for the selected user. These are three different lists.")
    f = st.file_uploader("Photo of the shelf / display", type=[e.lstrip(".") for e in sorted(config.SUPPORTED_IMAGE_EXT)],
                         key="lib_file")
    c = st.columns(3)
    mode = c[0].selectbox("Detection mode", ["auto", "shelf", "single"])
    top_n = c[1].slider("Recommend up to", 1, 15, 5)
    only = c[2].checkbox("Only recommend books in this photo", True)
    if st.button("Scan photo", type="primary", disabled=f is None):
        try:
            rep = svc.scanner.scan(load_image(f), mode=mode)
        except InvalidImageError as exc:
            st.error(str(exc))
            return
        st.session_state["lib_records"] = svc.catalogue.match_records(rep.records)
        st.session_state["lib_img"], st.session_state["lib_warn"] = rep.annotated, rep.warnings
    recs = st.session_state.get("lib_records")
    if recs is None:
        return
    for w in st.session_state.get("lib_warn", []):
        st.warning(w)
    if st.session_state.get("lib_img") is not None:
        with st.expander("Detected regions"):
            st.image(st.session_state["lib_img"])

    st.subheader("1 · Recognised in the photo")
    if not recs:
        st.info("Nothing could be read. Try a closer, sharper photo or a different mode.")
        return
    df = pd.DataFrame(recs)[["title", "author", "scan_confidence", "match_status", "match_score", "matched_book_id"]]
    edited = st.data_editor(df, key="lib_editor", hide_index=True, num_rows="dynamic",
                            disabled=["scan_confidence", "match_status", "match_score", "matched_book_id"])
    c = st.columns(2)
    if c[0].button("Re-match after my edits"):
        merged = []
        for i, row in enumerate(edited.to_dict("records")):
            base = dict(recs[i]) if i < len(recs) else {}
            base["title"] = "" if pd.isna(row.get("title")) else str(row["title"])
            base["author"] = "" if pd.isna(row.get("author")) else str(row["author"])
            merged.append(base)
        st.session_state["lib_records"] = svc.catalogue.match_records(merged)
        st.rerun()
    unmatched = [r for r in recs if r["match_status"] != "matched" and r.get("title")]
    if c[1].button(f"Add {len(unmatched)} unmatched/uncertain book(s) to catalogue", disabled=not unmatched):
        counts = svc.catalogue.add_many([{**r, "source": r.get("source") or "library_scan"} for r in unmatched])
        st.session_state["lib_records"] = svc.catalogue.match_records(recs)
        flash(f"Catalogue updated: {counts}")
        st.rerun()

    st.subheader("2 · Matched to the catalogue")
    matched = [r for r in recs if r["match_status"] == "matched"]
    uncertain = [r for r in recs if r["match_status"] == "uncertain"]
    st.write(f"{len(matched)} matched, {len(uncertain)} uncertain (score 0.65-0.80 - correct the title and re-match), "
             f"{len(recs) - len(matched) - len(uncertain)} not in the catalogue.")
    if matched:
        st.dataframe(pd.DataFrame(matched)[["title", "author", "matched_book_id", "match_score"]], hide_index=True)

    st.subheader("3 · Recommended for you")
    user = current_user(svc)
    if not user:
        st.info("Select or create a user to get personalised picks.")
        return
    books = svc.catalogue.list_books()
    if books.empty:
        empty_catalogue_notice()
        return
    ids = {r["matched_book_id"] for r in matched}
    if only and not ids:
        st.warning("No recognised book is in the catalogue yet, so nothing can be recommended from this photo. "
                   "Add the unmatched books (button above) or correct titles.")
        return
    res = Recommender(books).recommend(svc.users.get_history(user["user_id"]), user, top_n,
                                       ids if only else None, True, None, None, svc.users.get_history())
    for n in res.notes:
        st.info(n)
    for r in res.table.to_dict("records"):
        book_card(r, f"**Match score:** {r['score']:.2f}",
                  f"Why: {r['explanation']}  ·  " + ("Recognised in your photo (verify on the shelf)" if r["book_id"] in ids
                                                     else "Not seen in the photo - availability unknown"))
