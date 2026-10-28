"""Reusable UI pieces shared by several pages."""
import pandas as pd
import streamlit as st

import config
from services import ocr_service
from services.text_utils import confidence_label, is_blank


def flash(msg: str, kind: str = "success"):
    st.session_state["_flash"] = (kind, msg)


def show_flash():
    f = st.session_state.pop("_flash", None)
    if f:
        getattr(st, f[0])(f[1])


def ocr_banner():
    if not ocr_service.is_available():
        st.warning("OCR engine not found, so automatic text extraction is disabled. Install Tesseract "
                   "(see README > Setup) or add books manually / via Excel. Everything else still works.")


def empty_catalogue_notice():
    st.info("The catalogue is empty. Add books via **Scan Books**, **Excel Import**, **Book Catalogue**, "
            "or load the demo data from the **Home** page.")


def book_card(row, score_label: str = None, extra: str = None):
    with st.container(border=True):
        st.markdown(f"**{row['title']}**  \n{row.get('author') or 'Unknown author'}  ·  *{row.get('genre', 'Unknown')}*")
        desc = row.get("description")
        if not is_blank(desc):
            st.caption(desc)
        if score_label:
            st.markdown(score_label)
        if extra:
            st.caption(extra)


REVIEW_COLS = ["save", "title", "author", "genre", "subgenre", "isbn", "publication_year", "description",
               "scan_confidence", "confidence", "source", "in_catalogue"]


def records_frame(svc, records) -> pd.DataFrame:
    df = pd.DataFrame(records)
    for c in REVIEW_COLS:
        if c not in df.columns:
            df[c] = None if c in ("scan_confidence", "save", "in_catalogue") else ""
    df["publication_year"] = pd.to_numeric(df["publication_year"], errors="coerce")
    df["scan_confidence"] = pd.to_numeric(df["scan_confidence"], errors="coerce")
    df["in_catalogue"] = [svc.catalogue.find_duplicate(r.title, r.author, r.isbn) is not None
                          for r in df.itertuples()]
    df["confidence"] = df["scan_confidence"].map(confidence_label)
    df["save"] = ~df["in_catalogue"]
    return df


def review_and_save(svc, records, input_type: str, filename: str, key: str):
    """Editable review table -> save to the shared catalogue (duplicates are merged-empty-only, never overwritten)."""
    if not records:
        return
    st.subheader("Review detected books")
    st.caption("Edit any cell, tick the books to save, or add rows (tick Save for new rows). Low confidence means the "
               "text was hard to read - please verify. 'in_catalogue' marks books that already exist (only their "
               "blank fields will be filled).")
    df = records_frame(svc, records)
    edited = st.data_editor(
        df[REVIEW_COLS], key=f"editor_{key}", num_rows="dynamic", hide_index=True,
        disabled=["scan_confidence", "confidence", "source", "in_catalogue"],
        column_config={
            "save": st.column_config.CheckboxColumn("Save?"),
            "publication_year": st.column_config.NumberColumn("Year", format="%d"),
            "scan_confidence": st.column_config.ProgressColumn("Scan conf.", min_value=0.0, max_value=1.0),
        })
    if st.button("Save selected to catalogue", key=f"save_{key}", type="primary"):
        sel = edited[edited["save"] == True]  # noqa: E712
        recs = []
        for r in sel.to_dict("records"):
            r["source"] = r.get("source") if not is_blank(r.get("source")) else input_type
            recs.append(r)
        counts = svc.catalogue.add_many(recs)
        svc.catalogue.log_scan(input_type, filename, len(edited), counts["added"] + counts["merged"], notes=str(counts))
        st.session_state.pop(key, None)
        flash(f"Saved: {counts['added']} added, {counts['merged']} existing book(s) completed, "
              f"{counts['duplicate']} duplicate(s) skipped, {counts['invalid']} invalid row(s) ignored.")
        st.rerun()


def genre_options(svc, extra=()) -> list:
    g = set(config.KNOWN_GENRES) | set(svc.catalogue.list_books()["genre"]) | set(extra)
    return sorted(x for x in g if x and x != "Unknown")
