import streamlit as st

from services.text_utils import is_valid_isbn
from ui.components import empty_catalogue_notice, flash, genre_options, show_flash


def render(svc):
    st.title("📖 Book Catalogue")
    show_flash()
    books = svc.catalogue.list_books()
    t1, t2, t3 = st.tabs(["Browse", "Add / edit a book", "Delete"])
    with t1:
        if books.empty:
            empty_catalogue_notice()
        else:
            c = st.columns([2, 2, 2])
            q = c[0].text_input("Search title / author")
            gs = c[1].multiselect("Genre", sorted(books["genre"].unique()))
            srcs = c[2].multiselect("Source", sorted(books["source"].unique()))
            v = books
            if q:
                v = v[v["title"].str.contains(q, case=False, na=False, regex=False)
                      | v["author"].str.contains(q, case=False, na=False, regex=False)]
            if gs:
                v = v[v["genre"].isin(gs)]
            if srcs:
                v = v[v["source"].isin(srcs)]
            st.caption(f"{len(v)} of {len(books)} books. Metadata with source containing 'openlibrary' was retrieved "
                       "online; 'image_scan' values come from OCR and may need checking.")
            st.dataframe(v, hide_index=True)
    with t2:
        _form(svc, books)
    with t3:
        if books.empty:
            st.caption("Nothing to delete.")
        else:
            labels = {r.book_id: f"{r.title} - {r.author} ({r.book_id})" for r in books.itertuples()}
            bid = st.selectbox("Book", list(labels), format_func=labels.get, key="del_sel")
            ok = st.checkbox("I understand this also removes it from all reading histories")
            if st.button("Delete book", disabled=not ok):
                svc.catalogue.delete_book(bid)
                flash("Book deleted.")
                st.rerun()


def _form(svc, books):
    NEW = "➕ New book"
    labels = {NEW: NEW, **{r.book_id: f"{r.title} - {r.author} ({r.book_id})" for r in books.itertuples()}}
    sel = st.selectbox("Edit existing book or create new", list(labels), format_func=labels.get)
    cur = (svc.catalogue.get_book(sel) if sel != NEW else {}) or {}
    with st.form(f"book_form_{sel}"):
        c = st.columns(2)
        title = c[0].text_input("Title *", cur.get("title", ""))
        author = c[1].text_input("Author", cur.get("author", ""))
        gopts = genre_options(svc, [cur.get("genre", "")]) + ["Unknown"]
        genre = c[0].selectbox("Genre", gopts, index=gopts.index(cur["genre"]) if cur.get("genre") in gopts else len(gopts) - 1)
        sub = c[1].text_input("Subgenre", cur.get("subgenre", ""))
        isbn = c[0].text_input("ISBN (10 or 13 digits, validated)", cur.get("isbn", ""))
        year = c[1].number_input("Publication year (0 = unknown)", 0, 2100, int(cur.get("publication_year") or 0))
        lang = c[0].text_input("Language", cur.get("language", "English") if cur else "English")
        avail = c[1].text_input("Availability (only if you know it)", cur.get("availability", "Unknown"))
        desc = st.text_area("Description", cur.get("description", ""))
        go = st.form_submit_button("Save", type="primary")
    if go:
        rec = {"title": title, "author": author, "genre": genre, "subgenre": sub, "isbn": isbn,
               "publication_year": year or None, "language": lang, "availability": avail, "description": desc}
        try:
            if isbn and not is_valid_isbn(isbn):
                st.warning("The ISBN failed validation and was not stored.")
            if sel == NEW:
                rec["source"] = "manual"
                res = svc.catalogue.add_book(rec, on_duplicate="skip")
                flash("Book added." if res.status == "added" else
                      f"That book already exists as {res.book_id}; nothing was changed.",
                      "success" if res.status == "added" else "warning")
            else:
                svc.catalogue.update_book(sel, rec)
                flash("Book updated.")
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
