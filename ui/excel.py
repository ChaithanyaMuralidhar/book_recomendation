import streamlit as st

import config
from services import excel_service as xs
from ui.components import flash, records_frame, show_flash

XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def render(svc):
    st.title("📊 Excel Import & Export")
    show_flash()
    t1, t2, t3, t4 = st.tabs(["Import books", "Import reading history", "Export workbook", "Template"])
    with t1:
        _import_books(svc)
    with t2:
        _import_history(svc)
    with t3:
        _export(svc)
    with t4:
        st.write("Download a ready-to-fill workbook with the expected columns "
                 "(`title` is required; everything else is optional).")
        st.download_button("Download Excel template", xs.build_template(), "book_template.xlsx", XLSX)


def _import_books(svc):
    f = st.file_uploader("Excel (.xlsx) or CSV with book details", type=["xlsx", "xlsm", "csv"], key="xl_books")
    if not f:
        return
    try:
        df, issues = xs.read_books_file(f, f.name)
    except xs.ExcelFormatError as exc:
        st.error(str(exc))
        return
    for i in issues:
        st.warning(i)
    if df.empty:
        st.info("The file contains no importable rows.")
        return
    prev = records_frame(svc, df.to_dict("records"))
    cols = ["save", "title", "author", "genre", "subgenre", "isbn", "publication_year", "description", "in_catalogue"]
    edited = st.data_editor(prev[cols], key="xl_editor", hide_index=True, disabled=["in_catalogue"],
                            column_config={"publication_year": st.column_config.NumberColumn("Year", format="%d")})
    st.caption("Existing books are never overwritten: only their blank fields are completed.")
    if st.button("Import selected rows", type="primary"):
        sel = edited[edited["save"] == True].to_dict("records")  # noqa: E712
        for r in sel:
            r["source"] = "excel_import"
        counts = svc.catalogue.add_many(sel)
        svc.catalogue.log_scan("excel_import", f.name, len(edited), counts["added"] + counts["merged"], str(counts))
        flash(f"Import finished: {counts}")
        st.rerun()


def _import_history(svc):
    f = st.file_uploader("Reading history (.xlsx/.csv): user_id, book_id or title, status, rating, date",
                         type=["xlsx", "xlsm", "csv"], key="xl_hist")
    if not f:
        return
    try:
        df, _ = xs.read_history_file(f, f.name)
    except xs.ExcelFormatError as exc:
        st.error(str(exc))
        return
    st.dataframe(df, hide_index=True)
    auto = st.checkbox("Create users that do not exist yet", value=True)
    if st.button("Import history", type="primary"):
        ok = skipped = 0
        known = set(svc.users.list_users()["user_id"])
        problems = []
        for r in df.to_dict("records"):
            uid = r["user_id"]
            if uid not in known:
                if not auto:
                    skipped += 1
                    continue
                svc.users.create_user(r.get("user_name") or uid, user_id=uid)
                known.add(uid)
            bid = r["book_id"] if r["book_id"] and svc.catalogue.get_book(r["book_id"]) else \
                svc.catalogue.find_duplicate(r["title"], "", "")
            if not bid:
                problems.append(r["title"] or r["book_id"])
                continue
            try:
                svc.users.set_status(uid, bid, r["status"], r["rating"] or None,
                                     int(r["liked"]) if r["liked"] in ("1", "-1", "0") else None, r["read_date"] or None)
                ok += 1
            except (ValueError, KeyError) as exc:
                problems.append(f"{r['title'] or r['book_id']}: {exc}")
        msg = f"Imported {ok} history row(s); {skipped} skipped."
        if problems:
            msg += f" Not imported (book not in catalogue or invalid): {', '.join(map(str, problems[:10]))}"
        flash(msg, "success" if ok else "warning")
        st.rerun()


def _export(svc):
    recs = st.session_state.get("rec_table")
    if recs is None or len(recs) == 0:
        log = svc.users.recommendation_log()
        recs = log if len(log) else None
    data = xs.export_workbook(svc.catalogue.list_books(), svc.users.get_history(), recs, svc.catalogue.list_scan_logs())
    c = st.columns(2)
    c[0].download_button("⬇ Download workbook (4 sheets)", data, "book_recommendation_data.xlsx", XLSX, type="primary")
    if c[1].button("Save a copy to data/book_catalogue.xlsx"):
        config.EXPORT_XLSX.write_bytes(data)
        st.success(f"Saved to {config.EXPORT_XLSX.relative_to(config.BASE_DIR)}")
    st.caption("Sheets: Book Catalogue, Reading History, Recommendations (latest generated, else the log), Scan Logs.")
