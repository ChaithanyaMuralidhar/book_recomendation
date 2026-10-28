import datetime as dt

import streamlit as st

import config
from ui.components import flash, genre_options, show_flash
from ui.state import current_user


def render(svc):
    st.title("👤 User Profile & Reading History")
    show_flash()
    st.caption("Scanned books are **not** treated as read. Only the entries you add here (or import) shape your recommendations.")
    t_hist, t_add, t_pref, t_new = st.tabs(["Reading history", "Add entry", "Preferences", "Create user"])
    user = current_user(svc)
    with t_new:
        _create(svc)
    if not user:
        with t_hist:
            st.info("No users yet - create one in the **Create user** tab (or load the demo data on the Home page).")
        return
    uid = user["user_id"]
    with t_hist:
        st.subheader(f"{user['name']} ({uid})")
        h = svc.users.get_history(uid)
        if h.empty:
            st.info("No reading history yet. Use **Add entry**, import from Excel, or set preferences for new-user recommendations.")
        else:
            st.dataframe(h.drop(columns=["user_id"]), hide_index=True)
            freq = svc.users.genre_frequency(uid)
            if len(freq):
                import plotly.express as px
                st.markdown("**Books read by genre**")
                st.plotly_chart(px.bar(freq.rename_axis("genre").reset_index(name="books read"), x="genre", y="books read"))
            lab = {r.book_id: f"{r.title} ({r.status})" for r in h.itertuples()}
            rm = st.selectbox("Remove an entry", list(lab), format_func=lab.get)
            if st.button("Remove from history"):
                svc.users.remove_history(uid, rm)
                flash("Entry removed.")
                st.rerun()
        log = svc.users.recommendation_log(uid)
        if len(log):
            with st.expander("Previously recommended books"):
                st.dataframe(log.drop(columns=["user_id"]), hide_index=True)
    with t_add:
        books = svc.catalogue.list_books()
        if books.empty:
            st.info("Add books to the catalogue first.")
        else:
            labels = {r.book_id: f"{r.title} - {r.author}" for r in books.itertuples()}
            with st.form("hist_form"):
                bid = st.selectbox("Book", list(labels), format_func=labels.get)
                status = st.radio("Status", config.HISTORY_STATUSES, horizontal=True)
                rating = st.selectbox("Rating (read books)", [0, 1, 2, 3, 4, 5],
                                      format_func=lambda x: "no rating" if x == 0 else "★" * x)
                liked = st.radio("Opinion", ["no opinion", "liked", "disliked"], horizontal=True)
                date = st.date_input("Date", dt.date.today())
                if st.form_submit_button("Save entry", type="primary"):
                    svc.users.set_status(uid, bid, status, rating or None, {"liked": 1, "disliked": -1}.get(liked),
                                         date.isoformat() if status == "read" else None)
                    flash("History updated.")
                    st.rerun()
    with t_pref:
        st.write("Used for new users and as a light prior for everyone.")
        gopts = genre_options(svc, user["pref_genres"])
        g = st.multiselect("Preferred genres", gopts, default=[x for x in user["pref_genres"] if x in gopts])
        a = st.text_input("Favourite authors (comma-separated)", ", ".join(user["pref_authors"]))
        if st.button("Save preferences"):
            svc.users.update_preferences(uid, g, [x.strip() for x in a.split(",") if x.strip()])
            flash("Preferences saved.")
            st.rerun()


def _create(svc):
    with st.form("new_user"):
        name = st.text_input("Name")
        g = st.multiselect("Onboarding: genres you enjoy", genre_options(svc))
        a = st.text_input("Favourite authors (comma-separated)")
        if st.form_submit_button("Create user", type="primary"):
            try:
                uid = svc.users.create_user(name, g, [x.strip() for x in a.split(",") if x.strip()])
                st.session_state["_select_user"] = uid  # applied by app.py before the sidebar widget is built
                flash(f"User {uid} created.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
