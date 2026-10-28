import streamlit as st

from services.sample_data import load_sample
from ui.components import flash, show_flash


def render(svc):
    st.title("📚 Multimodal Intelligent Book Recommendation System")
    show_flash()
    books = svc.catalogue.list_books()
    users = svc.users.list_users()
    hist = svc.users.get_history()
    logs = svc.catalogue.list_scan_logs()
    c = st.columns(4)
    c[0].metric("Books in catalogue", len(books))
    c[1].metric("Users", len(users))
    c[2].metric("Reading-history entries", len(hist))
    c[3].metric("Scan sessions", len(logs))

    if books.empty:
        st.info("Welcome! The catalogue is empty. Load the bundled demo data to explore every feature, "
                "or start by scanning / importing your own books.")
        if st.button("Load demo dataset (52 books, 5 users)", type="primary"):
            r = load_sample(svc.catalogue, svc.users)
            flash(f"Demo data loaded: {r['added']} books, {r['users']} users, {r['history_rows']} history entries.")
            st.rerun()
    else:
        left, right = st.columns(2)
        with left:
            import plotly.express as px
            gc = books["genre"].value_counts().rename_axis("genre").reset_index(name="books")
            st.subheader("Catalogue by genre")
            st.plotly_chart(px.bar(gc, x="genre", y="books"))
        with right:
            st.subheader("Recent scan sessions")
            if logs.empty:
                st.caption("No scans yet - try **Scan Books**.")
            else:
                st.dataframe(logs.head(8), hide_index=True)
            st.subheader("How the pieces fit together")
            st.caption("Every input method (image, video/webcam, Excel, manual, VR inbox) writes to the same "
                       "catalogue. Reading history is stored separately - scanning a book never marks it as read. "
                       "Recommendations combine content similarity, your genre preferences, K-Means theme groups "
                       "and (when enough data exists) collaborative filtering.")
    st.divider()
    st.markdown("**Quick start:** 1) add books  →  2) pick a user & add reading history  →  3) open "
                "**AI Recommendations**  →  4) try **Scan a Library/Store** with a photo of a shelf.")
