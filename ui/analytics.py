import pandas as pd
import streamlit as st

from services.clustering_service import cluster_books
from services.recommendation_service import evaluate_leave_one_out
from ui.components import empty_catalogue_notice, show_flash
from ui.state import current_user


def render(svc):
    import plotly.express as px
    st.title("📈 ML Analytics & Clustering")
    show_flash()
    books = svc.catalogue.list_books()
    if books.empty:
        empty_catalogue_notice()
        return
    auto = st.checkbox("Choose number of clusters automatically (silhouette score)", True)
    k = None if auto else st.slider("Clusters (k)", 2, min(10, max(3, len(books))), 4)
    res = cluster_books(books, k)
    c = st.columns(3)
    c[0].metric("Books", len(books))
    c[1].metric("Clusters", res.k)
    c[2].metric("Silhouette", "n/a" if res.silhouette is None else f"{res.silhouette:.2f}")
    for n in res.notes:
        st.info(n)
    st.caption("Representation: TF-IDF of descriptions/subgenres + one-hot genre. Clusters group similar *books*; "
               "they do not by themselves predict what a reader will enjoy.")
    plot = res.assignments.merge(books[["book_id", "title", "genre"]], on="book_id")
    plot["cluster"] = plot["cluster"].astype(str)
    st.plotly_chart(px.scatter(plot, x="x", y="y", color="cluster", hover_name="title", hover_data=["genre"]))
    st.subheader("Cluster summaries")
    for s in res.summaries:
        with st.expander(f"Cluster {s['cluster']} · {s['size']} books · {s['label']}"):
            st.write(s["explanation"])
    st.subheader("Genre × cluster")
    st.dataframe(pd.crosstab(plot["genre"], plot["cluster"]))

    user = current_user(svc)
    if user:
        freq = svc.users.genre_frequency(user["user_id"])
        if len(freq):
            st.subheader(f"Reading profile of {user['name']}")
            st.plotly_chart(px.pie(freq.rename_axis("genre").reset_index(name="books"), names="genre", values="books"))

    with st.expander("Recommender self-check (leave-one-out) - read the caveat"):
        st.warning("This is a smoke test on demonstration data, **not** a valid experiment: the sample users are few "
                   "and synthetic, tastes are genre-clean, and no baselines are compared. A real evaluation needs "
                   "held-out interactions from real readers and baselines such as popularity or random.")
        if st.button("Run leave-one-out check"):
            out = evaluate_leave_one_out(books, svc.users.get_history(), svc.users.list_users())
            if not out["cases"]:
                st.info("Not enough reading history (need users with >= 3 liked read books).")
            else:
                st.json(out)
