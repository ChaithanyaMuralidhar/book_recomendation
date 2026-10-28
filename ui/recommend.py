import streamlit as st

import config
from services.recommendation_service import Recommender
from ui.components import book_card, empty_catalogue_notice, flash, genre_options, show_flash
from ui.state import current_user


def render(svc):
    st.title("✨ AI Recommendations")
    show_flash()
    books = svc.catalogue.list_books()
    if books.empty:
        empty_catalogue_notice()
        return
    user = current_user(svc)
    if not user:
        st.info("Create a user (Profile page) to get personalised recommendations.")
        return
    st.write(f"Recommendations for **{user['name']}**")
    with st.expander("Adjust ranking & filters"):
        c = st.columns(5)
        w = {k: c[i].slider(k.capitalize(), 0.0, 1.0, float(v), 0.05) for i, (k, v) in enumerate(config.DEFAULT_WEIGHTS.items())}
        c2 = st.columns(3)
        top_n = c2[0].slider("How many", 3, 20, 8)
        excl = c2[1].checkbox("Hide books already read", True)
        gi = c2[2].multiselect("Only these genres", genre_options(svc))
        st.caption("Weights are normalised automatically. 'Collab' is ignored unless enough users have rated books.")
    if st.button("Get recommendations", type="primary"):
        res = Recommender(books).recommend(svc.users.get_history(user["user_id"]), user, top_n, None, excl,
                                           gi or None, w, svc.users.get_history())
        st.session_state["rec_table"], st.session_state["rec_notes"] = res.table, res.notes
        if len(res.table) and res.weights_used:
            svc.users.log_recommendations(user["user_id"], res.table)
    table = st.session_state.get("rec_table")
    if table is None:
        return
    for n in st.session_state.get("rec_notes", []):
        st.info(n)
    if table.empty:
        st.warning("No recommendations available with the current filters.")
        return
    hist = svc.users.get_history(user["user_id"])
    for r in table.to_dict("records"):
        book_card(r, f"**Match score:** {r['score']:.2f}",
                  f"Why: {r['explanation']}  ·  Availability: {r['availability']}  ·  Metadata source: {r['info_source']}")
        c = st.columns([1, 1, 4])
        if c[0].button("I'm interested", key=f"int_{r['book_id']}"):
            if hist[hist["book_id"] == r["book_id"]].empty:  # never overwrite an existing read/rating entry
                svc.users.set_status(user["user_id"], r["book_id"], "interested")
            svc.users.mark_selected(user["user_id"], r["book_id"])
            flash(f"Noted your interest in '{r['title']}'.")
            st.rerun()
        with c[1].expander("Score details"):
            st.write({k: r[k] for k in ("content", "genre_pref", "cluster_affinity", "author_affinity", "collab")})
