"""Multimodal Intelligent Book Recommendation System - Streamlit entry point.  Run: streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="Book Recommendation System", page_icon="📚", layout="wide")

from ui import analytics, catalogue, excel, home, library, profile, recommend, scan, settings, video  # noqa: E402
from ui.state import get_services  # noqa: E402

PAGES = {
    "🏠 Home Dashboard": home,
    "📷 Scan Books": scan,
    "🎥 Video / Webcam / VR": video,
    "📊 Excel Import & Export": excel,
    "📖 Book Catalogue": catalogue,
    "👤 Profile & Reading History": profile,
    "✨ AI Recommendations": recommend,
    "🏛️ Scan a Library / Store": library,
    "📈 ML Analytics & Clustering": analytics,
    "⚙️ Settings & Help": settings,
}

st.markdown("<style>.stButton>button{border-radius:8px}[data-testid=stMetricValue]{font-size:1.6rem}</style>",
            unsafe_allow_html=True)
svc = get_services()
if "_select_user" in st.session_state:  # a page asked to switch user (widget keys cannot be set after creation)
    st.session_state["user_id"] = st.session_state.pop("_select_user")
with st.sidebar:
    st.title("📚 BookRec")
    page = st.radio("Navigate", list(PAGES), key="page", label_visibility="collapsed")
    users = svc.users.list_users()
    if len(users):
        names = dict(zip(users["user_id"], users["name"]))
        st.selectbox("Active user", list(names), format_func=lambda u: f"{names[u]} ({u})", key="user_id")
    else:
        st.caption("No users yet.")
    st.caption(f"{svc.catalogue.count()} books in catalogue")

PAGES[page].render(svc)
