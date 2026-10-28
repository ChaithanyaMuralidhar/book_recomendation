"""Shared service container (created once per Streamlit server) and session helpers."""
from dataclasses import dataclass

import streamlit as st

import config
from services.catalogue_service import CatalogueService
from services.db import Database
from services.image_scanner import ImageScanner
from services.metadata_service import MetadataService
from services.user_service import UserService


@dataclass
class Services:
    db: Database
    catalogue: CatalogueService
    users: UserService
    metadata: MetadataService
    scanner: ImageScanner


@st.cache_resource
def get_services() -> Services:
    db = Database(config.DB_PATH)
    cat = CatalogueService(db)
    meta = MetadataService(catalogue_provider=cat.list_books, online=True)
    return Services(db, cat, UserService(db), meta, ImageScanner(meta))


def current_user(svc: Services):
    """The user chosen in the sidebar (dict) or None when no users exist yet."""
    ids = list(svc.users.list_users()["user_id"])
    if not ids:
        return None
    uid = st.session_state.get("user_id")
    return svc.users.get_user(uid if uid in ids else ids[0])  # never assign: the sidebar widget owns this key
