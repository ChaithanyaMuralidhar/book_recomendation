"""Loads the bundled demonstration dataset (idempotent: re-running never creates duplicates)."""
import pandas as pd

import config
from services.catalogue_service import CatalogueService
from services.user_service import UserService


def load_sample(catalogue: CatalogueService, users: UserService) -> dict:
    books = pd.read_csv(config.SAMPLE_BOOKS_CSV, dtype=str, keep_default_na=False)
    result = catalogue.add_many(books.to_dict("records"))
    # map sample ids -> real ids by title/author, so history is correct even if the DB already had other books
    id_map = {r.book_id: catalogue.find_duplicate(r.title, r.author, r.isbn) for r in books.itertuples()}
    existing = set(users.list_users()["user_id"])
    n_users = 0
    for r in pd.read_csv(config.SAMPLE_USERS_CSV, dtype=str, keep_default_na=False).itertuples():
        if r.user_id not in existing:
            users.create_user(r.name, [g for g in r.pref_genres.split("|") if g],
                              [a for a in r.pref_authors.split("|") if a], user_id=r.user_id)
        n_users += 1
    n_hist = 0
    for r in pd.read_csv(config.SAMPLE_HISTORY_CSV, dtype=str, keep_default_na=False).itertuples():
        real_id = id_map.get(r.book_id)
        if real_id is None:
            continue
        users.set_status(r.user_id, real_id, r.status, int(r.rating) if r.rating else None,
                         int(r.liked) if r.liked else None, r.read_date or None)
        n_hist += 1
    return {**result, "users": n_users, "history_rows": n_hist}
