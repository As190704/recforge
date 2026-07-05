# data/ingest.py
"""
MovieLens 100K ingestion pipeline.

Downloads raw data, validates schema, cleans it, and writes
processed parquet files to data/processed/.

Run: python -m data.ingest
"""

import os
import zipfile
import urllib.request
import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ZIP_PATH = RAW_DIR / "ml-100k.zip"


def download_data():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists():
        print("Raw zip already exists, skipping download.")
        return
    print(f"Downloading MovieLens 100K from {MOVIELENS_URL} ...")
    urllib.request.urlretrieve(MOVIELENS_URL, ZIP_PATH)
    print("Download complete.")


def extract_data():
    extract_path = RAW_DIR / "ml-100k"
    if extract_path.exists():
        print("Already extracted, skipping.")
        return
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(RAW_DIR)
    print("Extraction complete.")


def load_ratings() -> pd.DataFrame:
    path = RAW_DIR / "ml-100k" / "u.data"
    df = pd.read_csv(
        path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        dtype={"user_id": int, "item_id": int, "rating": float, "timestamp": int},
    )
    return df


def load_items() -> pd.DataFrame:
    path = RAW_DIR / "ml-100k" / "u.item"
    genre_cols = [
        "unknown", "Action", "Adventure", "Animation", "Children",
        "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
        "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
        "Sci-Fi", "Thriller", "War", "Western",
    ]
    cols = ["item_id", "title", "release_date", "video_release_date", "imdb_url"] + genre_cols
    df = pd.read_csv(
        path,
        sep="|",
        names=cols,
        encoding="latin-1",
        usecols=["item_id", "title", "release_date"] + genre_cols,
        dtype={"item_id": int, "title": str},
    )
    return df


def load_users() -> pd.DataFrame:
    path = RAW_DIR / "ml-100k" / "u.user"
    df = pd.read_csv(
        path,
        sep="|",
        names=["user_id", "age", "gender", "occupation", "zip_code"],
        dtype={"user_id": int, "age": int, "gender": str, "occupation": str},
    )
    return df


def clean_ratings(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)

    # Remove duplicates (keep last interaction per user-item pair)
    df = df.drop_duplicates(subset=["user_id", "item_id"], keep="last")

    # Drop rows with null user_id, item_id, or rating
    df = df.dropna(subset=["user_id", "item_id", "rating"])

    # Validate rating range
    df = df[df["rating"].between(1.0, 5.0)]

    # Validate timestamp is positive
    df = df[df["timestamp"] > 0]

    after = len(df)
    print(f"Ratings: {before} rows → {after} rows after cleaning.")
    return df.reset_index(drop=True)


def clean_items(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)

    df = df.dropna(subset=["item_id", "title"])
    df = df.drop_duplicates(subset=["item_id"])

    # Extract year from title e.g. "Toy Story (1995)"
    df["year"] = df["title"].str.extract(r"\((\d{4})\)$").astype(float)

    # Build genre list as a pipe-separated string
    genre_cols = [
        "unknown", "Action", "Adventure", "Animation", "Children",
        "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
        "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
        "Sci-Fi", "Thriller", "War", "Western",
    ]
    df["genres"] = df[genre_cols].apply(
        lambda row: "|".join([g for g, v in row.items() if v == 1]), axis=1
    )

    # Build description field for content-based embedding
    df["description"] = df["title"] + " genres: " + df["genres"].str.replace("|", " ", regex=False)

    df = df[["item_id", "title", "year", "genres", "description"]]

    after = len(df)
    print(f"Items: {before} rows → {after} rows after cleaning.")
    return df.reset_index(drop=True)


def validate(ratings: pd.DataFrame, items: pd.DataFrame):
    assert ratings["user_id"].nunique() > 0, "No users found."
    assert ratings["item_id"].nunique() > 0, "No items found."
    assert ratings["rating"].between(1.0, 5.0).all(), "Ratings out of range."
    assert items["item_id"].nunique() == len(items), "Duplicate item IDs."
    assert items["description"].notna().all(), "Null descriptions found."

    # Check overlap between ratings and items
    rated_items = set(ratings["item_id"].unique())
    known_items = set(items["item_id"].unique())
    missing = rated_items - known_items
    print(f"Items in ratings but not in metadata: {len(missing)}")

    print("Validation passed.")


def save_processed(ratings: pd.DataFrame, items: pd.DataFrame):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ratings.to_parquet(PROCESSED_DIR / "ratings.parquet", index=False)
    items.to_parquet(PROCESSED_DIR / "items.parquet", index=False)
    print(f"Saved processed files to {PROCESSED_DIR}/")


def main():
    download_data()
    extract_data()

    ratings_raw = load_ratings()
    items_raw = load_items()

    ratings = clean_ratings(ratings_raw)
    items = clean_items(items_raw)

    validate(ratings, items)
    save_processed(ratings, items)

    print("\n--- Summary ---")
    print(f"Users:        {ratings['user_id'].nunique():>6,}")
    print(f"Items:        {ratings['item_id'].nunique():>6,}")
    print(f"Ratings:      {len(ratings):>6,}")
    print(f"Sparsity:     {1 - len(ratings) / (ratings['user_id'].nunique() * ratings['item_id'].nunique()):.4f}")
    print(f"Rating mean:  {ratings['rating'].mean():.2f}")


if __name__ == "__main__":
    main()