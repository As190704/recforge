# app/streamlit_app.py
"""
RecForge Streamlit frontend.

Pages:
  1. Similar Items — search for a film, see content-based similar films
  2. Personalized — enter a user ID, see hybrid recommendations
  3. Trending — popularity-based recommendations
  4. Metrics — evaluation dashboard

Calls the FastAPI backend via HTTP requests.
"""

import streamlit as st
import requests
import pandas as pd
import os

API_BASE = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="RecForge",
    page_icon="🎬",
    layout="wide",
)


# ── helpers ──────────────────────────────────────────────────────────────────

def api_get(path: str, params: dict = None):
    try:
        resp = requests.get(f"{API_BASE}{path}", params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the RecForge API. Make sure it's running.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API error: {e.response.status_code} — {e.response.text}")
        return None


def score_bar(score: float) -> str:
    """Return a simple text progress bar for a [0,1] score."""
    filled = int(score * 10)
    return "█" * filled + "░" * (10 - filled)


# ── sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.image("https://via.placeholder.com/200x60?text=RecForge", use_column_width=True)
st.sidebar.markdown("## Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🎬 Similar Items", "👤 Personalized", "🔥 Trending", "📊 Metrics"],
    label_visibility="collapsed",
)

health = api_get("/health")
if health and health.get("status") == "ok":
    st.sidebar.success("API connected")
else:
    st.sidebar.error("API offline")


# ── page: similar items ───────────────────────────────────────────────────────

if page == "🎬 Similar Items":
    st.title("🎬 Find Similar Films")
    st.markdown("Enter a MovieLens item ID to find content-based similar films.")

    col1, col2 = st.columns([3, 1])
    with col1:
        item_id_input = st.number_input(
            "Item ID", min_value=1, max_value=1682, value=1, step=1,
            help="MovieLens item IDs range from 1 to 1682. ID 1 = Toy Story (1995)."
        )
    with col2:
        n_similar = st.slider("Results", min_value=3, max_value=20, value=10)

    if st.button("Find Similar Films", type="primary"):
        data = api_get(f"/similar/{int(item_id_input)}", {"n": n_similar})
        if data:
            st.subheader(f"Films similar to: **{data['title']}**")
            for item in data["similar_items"]:
                with st.container():
                    col_a, col_b = st.columns([4, 1])
                    with col_a:
                        st.markdown(f"**{item['title']}**  \n*{item['genres'].replace('|', ' · ')}*")
                        st.caption(f"💡 {item['explanation']}")
                    with col_b:
                        st.markdown(f"`{score_bar(item['score'])}`  \n{item['score']:.3f}")
                st.divider()


# ── page: personalized ────────────────────────────────────────────────────────

elif page == "👤 Personalized":
    st.title("👤 Personalized Recommendations")
    st.markdown("Enter a MovieLens user ID to see your hybrid recommendations.")

    col1, col2 = st.columns([3, 1])
    with col1:
        user_id_input = st.number_input(
            "User ID", min_value=1, max_value=943, value=1, step=1,
            help="MovieLens has 943 users (IDs 1–943)."
        )
    with col2:
        n_recs = st.slider("Results", min_value=3, max_value=20, value=10)

    if st.button("Get Recommendations", type="primary"):
        data = api_get(f"/recommend/{int(user_id_input)}", {"n": n_recs})
        if data:
            rec_type_label = {
                "hybrid": "🔀 Hybrid (collaborative + content)",
                "popularity": "🔥 Popularity (cold start)",
            }.get(data["recommendation_type"], data["recommendation_type"])

            st.markdown(f"**Mode:** {rec_type_label}")
            st.markdown(f"Showing **{data['n']}** recommendations for User **{data['user_id']}**")
            st.markdown("---")

            for item in data["recommendations"]:
                with st.container():
                    col_a, col_b = st.columns([4, 1])
                    with col_a:
                        st.markdown(f"**{item['title']}**  \n*{item['genres'].replace('|', ' · ')}*")
                        st.caption(f"💡 {item['explanation']}")
                    with col_b:
                        st.markdown(f"`{score_bar(item['score'])}`  \n{item['score']:.3f}")
                st.divider()


# ── page: trending ────────────────────────────────────────────────────────────

elif page == "🔥 Trending":
    st.title("🔥 Trending Films")
    st.markdown("Most popular films by Bayesian rating score.")

    n_trending = st.slider("How many to show", min_value=5, max_value=30, value=10)

    data = api_get("/trending", {"n": n_trending})
    if data:
        for rank, item in enumerate(data["trending_items"], start=1):
            with st.container():
                col_a, col_b = st.columns([4, 1])
                with col_a:
                    st.markdown(f"**#{rank} — {item['title']}**  \n*{item['genres'].replace('|', ' · ')}*")
                    st.caption(f"💡 {item['explanation']}")
                with col_b:
                    st.markdown(f"`{score_bar(min(item['score'] / 5.0, 1.0))}`  \n{item['score']:.3f}")
            st.divider()


# ── page: metrics ─────────────────────────────────────────────────────────────

elif page == "📊 Metrics":
    st.title("📊 Model Evaluation Metrics")
    st.markdown(
        "Metrics computed on a **leave-last-out** test split: "
        "for each user, the most recent interaction is held out and "
        "the model must rank it within the top 10 predictions."
    )

    data = api_get("/metrics")
    if data:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Precision@10", f"{data['precision_at_10']:.4f}")
        col2.metric("Recall@10", f"{data['recall_at_10']:.4f}")
        col3.metric("NDCG@10", f"{data['ndcg_at_10']:.4f}")
        col4.metric("Coverage", f"{data['coverage']:.2%}")

        st.markdown(f"*Evaluated on **{data['users_evaluated']}** users.*")

        st.markdown("---")
        st.markdown("### What these mean")
        st.markdown("""
| Metric | What it measures |
|--------|-----------------|
| **Precision@10** | Of the 10 items shown, what fraction was the held-out test item? Expected ~0.1 max for single-item test. |
| **Recall@10** | Did the test item appear anywhere in the top 10? |
| **NDCG@10** | Was the test item ranked near the top? Penalizes lower ranks. |
| **Coverage** | What fraction of all 1,682 films appear in at least one user's top-10? Higher = more diverse. |
        """)

        st.info(
            "These are honest numbers from running the evaluation pipeline on a held-out test split. "
            "See `evaluation/run_evaluation.py` for the full comparison table including "
            "Popularity baseline and Collaborative-only results."
        )