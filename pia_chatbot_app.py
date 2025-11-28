import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import re

# -----------------------------
# Load model (CPU only – Render friendly)
# -----------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# -----------------------------
# Load PIA Excel
# -----------------------------
df = pd.read_excel('PIA Detailed Sections.xlsx', header=0)
df.columns = df.columns.str.strip()

# Precompute embeddings for semantic search
if "embeddings" not in st.session_state:
    st.session_state.embeddings = model.encode(
        df["Contents of Section"].fillna("").tolist(),
        show_progress_bar=True
    )

# -----------------------------
# Helper: Keyword Highlight
# -----------------------------
def highlight_keywords(text, keyword):
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    return pattern.sub(lambda m: f"**{m.group(0)}**", text)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📘 PIA 2021 – Smart Legal Search Chatbot")

search_mode = st.selectbox(
    "Choose Search Mode",
    ["Section Lookup", "Keyword Search", "Semantic Search", "Combined Search"]
)

user_input = st.text_input("🔍 Enter Search Query (e.g., 311, gas pricing, transition, host community…)")

if user_input:

    # ---------------------------------
    # SECTION LOOKUP
    # ---------------------------------
    if search_mode == "Section Lookup":
        try:
            result = df.loc[df["Section Numbers"].astype(str) == user_input.strip()]

            if not result.empty:
                row = result.iloc[0]
                st.subheader(f"📌 {row['Title of Section']} (Section {row['Section Numbers']})")
                st.write(row["Contents of Section"])
            else:
                st.error("❌ Section not found.")
        except Exception as e:
            st.error(f"⚠️ {e}")

    # ---------------------------------
    # KEYWORD SEARCH
    # ---------------------------------
    elif search_mode == "Keyword Search":
        keyword = user_input.strip()
        hits = df[df["Contents of Section"].str.contains(keyword, case=False, na=False)]

        if hits.empty:
            st.warning("No exact keyword matches found.")
        else:
            for _, row in hits.iterrows():
                st.markdown(f"### 📌 Section {row['Section Numbers']} — {row['Title of Section']}")
                highlighted = highlight_keywords(row["Contents of Section"], keyword)
                st.markdown(highlighted)

    # ---------------------------------
    # SEMANTIC SEARCH
    # ---------------------------------
    elif search_mode == "Semantic Search":
        query_embedding = model.encode([user_input])[0]
        all_embeddings = np.array(st.session_state.embeddings)

        # Compute cosine similarity
        similarity_scores = np.dot(all_embeddings, query_embedding) / (
            np.linalg.norm(all_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        top_n = 5
        top_indices = np.argsort(similarity_scores)[-top_n:][::-1]

        st.subheader("🔎 Top Relevant Sections")

        for idx in top_indices:
            row = df.iloc[idx]
            st.markdown(f"### 📌 {row['Title of Section']} (Section {row['Section Numbers']})")
            st.write(row["Contents of Section"])

    # ---------------------------------
    # COMBINED SEARCH (Keyword + Semantic)
    # ---------------------------------
    elif search_mode == "Combined Search":
        keyword = user_input.strip()

        # Keyword hits
        keyword_hits = df[df["Contents of Section"].str.contains(keyword, case=False, na=False)]

        # Semantic hits
        query_embedding = model.encode([user_input])[0]
        all_embeddings = np.array(st.session_state.embeddings)
        similarity_scores = np.dot(all_embeddings, query_embedding) / (
            np.linalg.norm(all_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        top_indices = np.argsort(similarity_scores)[-5:][::-1]

        st.subheader("🔎 Keyword Matches (if any)")
        if keyword_hits.empty:
            st.write("No keyword matches.")
        else:
            for _, row in keyword_hits.iterrows():
                st.markdown(f"### 📌 {row['Title of Section']} (Section {row['Section Numbers']})")
                highlighted = highlight_keywords(row["Contents of Section"], keyword)
                st.markdown(highlighted)

        st.subheader("🔎 Top Semantic Matches")
        for idx in top_indices:
            row = df.iloc[idx]
            st.markdown(f"### 📌 {row['Title of Section']} (Section {row['Section Numbers']})")
            st.write(row["Contents of Section"])
