import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="PIA Section Finder Chatbot", layout="wide")

# ---------------------------
# CUSTOM HEADER (LOGO + TITLE + NAME)
# ---------------------------
st.markdown(
    """
    <div style="text-align: center; margin-bottom: 20px;">
        <img src="https://raw.githubusercontent.com/sustanef/pia-chatbot/main/NMDPRA_logo.png"
             alt="NMDPRA Logo" width="130">
        <h1 style="margin-top: 10px;">📘 PIA 2021 – Section Finder & Chatbot</h1>
        <h3 style="color: #555;">By Abubakar Sani Hassan</h3>
        <hr>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# LOAD DATA (Cached)
# ---------------------------
@st.cache_data
def load_pia_data():
    df = pd.read_excel("PIA Detailed Sections.xlsx")
    df.columns = df.columns.str.strip()
    return df

df = load_pia_data()

# ---------------------------
# LOAD MODEL (Cached, CPU ONLY)
# ---------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")   # Efficient for Streamlit hosting

model = load_model()

# Precompute embeddings once for fast keyword search
@st.cache_resource
def compute_embeddings(texts):
    return model.encode(texts, convert_to_tensor=True)

embeddings = compute_embeddings(df["Contents of Section"].fillna("").tolist())

# ---------------------------
# UI TABS
# ---------------------------
tab1, tab2 = st.tabs(["🔎 Search by Section Number", "🧠 Ask a Question / Keyword Search"])

# ===========================
# TAB 1 — SECTION LOOKUP
# ===========================
with tab1:

    st.subheader("🔎 Look up a PIA Section")
    section_input = st.text_input("Enter a Section Number (e.g., 311)", "")

    if st.button("Search Section"):
        if section_input.strip():

            result = df[df["Section Numbers"].astype(str) == section_input.strip()]

            if not result.empty:
                row = result.iloc[0]

                st.success(f"Section {section_input} found ✔️")
                st.subheader(f"📌 {row['Title of Section']}")
                st.write(row["Contents of Section"])

            else:
                st.error("❌ Section not found. Please check the number.")

# ===========================
# TAB 2 — QUESTION / KEYWORD SEARCH
# ===========================
with tab2:

    st.subheader("🧠 Ask any question about the PIA or search with keywords")
    user_query = st.text_input("Enter your question or keyword(s)")

    if st.button("Search Content"):
        if user_query.strip():

            query_embedding = model.encode(user_query, convert_to_tensor=True)
            scores = util.cos_sim(query_embedding, embeddings)[0]

            best_idx = int(np.argmax(scores))
            best_row = df.iloc[best_idx]

            st.success("Best matching section found ✔️")

            st.subheader(f"📌 {best_row['Title of Section']} (Section {best_row['Section Numbers']})")
            st.write(best_row["Contents of Section"])

            st.caption(f"Similarity Score: {float(scores[best_idx]):.4f}")
