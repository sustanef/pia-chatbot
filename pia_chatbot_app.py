import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
from PIL import Image

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------
st.set_page_config(page_title="PIA Section Finder Chatbot", layout="wide")

# -----------------------------------------------------
# HEADER WITH LOGO
# -----------------------------------------------------
col1, col2 = st.columns([1, 6])

with col1:
    try:
        logo = Image.open("NMDPRA_logo.png")
        st.image(logo, width=110)
    except:
        st.warning("NMDPRA Logo not found.")

with col2:
    st.markdown("""
        <h1 style="margin-bottom: -10px;">PIA 2021 – Section Finder & Intelligent Chatbot</h1>
        <h4 style="color:#444;">Developed by <b>Abubakar Sani Hassan</b>,PhD,SMIEEE,MIET.</h4>
    """, unsafe_allow_html=True)

st.markdown("---")

# -----------------------------------------------------
# LOAD DATA
# -----------------------------------------------------
@st.cache_data
def load_pia_data():
    df = pd.read_excel("PIA Detailed Sections.xlsx")
    df.columns = df.columns.str.strip()
    return df

df = load_pia_data()

# -----------------------------------------------------
# LOAD MODEL
# -----------------------------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -----------------------------------------------------
# PRECOMPUTE EMBEDDINGS
# -----------------------------------------------------
@st.cache_resource
def compute_embeddings(texts):
    return model.encode(texts, convert_to_tensor=True)

embeddings = compute_embeddings(df["Contents of Section"].fillna("").tolist())

# -----------------------------------------------------
# TABS
# -----------------------------------------------------
tab_section, tab_query = st.tabs([
    "🔎 Search by Section Number",
    "🧠 Ask a Question / Keyword Search"
])

# =====================================================
# TAB 1 — SECTION LOOKUP
# =====================================================
with tab_section:

    st.subheader("🔎 Look up a PIA Section")

    section_input = st.text_input(
        "Enter a Section Number (e.g., 311)",
        key="section_input"
    )

    if st.button("Search Section"):
        if section_input.strip():

            result = df[df["Section Numbers"].astype(str).str.strip() == section_input.strip()]

            if not result.empty:
                row = result.iloc[0]

                st.success(f"Section {section_input} found ✔️")
                st.subheader(f"📌 {row['Title of Section']}")
                st.write(row["Contents of Section"])
            else:
                st.error("❌ Section not found. Please check the number.")

# =====================================================
# TAB 2 — QUESTION / KEYWORD SEARCH
# =====================================================
with tab_query:

    st.subheader("🧠 Ask any question about the PIA or search with keywords")

    user_query = st.text_input(
        "Enter your question or keyword(s)",
        key="keyword_query"
    )

    if st.button("Search Content"):
        if user_query.strip():

            query_emb = model.encode(user_query, convert_to_tensor=True)

            scores = util.cos_sim(query_emb, embeddings)[0]
            best_idx = int(np.argmax(scores))
            best_row = df.iloc[best_idx]

            st.success("Best matching section found ✔️")

            st.subheader(
                f"📌 {best_row['Title of Section']} "
                f"(Section {best_row['Section Numbers']})"
            )
            st.write(best_row["Contents of Section"])

            st.caption(f"Similarity Score: {float(scores[best_idx]):.4f}")
