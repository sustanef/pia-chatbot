import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load summarizer
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# Load Excel file — replace header=1 with correct value after preview
df = pd.read_excel('PIA Detailed Sections.xlsx', header=1)
df.columns = df.columns.str.strip()

# Optional: Confirm columns during testing
# st.write("Live columns loaded:", df.columns.tolist())

st.title("📑 PIA 2021 Section Finder Chatbot")

# User input
section_input = st.text_input("🔍 Enter a Section Number (e.g. 311)")

# Summary toggle
summarize_option = st.checkbox("Summarized version")

if section_input:
    try:
        result_row = df.loc[df['Section Numbers'].astype(str) == section_input]

        if not result_row.empty:
            section_title = result_row.iloc[0]['Title of Section']
            section_content = result_row.iloc[0]['Contents of Section']

            st.subheader(f"📌 {section_title} (Section {section_input})")

            if summarize_option:
                if len(section_content.split()) < 50:
                    st.info("Section is too short to summarize meaningfully.")
                    st.write(section_content)
                else:
                    summary = summarizer(section_content, max_length=200, min_length=60, do_sample=False)
                    st.subheader("🔍 Summary")
                    st.write(summary[0]['summary_text'])
            else:
                st.write(section_content)

        else:
            st.error("❌ Section not found. Please check the number you entered.")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
