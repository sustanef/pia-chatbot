import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load model safely on CPU for Streamlit Cloud
device = "cpu"
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# Load your Excel file
df = pd.read_excel('PIA Detailed Sections.xlsx')

# Clean up column names
df.columns = [col.strip() for col in df.columns]

st.title("📑 PIA 2021 Section Finder Chatbot")

# User input
section_input = st.text_input("🔍 Enter a Section Number (e.g. 311)")

if section_input:
    try:
        result_row = df.loc[df['Section Numbers'].astype(str) == section_input]

        if not result_row.empty:
            section_title = result_row.iloc[0]['Title of Section']
            section_content = result_row.iloc[0]['Contents of Section']
            st.subheader(f"📌 {section_title} (Section {section_input})")
            st.write(section_content)
        else:
            st.error("❌ Section not found. Please check the number you entered.")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
