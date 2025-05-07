import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load Excel file with correct header row
df = pd.read_excel('PIA Detailed Sections.xlsx', header=0)

# Clean up column names
df.columns = df.columns.str.strip()

# Diagnostic: Show column names
st.write("Live columns loaded:", df.columns.tolist())

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
