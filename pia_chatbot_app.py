import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import re

# Load PIA data
pia_df = pd.read_excel("PIA Detailed Sections.xlsx", header=1)
pia_df.columns = pia_df.columns.str.strip()

# Combine section titles and contents for semantic search
pia_df['combined_text'] = pia_df['Title of Section'].astype(str) + " " + pia_df['Contents of Section'].astype(str)

# Initialize sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Pre-compute embeddings for all sections
embeddings = model.encode(pia_df['combined_text'].tolist())

# Initialize summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Function to perform semantic search
def search_sections_semantic(query, top_k=1):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)
    top_indices = similarities[0].argsort()[-top_k:][::-1]
    return pia_df.iloc[top_indices]

# Function to directly fetch section by number
def get_section_by_number(section_number):
    match = pia_df[pia_df['Section Numbers'].astype(str).str.strip() == section_number]
    return match

# Function to summarize text
def summarize_text(text):
    if len(text.split()) < 50:
        return text
    summary = summarizer(text, max_length=130, min_length=50, do_sample=False)
    return summary[0]['summary_text']

# Streamlit UI setup
st.set_page_config(page_title="PIA 2021 Chatbot", page_icon="💬")
st.title("💬 Petroleum Industry Act (PIA) 2021 Chatbot")
st.write("Ask a question or type 'Section 311' to get details directly.")

# User input
query = st.text_input("🔍 Enter your question or section reference:")

# Summary option
summary_option = st.checkbox("Get a summarized response")

# Action on button click
if st.button("Ask"):
    if query:
        # Check for "Section XXX" pattern
        section_match = re.search(r'section\s*(\d+)', query, re.IGNORECASE)
        
        if section_match:
            section_number = section_match.group(1)
            results = get_section_by_number(section_number)

            if results.empty:
                st.warning(f"Could not find Section {section_number}. Showing closest match instead.")
                results = search_sections_semantic(query)
        else:
            results = search_sections_semantic(query)

        if results.empty:
            st.warning("No matching section found. Try a different question.")
        else:
            for _, row in results.iterrows():
                section_title = row.get('Title of Section', 'Untitled')
                section_number = row.get('Section Numbers', 'N/A')
                st.subheader(f"📌 {section_title} (Section {section_number})")

                content = row.get('Contents of Section', 'No content available.')
                display_content = summarize_text(content) if summary_option else content
                st.write(display_content)
    else:
        st.warning("Please enter a question or section number to search the PIA.")
