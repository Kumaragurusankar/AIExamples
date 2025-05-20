import streamlit as st
import requests

# Constants
API_URL = "http://localhost:8000/search"  # Replace with your cluster route
SAMPLE_PROMPTS = [
    "Find prd type DM or DS and currency USD",
    "Show deals with country United States or US",
    "bps between 1.5 and 2",
    "prodType is FX and currency is EUR"
]

# App Config
st.set_page_config(page_title="AI XML Matcher", layout="centered")

# Header
st.markdown("## 🤖 AI XML Matching Assistant")
st.markdown("""
Welcome to the AI-powered assistant for structured XML matching.  
Enter your query in natural language and get AI-curated matches!
""")

# Sample Prompts
st.markdown("### 🔍 Try a sample prompt:")
cols = st.columns(len(SAMPLE_PROMPTS))
for idx, prompt in enumerate(SAMPLE_PROMPTS):
    if cols[idx].button(f"Prompt {idx+1}", key=f"sample_{idx}"):
        st.session_state['query'] = prompt

# Query input
query = st.text_area("📝 Enter your prompt", value=st.session_state.get('query', ''), height=120)

# Disclaimer
disclaimer_checked = st.checkbox("I understand this is an AI-powered assistant and may not produce exact matches.")

# Submit Button
submit_btn = st.button("🚀 Search")

# Results
if submit_btn:
    if not disclaimer_checked:
        st.warning("You must accept the disclaimer to proceed.")
    elif not query.strip():
        st.error("Please enter a valid prompt.")
    else:
        with st.spinner("Querying AI engine..."):
            try:
                response = requests.post(API_URL, json={"prompt": query})
                response.raise_for_status()
                result = response.json().get("result", "No result found.")
                st.success("✅ Results found!")
                st.markdown("### 🧾 AI Response:")
                st.code(result, language="json")
            except Exception as e:
                st.error(f"Error: {str(e)}")
