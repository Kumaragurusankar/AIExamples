import streamlit as st
import requests

# ---------- CONFIG ----------
STYLES = """
    <style>
        .main {
            background-color: #f5f7fa;
            font-family: 'Segoe UI', sans-serif;
            padding: 2rem;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .sample-box {
            background-color: #ffffff;
            padding: 1rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        }
        .result-box {
            background-color: #ffffff;
            padding: 1rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            white-space: pre-wrap;
        }
        .logo {
            max-height: 80px;
            margin-bottom: 10px;
        }
    </style>
"""

BACKEND_SEARCH_ENDPOINT = "http://xml-search-app:8000/search"  # Update for your route/service name

# ---------- LAYOUT ----------
st.set_page_config(page_title="AI Trade Assist", layout="wide")
st.markdown(STYLES, unsafe_allow_html=True)

col1, col2 = st.columns([1, 3])

with col1:
    st.image("logo.png", width=120)  # Replace with your logo or file path
    st.markdown("### AI Trade Assist")
    st.markdown("Search and interpret XML rules using natural language.")

with col2:
    st.markdown("### Ask a question")
    user_query = st.text_input("Enter your prompt (e.g., 'Show deals with prd type DS and currency USD')")

    with st.expander("📘 Sample Prompts", expanded=False):
        st.markdown("""
        - *List deals where prd type is DS and currency is USD*
        - *Find all rules with bps greater than 1.5*
        - *What are the conditions for deal type DS in United States?*
        """)

    disclaimer = st.checkbox("⚠️ I understand that results may not be 100% accurate")

    if st.button("Run AI Search") and user_query and disclaimer:
        with st.spinner("Contacting AI engine..."):
            try:
                response = requests.post(BACKEND_SEARCH_ENDPOINT, json={"prompt": user_query})
                result = response.json().get("result", "No match found.")
                st.markdown("### 🔍 Results")
                st.markdown(f"<div class='result-box'>{result}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error occurred: {e}")
    elif not disclaimer and user_query:
        st.warning("Please accept the disclaimer before running the AI search.")
