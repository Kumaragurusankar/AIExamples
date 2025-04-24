import streamlit as st
import os
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer, util
import numpy as np
import glob
import pickle

# --- Config ---
XML_FOLDER = "./xml_files"
EMBEDDINGS_FILE = "xml_embeddings.pkl"
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Helper Functions ---
def parse_xml_to_text(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        return flatten_xml(root)
    except Exception as e:
        return f"Error parsing {file_path}: {str(e)}"

def flatten_xml(element, depth=0):
    content = []
    indent = "  " * depth
    for child in element:
        tag = child.tag
        text = (child.text or '').strip()
        content.append(f"{indent}{tag}: {text}")
        content.append(flatten_xml(child, depth + 1))
    return "\n".join(filter(None, content))

def load_or_generate_embeddings():
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "rb") as f:
            return pickle.load(f)

    xml_files = glob.glob(os.path.join(XML_FOLDER, "*.xml"))
    docs = []
    embeddings = []

    for file_path in xml_files:
        text = parse_xml_to_text(file_path)
        embedding = model.encode(text, convert_to_tensor=True)
        docs.append((file_path, text))
        embeddings.append(embedding)

    data = {"docs": docs, "embeddings": embeddings}
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(data, f)

    return data

def search_xml(query, data):
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = [util.cos_sim(query_embedding, emb)[0][0].item() for emb in data["embeddings"]]
    results = sorted(zip(scores, data["docs"]), reverse=True)
    return results[:5]  # Top 5 matches

# --- Streamlit UI ---
st.title("AI-Powered XML Search")
st.write("Search through 2000+ XML files using semantic understanding.")

query = st.text_input("Enter your search query (natural language or field-based)")

if query:
    with st.spinner("Loading and searching..."):
        data = load_or_generate_embeddings()
        top_matches = search_xml(query, data)

    st.success("Top Matches:")
    for score, (file_path, content) in top_matches:
        st.markdown(f"**{os.path.basename(file_path)}** - Score: {score:.4f}")
        with st.expander("Show XML Content"):
            st.text(content)