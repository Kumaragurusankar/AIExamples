# xml_search_streamlit.py
import streamlit as st
import os
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer, util
import numpy as np
import glob
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import json

# --- Config ---
XML_FOLDER = "./xml_files"
ES_INDEX = "xml_documents"
model = SentenceTransformer("all-MiniLM-L6-v2")
es = Elasticsearch("http://localhost:9200")

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

def create_index_if_not_exists():
    if not es.indices.exists(index=ES_INDEX):
        es.indices.create(index=ES_INDEX, body={
            "mappings": {
                "properties": {
                    "filename": {"type": "text"},
                    "content": {"type": "text"},
                    "embedding": {"type": "dense_vector", "dims": 384}
                }
            }
        })

def upload_documents_to_elasticsearch():
    xml_files = glob.glob(os.path.join(XML_FOLDER, "*.xml"))
    actions = []
    for file_path in xml_files:
        text = parse_xml_to_text(file_path)
        embedding = model.encode(text).tolist()
        actions.append({
            "_index": ES_INDEX,
            "_source": {
                "filename": os.path.basename(file_path),
                "content": text,
                "embedding": embedding
            }
        })
    if actions:
        bulk(es, actions)

def search_elasticsearch(query):
    query_vector = model.encode(query).tolist()
    body = {
        "size": 5,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {"query_vector": query_vector}
                }
            }
        }
    }
    response = es.search(index=ES_INDEX, body=body)
    return response['hits']['hits']

# --- Streamlit UI ---
st.title("AI-Powered XML Search with Elasticsearch")
st.write("Search through 2000+ XML files using semantic understanding and Elasticsearch.")

query = st.text_input("Enter your search query (natural language or field-based)")

if query:
    with st.spinner("Connecting to Elasticsearch and searching..."):
        create_index_if_not_exists()
        if es.count(index=ES_INDEX)['count'] == 0:
            upload_documents_to_elasticsearch()
        top_matches = search_elasticsearch(query)

    st.success("Top Matches:")
    for match in top_matches:
        source = match['_source']
        st.markdown(f"**{source['filename']}**")
        with st.expander("Show XML Content"):
            st.text(source['content'])
