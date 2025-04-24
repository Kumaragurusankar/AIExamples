# Prototype: XML Matching AI using LangGraph + LangChain + FAISS + Streamlit

import os
import json
import xml.etree.ElementTree as ET
import streamlit as st
import pandas as pd
import faiss
import numpy as np
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langgraph.graph import StateGraph

# 1. Parse XML into flat dictionary

def parse_xml_to_flat_dict(xml_str):
    root = ET.fromstring(xml_str)
    flat_dict = {}
    for field in root.findall(".//field"):
        name = field.find("name").text
        value = field.find("value").text if field.find("value") is not None else ""
        operator = field.find("operator").text if field.find("operator") is not None else ""
        key = name if operator == '' else f"{name}_{operator}"
        flat_dict[key] = value
    return flat_dict

# 2. Read XML strings from CSV
csv_path = "xml_data.csv"  # CSV should have a column named 'xml'
df = pd.read_csv(csv_path)
xml_strings = df['xml'].tolist()

# 3. Convert to flat dict and generate vector embeddings
flat_texts = [json.dumps(parse_xml_to_flat_dict(xml)) for xml in xml_strings]
embedding_model = OpenAIEmbeddings()
embeddings = embedding_model.embed_documents(flat_texts)

# 4. Create FAISS index
dimension = len(embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype('float32'))

# 5. Define LLM Prompt to interpret user input
prompt_template = PromptTemplate(
    input_variables=["query"],
    template="""
You are a smart data parser. Convert the following user request into structured search terms:

Query: {query}

Respond with key-value pairs in JSON format.
"""
)

llm = OpenAI(temperature=0)
llm_chain = LLMChain(llm=llm, prompt=prompt_template)

# 6. LangGraph Nodes

def interpret_prompt_node(state):
    query = state["query"]
    structured = llm_chain.run(query)
    try:
        structured_dict = json.loads(structured)
    except:
        structured_dict = {}
    return { "query": query, "structured": structured_dict }

def search_faiss_node(state):
    structured_dict = state["structured"]
    query_text = json.dumps(structured_dict)
    query_embedding = embedding_model.embed_query(query_text)
    D, I = index.search(np.array([query_embedding]).astype('float32'), k=5)
    matches = [flat_texts[i] for i in I[0]]
    return { **state, "results": matches }

def return_results_node(state):
    return { "final_results": state["results"] }

# 7. Build LangGraph
graph = StateGraph()
graph.add_node("InterpretPrompt", interpret_prompt_node)
graph.add_node("SearchFAISS", search_faiss_node)
graph.add_node("ReturnResults", return_results_node)

graph.set_entry_point("InterpretPrompt")
graph.add_edge("InterpretPrompt", "SearchFAISS")
graph.add_edge("SearchFAISS", "ReturnResults")
graph.set_finish_point("ReturnResults")

runnable_graph = graph.compile()

# 8. Streamlit Interface
st.title("🔍 XML ConditionSet Matcher with FAISS + AI")
st.markdown("""
Upload a CSV file containing XML strings under a column named `xml`.
The app will match your natural language query to the closest condition sets.
""")

user_query = st.text_input("Enter your search prompt:", "Find XMLs where product type is DM or DS, currency is USD, and cps between 0.2 and 0.3")

if st.button("Search"):
    with st.spinner("Running query through LangGraph and FAISS..."):
        initial_state = { "query": user_query }
        result = runnable_graph.invoke(initial_state)

        st.subheader("🔎 Matched Results")
        if result["final_results"]:
            for r in result["final_results"]:
                st.json(json.loads(r))
        else:
            st.warning("No matching results found.")
