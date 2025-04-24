# Prototype: XML Matching AI using LangGraph + LangChain + FAISS + Streamlit

import os
import json
import xml.etree.ElementTree as ET
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
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

# 2. Convert dicts into documents for embedding
def create_text_representation(doc):
    return ", ".join([f"{k}: {v}" for k, v in doc.items()])

# 3. Load and process XML data (demo with static samples)
xml_samples = [
    '''<conditionSet><and><field><name>prdType</name><value>DM,DS</value></field><field><name>current</name><value>USD</value></field><field><name>cps</name><operator>gtoe</operator><value>0.200</value></field><field><name>cps</name><operator>ltoe</operator><value>0.300</value></field></and></conditionSet>''',
    '''<conditionSet><and><field><name>prdType</name><value>DX</value></field><field><name>current</name><value>EUR</value></field><field><name>cps</name><operator>gtoe</operator><value>0.100</value></field><field><name>cps</name><operator>ltoe</operator><value>0.150</value></field></and></conditionSet>'''
]

parsed_docs = [parse_xml_to_flat_dict(xml) for xml in xml_samples]
doc_texts = [create_text_representation(doc) for doc in parsed_docs]

# 4. Embed and store in FAISS
embedding = OpenAIEmbeddings()
faiss_store = FAISS.from_texts(doc_texts, embedding)

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
    return { "query": query, "structured": structured }

def search_vector_store_node(state):
    results = faiss_store.similarity_search(state["query"], k=2)
    return { **state, "results": results }

def return_results_node(state):
    return { "final_results": state["results"] }

# 7. Build LangGraph
graph = StateGraph()
graph.add_node("InterpretPrompt", interpret_prompt_node)
graph.add_node("SearchVectorStore", search_vector_store_node)
graph.add_node("ReturnResults", return_results_node)

graph.set_entry_point("InterpretPrompt")
graph.add_edge("InterpretPrompt", "SearchVectorStore")
graph.add_edge("SearchVectorStore", "ReturnResults")
graph.set_finish_point("ReturnResults")

runnable_graph = graph.compile()

# 8. Streamlit Interface
st.title("🔍 XML ConditionSet Matcher with AI")
user_query = st.text_input("Enter your search prompt:", "Find XMLs where product type is DM or DS, currency is USD, and cps between 0.2 and 0.3")

if st.button("Search"):
    with st.spinner("Running query through LangGraph..."):
        initial_state = { "query": user_query }
        result = runnable_graph.invoke(initial_state)

        st.subheader("🔎 Matched Results")
        for r in result["final_results"]:
            st.markdown(f"- {r.page_content}")
