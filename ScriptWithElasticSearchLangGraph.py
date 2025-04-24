# Prototype: XML Matching AI using LangGraph + LangChain + Elasticsearch + Streamlit

import os
import json
import xml.etree.ElementTree as ET
import streamlit as st
from elasticsearch import Elasticsearch
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

# 2. Elasticsearch setup
es = Elasticsearch("http://localhost:9200")
index_name = "xml_conditions"

# 3. Load and index sample XML data
xml_samples = [
    '''<conditionSet><and><field><name>prdType</name><value>DM,DS</value></field><field><name>current</name><value>USD</value></field><field><name>cps</name><operator>gtoe</operator><value>0.200</value></field><field><name>cps</name><operator>ltoe</operator><value>0.300</value></field></and></conditionSet>''',
    '''<conditionSet><and><field><name>prdType</name><value>DX</value></field><field><name>current</name><value>EUR</value></field><field><name>cps</name><operator>gtoe</operator><value>0.100</value></field><field><name>cps</name><operator>ltoe</operator><value>0.150</value></field></and></conditionSet>'''
]

# Index data
for i, xml in enumerate(xml_samples):
    flat_dict = parse_xml_to_flat_dict(xml)
    es.index(index=index_name, id=i, document=flat_dict)

# 4. Define LLM Prompt to interpret user input
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

# 5. LangGraph Nodes
def interpret_prompt_node(state):
    query = state["query"]
    structured = llm_chain.run(query)
    try:
        structured_dict = json.loads(structured)
    except:
        structured_dict = {}
    return { "query": query, "structured": structured_dict }

def search_elasticsearch_node(state):
    structured = state["structured"]
    must_clauses = [ {"match": {k: v}} for k, v in structured.items() ]
    body = { "query": { "bool": { "must": must_clauses } } }
    results = es.search(index=index_name, body=body)
    hits = [hit['_source'] for hit in results['hits']['hits']]
    return { **state, "results": hits }

def return_results_node(state):
    return { "final_results": state["results"] }

# 6. Build LangGraph
graph = StateGraph()
graph.add_node("InterpretPrompt", interpret_prompt_node)
graph.add_node("SearchElasticsearch", search_elasticsearch_node)
graph.add_node("ReturnResults", return_results_node)

graph.set_entry_point("InterpretPrompt")
graph.add_edge("InterpretPrompt", "SearchElasticsearch")
graph.add_edge("SearchElasticsearch", "ReturnResults")
graph.set_finish_point("ReturnResults")

runnable_graph = graph.compile()

# 7. Streamlit Interface
st.title("🔍 XML ConditionSet Matcher with Elasticsearch + AI")
user_query = st.text_input("Enter your search prompt:", "Find XMLs where product type is DM or DS, currency is USD, and cps between 0.2 and 0.3")

if st.button("Search"):
    with st.spinner("Running query through LangGraph..."):
        initial_state = { "query": user_query }
        result = runnable_graph.invoke(initial_state)

        st.subheader("🔎 Matched Results")
        for r in result["final_results"]:
            st.json(r)
