# Prototype: XML Matching AI using LangGraph + LangChain + FAISS

import os
import json
import xml.etree.ElementTree as ET
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

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

# 3. Load and process all XML files (demo with dummy data)
xml_samples = [
    '''<conditionSet><and><field><name>prdType</name><value>DM,DS</value></field><field><name>current</name><value>USD</value></field><field><name>cps</name><operator>gtoe</operator><value>0.200</value></field><field><name>cps</name><operator>ltoe</operator><value>0.300</value></field></and></conditionSet>''',
    '''<conditionSet><and><field><name>prdType</name><value>DX</value></field><field><name>current</name><value>EUR</value></field><field><name>cps</name><operator>gtoe</operator><value>0.100</value></field><field><name>cps</name><operator>ltoe</operator><value>0.150</value></field></and></conditionSet>'''
]

parsed_docs = [parse_xml_to_flat_dict(xml) for xml in xml_samples]
doc_texts = [create_text_representation(doc) for doc in parsed_docs]

# 4. Embed and store in FAISS
embedding = OpenAIEmbeddings()
faiss_store = FAISS.from_texts(doc_texts, embedding)

# 5. Define LLM Prompt to interpret user input to query
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

# 6. Process user query
user_query = "Find XMLs where product type is DM or DS, currency is USD, and cps between 0.2 and 0.3"
structured_output = llm_chain.run(user_query)
print("\nStructured Output (LLM):")
print(structured_output)

# 7. Semantic search in vector store
results = faiss_store.similarity_search(user_query, k=2)
print("\nTop Matching XML Documents:")
for i, res in enumerate(results):
    print(f"\nResult {i+1}:")
    print(res.page_content)

# Optional: Match logic to post-filter XMLs based on extracted logic from structured_output (can be added next)
