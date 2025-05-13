# Project layout:
# - xml_embedder_app/
# - xml_search_app/
# - shared/embedder.py (reusable embedding logic)

# --------------------- shared/embedder.py ---------------------
import os
from azure.identity import DefaultAzureCredential
from openai import AzureOpenAI

def get_embedding(text: str):
    client = AzureOpenAI(
        azure_endpoint=os.environ["OPENAI_API_BASE"],
        api_version=os.environ["OPENAI_API_VERSION"],
        azure_ad_token_provider=DefaultAzureCredential()
    )
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

# --------------------- xml_embedder_app/main.py ---------------------
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from xml_utils import xml_to_structured_json
from vector_store import FaissVectorStore
from shared.embedder import get_embedding
import json

app = FastAPI()
vector_store = FaissVectorStore(dimension=1536, index_path="shared_faiss_index.ivf")

class XMLData(BaseModel):
    xml: str

@app.post("/embed")
def embed_xml(data: XMLData):
    try:
        parsed = xml_to_structured_json(data.xml)
        embedding = get_embedding(json.dumps(parsed))
        vector_store.add([embedding], [parsed])
        return {"message": "Embedding added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --------------------- xml_embedder_app/vector_store.py ---------------------
import faiss
import numpy as np
import os
import threading
import json

class FaissVectorStore:
    def __init__(self, dimension: int, index_path: str, nlist: int = 100):
        self.dimension = dimension
        self.index_path = index_path
        self.lock = threading.Lock()
        self.metadata_path = index_path + ".meta.json"

        self.metadata_store = {}
        self.current_id = 0

        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, "r") as f:
                    self.metadata_store = json.load(f)
                self.current_id = max(map(int, self.metadata_store.keys()), default=-1) + 1
        else:
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            # You should train with representative data before production use
            train_data = np.random.random((1000, dimension)).astype('float32')
            self.index.train(train_data)

    def add(self, vectors, metadata):
        with self.lock:
            self.index.add(np.array(vectors).astype('float32'))
            for meta in metadata:
                self.metadata_store[self.current_id] = meta
                self.current_id += 1
            self.save()

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata_store, f)

    def search(self, vector, k=5):
        with self.lock:
            self.index.nprobe = 10
            D, I = self.index.search(np.array([vector]).astype('float32'), k)
            return [self.metadata_store[str(i)] for i in I[0] if str(i) in self.metadata_store]

# --------------------- xml_embedder_app/xml_utils.py ---------------------
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Union

def xml_to_structured_json(xml_str: str) -> Dict[str, Any]:
    root = ET.fromstring(xml_str)
    logic_node = root.find("./and") or root.find("./or")
    logic = logic_node.tag.upper() if logic_node is not None else "AND"

    rules: List[Dict[str, Union[str, float, List[str]]]] = []

    for field in logic_node.findall("./field"):
        name = field.findtext("name")
        value = field.findtext("value")
        operator = field.findtext("operator") or "equal"

        operator_map = {
            "gtoe": "greater_than_or_equal",
            "ltoe": "less_than_or_equal",
            "equal": "equal"
        }
        norm_operator = operator_map.get(operator, operator)

        if value and "," in value:
            value = [v.strip() for v in value.split(",")]
            norm_operator = "in"

        if isinstance(value, str) and value.replace('.', '', 1).isdigit():
            value = float(value)

        rules.append({"field": name, "operator": norm_operator, "value": value})

    return {"condition": logic, "rules": rules}

# --------------------- xml_search_app/main.py ---------------------
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pipeline import build_pipeline
from vector_store import FaissVectorStore
from shared.embedder import get_embedding

app = FastAPI()
vector_store = FaissVectorStore(dimension=1536, index_path="shared_faiss_index.ivf")
vector_store.index.nprobe = 10

pipeline = build_pipeline(vector_store, get_embedding)

class Query(BaseModel):
    prompt: str

@app.post("/search")
def search_xml(query: Query):
    try:
        result = pipeline.invoke({"query": query.prompt, "structured": {}, "results": [], "final_output": ""})
        return {"result": result.get("final_output", "No match found")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --------------------- xml_search_app/vector_store.py ---------------------
# (Same as embedder version)

# --------------------- xml_search_app/pipeline.py ---------------------
from langgraph.graph import StateGraph
from langgraph.graph.state import State
from langchain.chains import LLMChain
from langchain.chat_models import AzureChatOpenAI
from prompt_templates import interpret_prompt, refine_prompt
import os
import json
from azure.identity import DefaultAzureCredential

llm = AzureChatOpenAI(
    azure_endpoint=os.environ["OPENAI_API_BASE"],
    azure_deployment="gpt-35-deployment",
    api_version=os.environ["OPENAI_API_VERSION"],
    azure_ad_token_provider=DefaultAzureCredential(),
    temperature=0
)

llm_chain = LLMChain(llm=llm, prompt=interpret_prompt)
refine_chain = LLMChain(llm=llm, prompt=refine_prompt)

class AppState(State):
    query: str
    structured: dict
    results: list
    final_output: str

def interpret_node(state):
    structured = llm_chain.run(state["query"])
    try:
        structured_dict = json.loads(structured)
    except:
        structured_dict = {}
    return {**state, "structured": structured_dict}

def search_node(state, vector_store, get_embedding_fn):
    emb = get_embedding_fn(json.dumps(state["structured"]))
    results = vector_store.search(emb)
    return {**state, "results": results}

def refine_node(state):
    response = refine_chain.run({
        "query": state["query"],
        "candidates": json.dumps(state["results"], indent=2)
    })
    return {**state, "final_output": response}

def build_pipeline(vector_store, get_embedding_fn):
    graph = StateGraph()
    graph.add_node("Interpret", interpret_node)
    graph.add_node("Search", lambda s: search_node(s, vector_store, get_embedding_fn))
    graph.add_node("Refine", refine_node)
    graph.set_entry_point("Interpret")
    graph.add_edge("Interpret", "Search")
    graph.add_edge("Search", "Refine")
    graph.set_finish_point("Refine")
    return graph.compile(AppState)

# --------------------- prompt_templates.py ---------------------
from langchain.prompts import PromptTemplate

interpret_prompt = PromptTemplate.from_template("""
You are an expert at converting user queries into structured search criteria.
Given the following query:
"{query}"
Return a structured JSON object with:
- condition: "AND" or "OR"
- rules: a list of {{"field": ..., "operator": ..., "value": ...}}
""")

refine_prompt = PromptTemplate.from_template("""
Given the original query:
"{query}"

And the following candidate results:
{candidates}

Please write a clear and concise natural language answer summarizing the most relevant result(s).
""")
