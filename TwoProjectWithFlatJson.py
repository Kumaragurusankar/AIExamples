
# Project structure:
# xml_embedder_app/
# ├── main.py
# ├── vector_store.py
# ├── xml_utils.py
# ├── embeddings.py
# ├── models.py
# └── Dockerfile

# xml_search_app/
# ├── main.py
# ├── vector_store.py
# ├── pipeline.py
# ├── prompt_templates.py
# ├── models.py
# └── Dockerfile

# ---------------------- xml_embedder_app/main.py ----------------------
from fastapi import FastAPI
from pydantic import BaseModel
from xml_utils import xml_to_structured_json
from embeddings import get_embedding
from vector_store import FaissVectorStore
import json

app = FastAPI()
vector_store = FaissVectorStore(dimension=1536)

class XMLData(BaseModel):
    xml: str

@app.post("/embed")
def embed_xml(data: XMLData):
    parsed = xml_to_structured_json(data.xml)
    embedding = get_embedding(json.dumps(parsed))
    vector_store.add([embedding], [parsed])
    return {"message": "Embedding added successfully"}

# ---------------------- xml_search_app/main.py ----------------------
from fastapi import FastAPI
from pydantic import BaseModel
from pipeline import build_pipeline
from vector_store import FaissVectorStore
from embeddings import get_embedding

app = FastAPI()
vector_store = FaissVectorStore(dimension=1536)

pipeline = build_pipeline(vector_store, get_embedding)

class Query(BaseModel):
    prompt: str

@app.post("/search")
def search_xml(query: Query):
    result = pipeline.invoke({"query": query.prompt, "structured": {}, "results": [], "final_output": ""})
    return {"result": result.get("final_output", "No match found")}

# ---------------------- Shared Module Examples ----------------------
# xml_utils.py
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Union

def xml_to_structured_json(xml_str: str) -> Dict[str, Any]:
    root = ET.fromstring(xml_str)

    # Assumes <and> or <or> inside <conditionSet>
    logic_node = root.find("./and") or root.find("./or")
    logic = logic_node.tag.upper() if logic_node is not None else "AND"

    rules: List[Dict[str, Union[str, float, List[str]]]] = []

    for field in logic_node.findall("./field"):
        name = field.findtext("name")
        value = field.findtext("value")
        operator = field.findtext("operator")

        if not operator:
            operator = "equal"

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

        rules.append({
            "field": name,
            "operator": norm_operator,
            "value": value
        })

    return {
        "condition": logic,
        "rules": rules
    }

# embeddings.py
import os
from azure.identity import DefaultAzureCredential
from openai import AzureOpenAI

def get_embedding(text):
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

# vector_store.py (same for both apps)
import faiss
import numpy as np

class FaissVectorStore:
    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatL2(dimension)
        self.data = []

    def add(self, vectors, metadata):
        self.index.add(np.array(vectors).astype("float32"))
        self.data.extend(metadata)

    def search(self, vector, k=5):
        D, I = self.index.search(np.array([vector]).astype("float32"), k)
        return [self.data[i] for i in I[0] if i < len(self.data)]

# pipeline.py
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

## models.py

from pydantic import BaseModel
from typing import Any, Dict, List, Union

class XMLData(BaseModel):
    xml: str

class Query(BaseModel):
    prompt: str

class StructuredRule(BaseModel):
    field: str
    operator: str
    value: Union[str, float, List[str]]

class StructuredQuery(BaseModel):
    condition: str
    rules: List[StructuredRule]

class SearchResult(BaseModel):
    score: float
    metadata: Dict[str, Any]

from langchain.prompts import PromptTemplate

# This prompt interprets the user query into a structured search format
interpret_prompt = PromptTemplate.from_template("""
You are an expert at converting user queries into structured search criteria.
Given the following query:
"{query}"
Return a structured JSON object with:
- condition: "AND" or "OR"
- rules: a list of {{"field": ..., "operator": ..., "value": ...}}

Example output:
{{
  "condition": "AND",
  "rules": [
    {{"field": "age", "operator": "greater_than_or_equal", "value": 30}},
    {{"field": "country", "operator": "equal", "value": "USA"}}
  ]
}}
""")

# This prompt refines the top search results into a final natural language answer
refine_prompt = PromptTemplate.from_template("""
Given the original query:
"{query}"

And the following candidate results:
{candidates}

Please write a clear and concise natural language answer summarizing the most relevant result(s).
""")
----------------------------

# xml_search_app/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pipeline import build_pipeline
from vector_store import FaissVectorStore
from embeddings import get_embedding

app = FastAPI()
vector_store = FaissVectorStore(dimension=1536, index_path="shared_faiss_index.idx")

# Optional: Reload index on startup to ensure it's synced
vector_store.reload_index()

pipeline = build_pipeline(vector_store, get_embedding)

class Query(BaseModel):
    prompt: str

@app.post("/search")
def search_xml(query: Query):
    try:
        # Optional: reload index before every search to catch new embeddings
        vector_store.reload_index()
        result = pipeline.invoke({"query": query.prompt, "structured": {}, "results": [], "final_output": ""})
        return {"result": result.get("final_output", "No match found")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

-------------------------------

# xml_embedder_app/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from xml_utils import xml_to_structured_json
from embeddings import get_embedding
from vector_store import FaissVectorStore
import json

app = FastAPI()
vector_store = FaissVectorStore(dimension=1536, index_path="shared_faiss_index.idx")

class XMLData(BaseModel):
    xml: str

@app.post("/embed")
def embed_xml(data: XMLData):
    try:
        parsed = xml_to_structured_json(data.xml)
        embedding = get_embedding(json.dumps(parsed))
        vector_store.add([embedding], [parsed])
        return {"message": "Embedding added and saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

-------------------


# vector_store.py

import faiss
import numpy as np
import threading
import os

class FaissVectorStore:
    def __init__(self, dimension: int, index_path: str = "faiss_index.idx"):
        self.dimension = dimension
        self.index_path = index_path
        self.lock = threading.Lock()

        if os.path.exists(self.index_path):
            print(f"Loading existing FAISS index from {self.index_path}")
            self.index = faiss.read_index(self.index_path)
        else:
            print("Creating new FAISS index")
            self.index = faiss.IndexFlatL2(dimension)

        # Metadata store (in-memory) — this should eventually go to a DB!
        self.data = []

    def save_index(self):
        with self.lock:
            faiss.write_index(self.index, self.index_path)
            print(f"FAISS index saved to {self.index_path}")

    def add(self, vectors, metadata):
        with self.lock:
            self.index.add(np.array(vectors).astype("float32"))
            self.data.extend(metadata)
            self.save_index()

    def search(self, vector, k=5):
        with self.lock:
            if self.index.ntotal == 0:
                return []

            D, I = self.index.search(np.array([vector]).astype("float32"), k)
            return [self.data[i] for i in I[0] if i < len(self.data)]

    def reload_index(self):
        with self.lock:
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
                print(f"FAISS index reloaded from {self.index_path}")
            else:
                print(f"No FAISS index file found at {self.index_path}")

-------------------------

#code for 10000 records

class FaissVectorStore:
    def __init__(self, dimension: int, index_file: str = "faiss.index", nlist: int = 100):
        self.dimension = dimension
        self.index_file = index_file
        
        # Try loading the index file, or create a new index
        try:
            self.index = faiss.read_index(self.index_file)
        except:
            # Use IndexIVFFlat for large datasets (default is 100 partitions)
            quantizer = faiss.IndexFlatL2(self.dimension)  # The quantizer used for the IVF index
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            self.index.train(np.random.random((1000, self.dimension)).astype('float32'))  # Train the index with random data for now
            self.index.add(np.random.random((1000, self.dimension)).astype('float32'))  # Add some initial data
        
        self.metadata_store: Dict[int, Dict[str, Any]] = {}
        self.current_id = 0

    def add(self, vectors: List[List[float]], metadata: List[Dict[str, Any]]):
        vectors = np.array(vectors).astype('float32')
        self.index.add(vectors)  # Add vectors to FAISS index
        for meta in metadata:
            self.metadata_store[self.current_id] = meta
            self.current_id += 1

    def save_index(self):
        faiss.write_index(self.index, self.index_file)  # Save index to disk
        print(f"Index saved to {self.index_file}")

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        query_vector = np.array([query_vector]).astype('float32')
        D, I = self.index.search(query_vector, top_k)  # Perform search
        results = []
        for idx in I[0]:
            if idx in self.metadata_store:
                results.append(self.metadata_store[idx])
        return results




