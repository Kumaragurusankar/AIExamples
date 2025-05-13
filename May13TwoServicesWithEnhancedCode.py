# xml_embedder_app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from xml_utils import xml_to_structured_json
from embeddings import get_embedding
from vector_store import FaissVectorStore
import json

app = FastAPI()
vector_store = FaissVectorStore(dimension=1536, index_path="faiss.index", metadata_path="metadata.json")

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


# xml_search_app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pipeline import build_pipeline
from vector_store import FaissVectorStore
from embeddings import get_embedding

app = FastAPI()
vector_store = FaissVectorStore(dimension=1536, index_path="faiss.index", metadata_path="metadata.json")

@app.on_event("startup")
def reload_index_on_start():
    vector_store.reload_index()

pipeline = build_pipeline(vector_store, get_embedding)

class Query(BaseModel):
    prompt: str

@app.post("/search")
def search_xml(query: Query):
    try:
        vector_store.reload_index()
        result = pipeline.invoke({"query": query.prompt, "structured": {}, "results": [], "final_output": ""})
        return {"result": result.get("final_output", "No match found")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# vector_store.py
import faiss
import numpy as np
import os
import json
from threading import Lock
from typing import List, Dict, Any

class FaissVectorStore:
    def __init__(self, dimension: int, index_path: str = "faiss.index", metadata_path: str = "metadata.json"):
        self.dimension = dimension
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.lock = Lock()

        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))

        self.metadata_store: Dict[int, Dict[str, Any]] = {}
        self.current_id = 0
        self._load_metadata()

    def _load_metadata(self):
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r") as f:
                self.metadata_store = json.load(f)
                self.metadata_store = {int(k): v for k, v in self.metadata_store.items()}
                self.current_id = max(self.metadata_store.keys(), default=-1) + 1

    def _save_metadata(self):
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata_store, f)

    def add(self, vectors: List[List[float]], metadata: List[Dict[str, Any]]):
        with self.lock:
            ids = list(range(self.current_id, self.current_id + len(vectors)))
            self.index.add_with_ids(np.array(vectors, dtype="float32"), np.array(ids))
            for i, meta in zip(ids, metadata):
                self.metadata_store[i] = meta
            self.current_id += len(vectors)
            self._save_index()
            self._save_metadata()

    def search(self, vector: List[float], k: int = 5) -> List[Dict[str, Any]]:
        with self.lock:
            if self.index.ntotal == 0:
                return []
            D, I = self.index.search(np.array([vector], dtype="float32"), k)
            results = []
            for dist, idx in zip(D[0], I[0]):
                if idx in self.metadata_store:
                    results.append({
                        "score": float(dist),
                        "metadata": self.metadata_store[idx]
                    })
            return results

    def _save_index(self):
        faiss.write_index(self.index, self.index_path)

    def reload_index(self):
        with self.lock:
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
            self._load_metadata()


# embedder.py
import json
from embeddings import get_embedding
from xml_utils import xml_to_structured_json
from vector_store import FaissVectorStore

vector_store = FaissVectorStore(dimension=1536, index_path="faiss.index", metadata_path="metadata.json")

def embed_xml_string(xml_str: str):
    parsed = xml_to_structured_json(xml_str)
    embedding = get_embedding(json.dumps(parsed))
    vector_store.add([embedding], [parsed])
    print("Embedding generated and stored successfully.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python embedder.py '<xml_string>'")
    else:
        embed_xml_string(sys.argv[1])
