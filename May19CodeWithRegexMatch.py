# prompt_normalizer.py
import re

FIELD_SYNONYMS = {
    "prod type": "prdType",
    "product type": "prdType",
    "prodtype": "prdType",
    "currency": "current",
    "curr": "current",
    "country": "country",
    "nation": "country"
}

def normalize_field_name(name: str) -> str:
    return FIELD_SYNONYMS.get(name.strip().lower(), name)

def extract_conditions_from_prompt(prompt: str) -> dict:
    conditions = {}
    pattern = re.compile(r"(\b[\w\s]+\b)\s+(is|=|equals|:)\s+([A-Za-z0-9]+)", re.IGNORECASE)

    for match in pattern.findall(prompt):
        raw_field, _, raw_value = match
        norm_field = normalize_field_name(raw_field)
        conditions[norm_field] = raw_value.upper()

    return conditions


# Update xml_search_app/main.py to use the prompt parser
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pipeline import build_pipeline
from vector_store import FaissVectorStore
from embeddings import get_embedding
from prompt_normalizer import extract_conditions_from_prompt

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
        conditions = extract_conditions_from_prompt(query.prompt)
        result = pipeline.invoke({
            "query": query.prompt,
            "structured": conditions,
            "results": [],
            "final_output": ""
        })
        return {"result": result.get("final_output", "No match found")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
