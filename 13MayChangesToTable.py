# ---------------------- xml_embedder_app/main.py ----------------------
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from xml_utils import xml_to_structured_json
from embeddings import get_embedding
from vector_store import FaissVectorStore
import json
import psycopg2
import os

app = FastAPI()
vector_store = FaissVectorStore(dimension=1536, index_path="shared_faiss_index.idx")

# Database connection configuration (can be updated as needed)
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("DB_NAME", "mydb"),
    "user": os.getenv("DB_USER", "user"),
    "password": os.getenv("DB_PASSWORD", "password")
}

@app.post("/embed")
def embed_all_from_db():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT condition_set, rate_id, deal_id FROM rates")
        rows = cursor.fetchall()

        for condition_set, rate_id, deal_id in rows:
            parsed = xml_to_structured_json(condition_set, rate_id, deal_id)
            embedding = get_embedding(json.dumps(parsed))
            vector_store.add([embedding], [parsed])

        cursor.close()
        conn.close()

        return {"message": f"Processed {len(rows)} records and saved embeddings."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------- xml_utils.py ----------------------
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Union

def xml_to_structured_json(xml_str: str, rate_id: Union[str, int], deal_id: Union[str, int]) -> Dict[str, Any]:
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
        "rate_id": rate_id,
        "deal_id": deal_id,
        "condition": logic,
        "rules": rules
    }
