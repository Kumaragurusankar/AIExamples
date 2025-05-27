# backend/main.py

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import pandas as pd
import oracledb
import os

app = FastAPI()

# Allow CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with actual domain in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Oracle DB config
DB_CONFIG = {
    "user": "your_username",
    "password": "your_password",
    "dsn": "your_host:1521/your_service_name"
}

# Endpoint to get rate cards by rateId
@app.get("/rate-cards")
def get_rate_cards(rate_ids: List[str] = Query(...)):
    try:
        with oracledb.connect(**DB_CONFIG) as conn:
            cursor = conn.cursor()
            placeholders = ",".join([f":{i+1}" for i in range(len(rate_ids))])
            query = f"SELECT * FROM rate_cards WHERE rateId IN ({placeholders})"
            cursor.execute(query, rate_ids)
            columns = [col[0] for col in cursor.description]
            rows = cursor.fetchall()
            data = [dict(zip(columns, row)) for row in rows]
            return {"data": data}
    except Exception as e:
        return {"error": str(e)}
