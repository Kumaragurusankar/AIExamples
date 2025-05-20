# pipeline.py
from typing import Dict, Any
from prompt_parser import parse_user_prompt
from langchain_core.runnables import RunnableLambda

# This assumes you already have your FAISS vector store and embedding function

def build_pipeline(vector_store, get_embedding):
    def process(inputs: Dict[str, Any]) -> Dict[str, Any]:
        prompt = inputs["query"]

        # Step 1: Parse prompt into structured rules
        user_rules = parse_user_prompt(prompt)

        # Step 2: Embed the prompt
        embedded = get_embedding(prompt)

        # Step 3: Perform vector search
        matches = vector_store.search(embedded, k=5)

        # Step 4: Compare user_rules with stored metadata rules (optional)
        final_result = []
        for match in matches:
            meta = match["metadata"]
            if is_rule_match(user_rules, meta.get("rules", [])):
                final_result.append(meta)

        inputs["final_output"] = final_result
        return inputs

    return RunnableLambda(process)


def is_rule_match(user_rules: list, stored_rules: list) -> bool:
    # Simple rule comparator; can be enhanced
    for ur in user_rules:
        for sr in stored_rules:
            if ur["field"] == sr["field"]:
                if ur["operator"] == "in" and isinstance(sr["value"], list):
                    if set(ur["value"]).intersection(set(sr["value"])):
                        return True
                elif ur["operator"] == sr["operator"] and ur["value"] == sr["value"]:
                    return True
    return False
