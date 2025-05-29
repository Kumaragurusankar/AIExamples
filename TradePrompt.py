def extract_prompt_metadata(user_prompt: str) -> dict:
    # Simulated Gemini-like logic for now
    prompt = user_prompt.lower()
    if "trade id" in prompt:
        trade_id = prompt.split("trade id")[-1].strip().strip(":").split()[0].upper()
        return {
            "intent": "get_rate_card_by_trade_id",
            "trade_id": trade_id,
            "date_from": None,
            "date_to": None,
            "plain_query": f"rate card for trade {trade_id}"
        }
    elif "between" in prompt and "may" in prompt:
        return {
            "intent": "get_rate_card_by_date_range",
            "trade_id": None,
            "date_from": "2025-05-01",
            "date_to": "2025-05-02",
            "plain_query": "rate card for trades between may 1 and may 2"
        }
    else:
        return {
            "intent": "semantic_similarity_search",
            "trade_id": None,
            "date_from": None,
            "date_to": None,
            "plain_query": user_prompt
        }
