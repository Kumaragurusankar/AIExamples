def generate_summary(json_obj):
    parts = []
    for rule in json_obj["rules"]:
        val = ", ".join(rule["value"]) if isinstance(rule["value"], list) else rule["value"]
        parts.append(f"{rule['field']} {rule['operator']} {val}")
    return " and ".join(parts)
