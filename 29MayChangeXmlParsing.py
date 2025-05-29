def xml_to_structured_json(xml_str: str, rate_id: Union[str, int], deal_id: Union[str, int]) -> Dict[str, Any]:
    root = ET.fromstring(xml_str)

    logic_node = root.find("./and") or root.find("./or")
    logic = logic_node.tag.upper() if logic_node is not None else "AND"

    rules: List[Dict[str, Union[str, float, List[str]]]] = []

    for field in logic_node.findall("./field"):
        name = field.findtext("name", "").strip()
        values = [v.text.strip() for v in field.findall("value") if v.text and v.text.strip()]
        operator = field.findtext("operator", "").strip().lower()

        if not operator:
            operator = "equal"

        operator_map = {
            "gtoe": "greater_than_or_equal",
            "ltoe": "less_than_or_equal",
            "equal": "equal"
        }
        norm_operator = operator_map.get(operator, operator)

        if len(values) > 1:
            value = values
            norm_operator = "in"
        else:
            # Attempt to convert single value to float if numeric
            raw_value = values[0] if values else ""
            try:
                value = float(raw_value) if raw_value.replace(".", "", 1).isdigit() else raw_value
            except:
                value = raw_value

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
