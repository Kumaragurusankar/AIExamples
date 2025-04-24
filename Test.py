def interpret_prompt_node(state):
    query = state["query"]
    structured = llm_chain.run(query)
    try:
        structured_dict = json.loads(structured)
    except:
        structured_dict = {}
    return {
        "query": query,
        "structured": structured_dict,
        "results": []
    }

def search_faiss_node(state):
    structured_dict = state["structured"]
    query_text = json.dumps(structured_dict)
    query_embedding = get_embedding(query_text)
    D, I = index.search(np.array([query_embedding]).astype('float32'), k=5)
    matches = [flat_texts[i] for i in I[0]]
    return {
        "query": state["query"],
        "structured": state["structured"],
        "results": matches
    }

def return_results_node(state):
    return {
        "final_results": state["results"]
    }