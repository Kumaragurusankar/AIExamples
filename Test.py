# 6. LangGraph Nodes
def interpret_prompt_node(state):
    query = state.query
    structured = llm_chain.run(query)
    try:
        structured_dict = json.loads(structured)
    except:
        structured_dict = {}
    return State(query=query, structured=structured_dict, results=[])

def search_faiss_node(state):
    query_text = json.dumps(state.structured)
    query_embedding = get_embedding(query_text)
    D, I = index.search(np.array([query_embedding]).astype('float32'), k=5)
    matches = [flat_texts[i] for i in I[0]]
    return State(query=state.query, structured=state.structured, results=matches)

def return_results_node(state):
    return {"final_results": state.results}

# 7. Define LangGraph State Class
class AppState(State):
    query: str
    structured: dict
    results: list

# 8. Build LangGraph
graph = StateGraph(input_state=AppState, output_state=dict)
graph.add_node("InterpretPrompt", interpret_prompt_node)
graph.add_node("SearchFAISS", search_faiss_node)
graph.add_node("ReturnResults", return_results_node)

graph.set_entry_point("InterpretPrompt")
graph.add_edge("InterpretPrompt", "SearchFAISS")
graph.add_edge("SearchFAISS", "ReturnResults")
graph.set_finish_point("ReturnResults")

runnable_graph = graph.compile()
