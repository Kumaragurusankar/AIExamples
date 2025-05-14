from langgraph.graph import StateGraph
from langgraph.graph.state import State
from langchain.chains import LLMChain
from langchain.llms import VertexAI
from prompt_templates import interpret_prompt, refine_prompt
import json

# Initialize Vertex AI LLM (e.g., Gemini or PaLM)
llm = VertexAI(
    model_name="gemini-pro",  # or use "text-bison" for PaLM
    temperature=0.0
)

llm_chain = LLMChain(llm=llm, prompt=interpret_prompt)
refine_chain = LLMChain(llm=llm, prompt=refine_prompt)

class AppState(State):
    query: str
    structured: dict
    results: list
    final_output: str

def interpret_node(state):
    structured = llm_chain.run(state["query"])
    try:
        structured_dict = json.loads(structured)
    except:
        structured_dict = {}
    return {**state, "structured": structured_dict}

def search_node(state, vector_store, get_embedding_fn):
    emb = get_embedding_fn(json.dumps(state["structured"]))
    results = vector_store.search(emb)
    return {**state, "results": results}

def refine_node(state):
    response = refine_chain.run({
        "query": state["query"],
        "candidates": json.dumps(state["results"], indent=2)
    })
    return {**state, "final_output": response}

def build_pipeline(vector_store, get_embedding_fn):
    graph = StateGraph()
    graph.add_node("Interpret", interpret_node)
    graph.add_node("Search", lambda s: search_node(s, vector_store, get_embedding_fn))
    graph.add_node("Refine", refine_node)
    graph.set_entry_point("Interpret")
    graph.add_edge("Interpret", "Search")
    graph.add_edge("Search", "Refine")
    graph.set_finish_point("Refine")
    return graph.compile(AppState)
