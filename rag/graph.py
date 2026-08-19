from langgraph.graph import StateGraph, END
from rag.nodes import route_node, structured_node, semantic_node, check_node, RAGState

def route_decision(state: RAGState) -> str:
    return "structured" if state["route"] == "structured" else "semantic"

def confidence_decision(state: RAGState) -> str:
    if state["confident"] or state["attempts"] >= 2:
        return "end"
    return "retry"

def build_graph():
    graph = StateGraph(RAGState)
    graph.add_node("route", route_node)
    graph.add_node("structured", structured_node)
    graph.add_node("semantic", semantic_node)
    graph.add_node("check", check_node)
    graph.set_entry_point("route")
    graph.add_conditional_edges("route", route_decision, {"structured": "structured", "semantic": "semantic"})
    graph.add_edge("structured", "check")
    graph.add_edge("semantic", "check")
    graph.add_conditional_edges("check", confidence_decision, {"end": END, "retry": "semantic"})
    return graph.compile()

rag_app = build_graph()