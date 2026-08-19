from typing import TypedDict, Optional
import json
import ollama

class RAGState(TypedDict):
    query: str
    route: Optional[str]
    answer: Optional[str]
    confident: Optional[bool]
    attempts: int

with open("data/invoices_structured.json") as f:
    invoices = json.load(f)

def route_node(state: RAGState) -> RAGState:
    prompt = f"""Classify as "structured" or "semantic".
"structured" = comparing/aggregating numbers across invoices.
"semantic" = about content/meaning within an invoice.

Question: {state['query']}
Respond with ONLY one word."""
    r = ollama.chat(model="llama3.2:3b", messages=[{"role": "user", "content": prompt}], options={"temperature": 0})
    raw = r["message"]["content"].strip().lower()
    state["route"] = "structured" if "structured" in raw else "semantic"
    return state

def structured_node(state: RAGState) -> RAGState:
    valid = [i for i in invoices if i.get("total") is not None]
    top = max(valid, key=lambda i: i["total"])
    state["answer"] = f"{top['customer']} — {top['filename']} — ${top['total']}"
    return state

def semantic_node(state: RAGState) -> RAGState:
    context = "\n\n".join(i["full_text"][:500] for i in invoices)
    prompt = f"Answer using this context:\n{context}\n\nQuestion: {state['query']}"
    r = ollama.chat(model="llama3.2:3b", messages=[{"role": "user", "content": prompt}], options={"temperature": 0})
    state["answer"] = r["message"]["content"]
    return state

def check_node(state: RAGState) -> RAGState:
    if state["route"] == "structured":
        state["confident"] = True
        state["attempts"] = state.get("attempts", 0) + 1
        return state
    prompt = f"""Question: {state['query']}
Answer: {state['answer']}

Does this answer actually address the question directly? Respond ONLY yes or no."""
    r = ollama.chat(model="llama3.2:3b", messages=[{"role": "user", "content": prompt}], options={"temperature": 0})
    state["confident"] = "yes" in r["message"]["content"].strip().lower()
    state["attempts"] = state.get("attempts", 0) + 1
    return state