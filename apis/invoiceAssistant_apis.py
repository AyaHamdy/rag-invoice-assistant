from flask import Blueprint, request, jsonify
from rag.graph import rag_app

bp = Blueprint("apis", __name__)

@bp.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("query")
    if not query:
        return jsonify({"error": "Missing 'query' field"}), 400

    result = rag_app.invoke({"query": query, "attempts": 0})
    return jsonify({
        "query": query,
        "route": result["route"],
        "answer": result["answer"],
        "confident": result.get("confident"),
    })