
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from corpus import documents

app = Flask(__name__)

model = SentenceTransformer("all-MiniLM-L6-v2")

doc_embeddings = model.encode(documents, convert_to_numpy=True)
print(f"Embeddings loaded — {len(documents)} documents indexed.")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/search", methods=["POST"])
def search():
    # Parse query and top_k from JSON body
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    top_k = data.get("top_k", 5)

    if not query:
        return jsonify([])

    # Encode query and compute cosine similarity against all documents
    query_embedding = model.encode([query], convert_to_numpy=True)
    scores = cosine_similarity(query_embedding, doc_embeddings)[0]

    # Build sorted results, limited to top_k
    ranked = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    results = [{"text": text, "score": float(score)} for text, score in ranked]
    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
