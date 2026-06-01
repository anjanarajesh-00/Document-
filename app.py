"""
app.py
------
Flask REST API — deploy your trained model as a web service.

Endpoints:
    POST /predict          — classify a single document
    POST /predict/batch    — classify multiple documents
    GET  /health           — health check
    GET  /classes          — list all category names

Start the server:
    python app.py
    # → Running on http://0.0.0.0:5000

Example calls:
    # Single prediction
    curl -X POST http://localhost:5000/predict \
         -H "Content-Type: application/json" \
         -d '{"text": "NASA launches new satellite"}'

    # Batch prediction
    curl -X POST http://localhost:5000/predict/batch \
         -H "Content-Type: application/json" \
         -d '{"texts": ["NASA launches new satellite", "Game 7 tonight"]}'

PRODUCTION NOTES:
    - Replace Flask dev server with gunicorn: gunicorn -w 4 app:app
    - Add authentication (API key header) before exposing publicly
    - Add rate limiting: pip install flask-limiter
    - Add logging to a file / cloud service (CloudWatch, Datadog, etc.)
    - Use Docker: see Dockerfile example at the bottom of this file
"""

import os
import sys
import logging
import time
import joblib
from flask import Flask, request, jsonify

sys.path.insert(0, os.path.dirname(__file__))
from predict import load_pipeline, predict_one, predict_many

# ─── Setup ────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load model once at startup (not per-request)
MODEL_DIR = os.environ.get("MODEL_DIR", "models")
_bundle = None

def get_bundle():
    global _bundle
    if _bundle is None:
        logger.info("Loading pipeline from '%s'...", MODEL_DIR)
        _bundle = load_pipeline(MODEL_DIR)
        logger.info("Pipeline loaded. Classes: %s", _bundle["class_names"])
    return _bundle


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint — useful for load balancers / k8s probes."""
    try:
        bundle = get_bundle()
        return jsonify({
            "status": "ok",
            "classes": bundle["class_names"],
            "num_classes": len(bundle["class_names"]),
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/classes", methods=["GET"])
def classes():
    """Return all category names."""
    bundle = get_bundle()
    return jsonify({"classes": bundle["class_names"]})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Classify a single document.

    Request body:
        { "text": "Your document text here" }

    Response:
        {
            "category": "sci.space",
            "confidence": 0.94,
            "all_probabilities": { "sci.space": 0.94, ... },
            "latency_ms": 12.3
        }
    """
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Request body must contain a 'text' field"}), 400

    text = str(data["text"]).strip()
    if not text:
        return jsonify({"error": "'text' field must not be empty"}), 400
    if len(text) > 100_000:
        return jsonify({"error": "Text too long (max 100,000 chars)"}), 400

    t0 = time.perf_counter()
    try:
        result = predict_one(text, get_bundle())
    except Exception as e:
        logger.exception("Prediction error")
        return jsonify({"error": str(e)}), 500

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    return jsonify({
        "category":          result["category"],
        "confidence":        result["confidence"],
        "all_probabilities": result["all_probs"],
        "latency_ms":        latency_ms,
    })


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """
    Classify multiple documents at once.

    Request body:
        { "texts": ["Text one...", "Text two..."] }

    Response:
        {
            "predictions": [
                { "category": "sci.space", "confidence": 0.94, ... },
                { "category": "rec.sport.hockey", "confidence": 0.87, ... }
            ],
            "count": 2,
            "latency_ms": 34.1
        }
    """
    data = request.get_json(silent=True)
    if not data or "texts" not in data:
        return jsonify({"error": "Request body must contain a 'texts' list"}), 400

    texts = data["texts"]
    if not isinstance(texts, list):
        return jsonify({"error": "'texts' must be a list"}), 400
    if len(texts) > 1000:
        return jsonify({"error": "Batch size limit is 1000 documents"}), 400

    t0 = time.perf_counter()
    try:
        results = predict_many(texts, get_bundle())
    except Exception as e:
        logger.exception("Batch prediction error")
        return jsonify({"error": str(e)}), 500

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    return jsonify({
        "predictions": [
            {
                "category":          r["category"],
                "confidence":        r["confidence"],
                "all_probabilities": r["all_probs"],
            }
            for r in results
        ],
        "count":      len(results),
        "latency_ms": latency_ms,
    })


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    print(f"\n🚀  Document Classifier API")
    print(f"   http://localhost:{port}/health\n")
    # Pre-load model before first request
    get_bundle()
    app.run(host="0.0.0.0", port=port, debug=debug)


# ─── DOCKERFILE (copy this to a file named 'Dockerfile') ─────────────────────
"""
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

COPY . .

# Train on startup if no model exists
# RUN python train.py

ENV PORT=5000
ENV MODEL_DIR=models
EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
"""
