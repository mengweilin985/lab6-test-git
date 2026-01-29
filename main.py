import os
from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

nlp_pipeline = None  # DO NOT load at import time


@app.route("/", methods=["GET"])
def health():
    return "OK", 200


@app.route("/analyze", methods=["POST"])
def analyze_text():
    global nlp_pipeline

    if nlp_pipeline is None:
        print("Loading NLP model...")
        nlp_pipeline = pipeline("sentiment-analysis")

    data = request.json or {}
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    result = nlp_pipeline(text)
    return jsonify({"result": result})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
