"""
Flask API for Automatic Question Generation
Exposes the NLP engine via REST endpoints.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS

from nlp_engine.question_generator import generate_questions
from nlp_engine.question_ranker import rank_questions, filter_questions
from nlp_engine.difficulty import classify_difficulty, get_difficulty_stats
from nlp_engine.blooms_taxonomy import classify_blooms, get_blooms_stats, BloomsLevel
from nlp_engine.evaluator import evaluate_on_squad, generate_report
from nlp_engine.dataset import get_default_dataset

import random
import json
import os
import tempfile
import fitz  # PyMuPDF

app = Flask(__name__)
CORS(app)


@app.route("/api/health", methods=["GET"])
def health():
    """Health check."""
    return jsonify({"status": "ok", "service": "AQG Engine"})


@app.route("/api/generate", methods=["POST"])
def generate():
    """
    Generate questions from input text.

    Request body:
        { "text": "...", "max_questions": 10 }

    Response:
        { "questions": [...], "stats": {...} }
    """
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data["text"].strip()
    if len(text) < 20:
        return jsonify({"error": "Text too short. Provide at least a few sentences."}), 400

    max_q = data.get("max_questions", 10)

    # Generate questions
    raw_questions = generate_questions(text, max_questions=max_q * 2)
    filtered = filter_questions(raw_questions)
    ranked = rank_questions(filtered, top_k=max_q)

    # Enrich with difficulty and Bloom's taxonomy
    results = []
    for q in ranked:
        q_dict = q.to_dict()
        q_dict["difficulty"] = classify_difficulty(q)
        q_dict["blooms_level"] = classify_blooms(q)
        q_dict["blooms_color"] = BloomsLevel.COLORS.get(q_dict["blooms_level"], "#666")
        results.append(q_dict)

    # Stats
    stats = {
        "total_generated": len(raw_questions),
        "after_filtering": len(filtered),
        "returned": len(results),
        "difficulty": get_difficulty_stats(ranked),
        "blooms": get_blooms_stats(ranked),
    }

    return jsonify({
        "questions": results,
        "stats": stats,
    })


@app.route("/api/quiz", methods=["POST"])
def quiz():
    """
    Generate a quiz from input text.
    🔥 Innovation #3 — Interactive quiz with answers, difficulty, and scoring.

    Request body:
        { "text": "...", "num_questions": 5, "difficulty": "all" }

    Response:
        { "quiz": [...] }
    """
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data["text"].strip()
    num_q = data.get("num_questions", 5)
    difficulty_filter = data.get("difficulty", "all")

    # Generate more questions than needed for filtering
    raw_questions = generate_questions(text, max_questions=num_q * 3)
    filtered = filter_questions(raw_questions)
    ranked = rank_questions(filtered, top_k=num_q * 2)

    # Apply difficulty filter
    quiz_items = []
    for q in ranked:
        difficulty = classify_difficulty(q)

        if difficulty_filter != "all" and difficulty != difficulty_filter:
            continue

        quiz_items.append({
            "id": len(quiz_items) + 1,
            "question": q.question,
            "answer": q.answer,
            "difficulty": difficulty,
            "blooms_level": classify_blooms(q),
            "source_sentence": q.source_sentence,
            "question_word": q.question_word,
        })

        if len(quiz_items) >= num_q:
            break

    return jsonify({
        "quiz": quiz_items,
        "total_available": len(ranked),
        "difficulty_filter": difficulty_filter,
    })


@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    """
    Run evaluation on SQuAD dataset.

    Request body:
        { "sample_size": 20, "split": "dev" }

    Response:
        { "metrics": {...}, "report": "..." }
    """
    data = request.get_json() or {}
    sample_size = data.get("sample_size", 20)
    split = data.get("split", "dev")

    # Cap at reasonable size for API requests
    sample_size = min(sample_size, 100)

    results = evaluate_on_squad(sample_size=sample_size, split=split)
    report = generate_report(results)

    return jsonify({
        "metrics": results["metrics"],
        "report": report,
        "sample_results": results.get("sample_results", [])[:5],
    })


@app.route("/api/stats", methods=["GET"])
def stats():
    """Get dataset statistics."""
    try:
        dataset = get_default_dataset()
        dataset.load()
        train_stats = dataset.get_stats("train")
        dev_stats = dataset.get_stats("dev")
        return jsonify({
            "train": train_stats,
            "dev": dev_stats,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/sample", methods=["GET"])
def sample():
    """Get a random sample context from the dataset for testing."""
    try:
        dataset = get_default_dataset()
        dataset.load()
        contexts = dataset.extract_contexts(split="dev", limit=100)
        ctx = random.choice(contexts)
        return jsonify({
            "context": ctx["context"],
            "title": ctx["title"],
            "reference_questions": [q["question"] for q in ctx["questions"]],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/upload-pdf", methods=["POST"])
def upload_pdf():
    """
    Upload a PDF file and extract text from it.

    Request: multipart/form-data with a 'file' field.
    Optional query params: start_page, end_page (1-indexed).

    Response:
        { "text": "...", "num_pages": N, "pages_extracted": [start, end], "char_count": N }
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Send a PDF file with field name 'file'."}), 400

    file = request.files["file"]
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported."}), 400

    # Save to temp file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        file.save(tmp.name)
        tmp.close()

        doc = fitz.open(tmp.name)
        total_pages = len(doc)

        # Optional page range (1-indexed from frontend)
        start_page = max(int(request.form.get("start_page", 1)) - 1, 0)
        end_page = min(int(request.form.get("end_page", total_pages)), total_pages)

        # Safety cap — max 50 pages at a time
        if end_page - start_page > 50:
            end_page = start_page + 50

        extracted_text = []
        for page_num in range(start_page, end_page):
            page = doc[page_num]
            text = page.get_text("text")
            if text.strip():
                extracted_text.append(text.strip())

        doc.close()

        full_text = "\n\n".join(extracted_text)

        # Clean up common PDF artifacts
        import re
        full_text = re.sub(r'\n{3,}', '\n\n', full_text)  # excessive newlines
        full_text = re.sub(r'[ \t]{2,}', ' ', full_text)   # excessive spaces
        full_text = re.sub(r'-\n(\w)', r'\1', full_text)   # hyphenated line breaks

        if len(full_text.strip()) < 20:
            return jsonify({"error": "Could not extract meaningful text from the PDF. The file may be image-based (scanned)."}), 400

        return jsonify({
            "text": full_text.strip(),
            "num_pages": total_pages,
            "pages_extracted": [start_page + 1, end_page],
            "char_count": len(full_text.strip()),
        })
    except Exception as e:
        return jsonify({"error": f"Failed to process PDF: {str(e)}"}), 500
    finally:
        os.unlink(tmp.name)


if __name__ == "__main__":
    print("🚀 Starting AQG API server...")
    print("   Endpoints:")
    print("   POST /api/generate    — Generate questions from text")
    print("   POST /api/upload-pdf  — Upload PDF and extract text")
    print("   POST /api/quiz        — Generate interactive quiz")
    print("   POST /api/evaluate    — Run evaluation on SQuAD")
    print("   GET  /api/stats       — Dataset statistics")
    print("   GET  /api/sample      — Random sample context")
    print("   GET  /api/health      — Health check")
    app.run(host="0.0.0.0", port=5001, debug=True)
