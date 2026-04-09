"""
Question Ranker
Filters and ranks generated questions by quality using
heuristic features and a lightweight ML classifier.
"""

import re
import numpy as np
from .preprocessor import get_nlp


def rank_questions(questions: list, top_k: int = 10) -> list:
    """
    Rank generated questions by quality.

    Uses a combination of heuristic features:
    - Length appropriateness
    - Grammar quality heuristics
    - Answer relevance
    - Diversity
    - Question word presence
    """
    if not questions:
        return []

    scored = []
    for q in questions:
        score = _compute_quality_score(q)
        scored.append((q, score))

    # Sort by combined score (original confidence * quality score)
    scored.sort(key=lambda x: x[1], reverse=True)

    # Deduplicate similar questions
    final = []
    seen_answers = set()
    for q, score in scored:
        # Avoid multiple questions about the same answer
        answer_key = q.answer.lower().strip()
        if answer_key in seen_answers:
            continue
        seen_answers.add(answer_key)
        q.confidence = round(score, 3)
        final.append(q)
        if len(final) >= top_k:
            break

    return final


def _compute_quality_score(question) -> float:
    """Compute a quality score for a generated question (0 to 1)."""
    score = question.confidence
    q_text = question.question

    # --- Feature 1: Length appropriateness ---
    word_count = len(q_text.split())
    if 4 <= word_count <= 15:
        score *= 1.1
    elif word_count < 4:
        score *= 0.5
    elif word_count > 20:
        score *= 0.7

    # --- Feature 2: Starts with a proper question word ---
    q_lower = q_text.lower()
    good_starts = ["who ", "what ", "where ", "when ", "why ", "how ", "which "]
    if any(q_lower.startswith(s) for s in good_starts):
        score *= 1.1

    # --- Feature 3: No repeated words (indicates generation artifact) ---
    words = q_text.lower().split()
    unique_ratio = len(set(words)) / len(words) if words else 0
    if unique_ratio < 0.6:
        score *= 0.4  # Penalize repetitive questions

    # --- Feature 4: Contains proper nouns / named entities (more specific) ---
    nlp = get_nlp()
    doc = nlp(q_text)
    has_proper_noun = any(t.pos_ == "PROPN" for t in doc)
    if has_proper_noun:
        score *= 1.05

    # --- Feature 5: Answer is not in the question ---
    if question.answer.lower() in q_text.lower():
        score *= 0.3  # Heavy penalty — answer shouldn't appear in question

    # --- Feature 6: Grammaticality heuristic ---
    # Check if question has a verb
    has_verb = any(t.pos_ == "VERB" or t.pos_ == "AUX" for t in doc)
    if not has_verb:
        score *= 0.5

    # --- Feature 7: Method bonus ---
    if question.method == "template":
        score *= 1.05
    elif question.method == "rule":
        score *= 1.0

    return min(score, 1.0)


def filter_questions(questions: list) -> list:
    """Remove obviously bad questions."""
    filtered = []
    for q in questions:
        # Skip if answer appears in question
        if q.answer.lower() in q.question.lower():
            continue
        # Skip very short questions
        if len(q.question.split()) < 3:
            continue
        # Skip if question is just the question word
        if q.question.strip("?").strip().lower() in {"who", "what", "where", "when", "why", "how"}:
            continue
        # Skip if question has no verb
        nlp = get_nlp()
        doc = nlp(q.question)
        has_verb = any(t.pos_ in ("VERB", "AUX") for t in doc)
        if not has_verb:
            continue
        filtered.append(q)
    return filtered
