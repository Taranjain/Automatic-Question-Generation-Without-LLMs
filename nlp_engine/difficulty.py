"""
Difficulty Level Classifier
🔥 Innovation #1 — Classifies generated questions as Easy / Medium / Hard
based on linguistic features of the source sentence and answer.
"""

import textstat
from .preprocessor import get_nlp


class DifficultyLevel:
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


def classify_difficulty(question) -> str:
    """
    Classify a generated question's difficulty level.

    Factors:
    1. Source sentence readability (Flesch Reading Ease)
    2. Answer type complexity
    3. Sentence length
    4. Syntactic depth (number of clauses)
    5. Question word complexity
    6. Number of named entities in source
    """
    features = extract_difficulty_features(question)
    score = _compute_difficulty_score(features)

    if score <= 0.33:
        return DifficultyLevel.EASY
    elif score <= 0.66:
        return DifficultyLevel.MEDIUM
    else:
        return DifficultyLevel.HARD


def extract_difficulty_features(question) -> dict:
    """Extract features used for difficulty classification."""
    source = question.source_sentence
    q_text = question.question
    answer = question.answer
    nlp = get_nlp()

    # 1. Readability
    flesch_score = textstat.flesch_reading_ease(source)

    # 2. Sentence length
    word_count = len(source.split())

    # 3. Syntactic depth — count subordinate clauses
    doc = nlp(source)
    clause_count = sum(1 for token in doc if token.dep_ in ("advcl", "ccomp", "xcomp", "relcl", "acl"))

    # 4. Number of entities
    entity_count = len(list(doc.ents))

    # 5. Answer complexity
    answer_word_count = len(answer.split())
    answer_is_entity = question.answer_type == "entity" if hasattr(question, 'answer_type') else False

    # 6. Question word complexity
    q_word = question.question_word if hasattr(question, 'question_word') else ""
    q_word_complexity = _question_word_difficulty(q_word)

    # 7. Number of prepositional phrases
    prep_count = sum(1 for token in doc if token.dep_ == "prep")

    return {
        "flesch_score": flesch_score,
        "word_count": word_count,
        "clause_count": clause_count,
        "entity_count": entity_count,
        "answer_word_count": answer_word_count,
        "answer_is_entity": answer_is_entity,
        "q_word_complexity": q_word_complexity,
        "prep_count": prep_count,
    }


def _compute_difficulty_score(features: dict) -> float:
    """
    Compute a difficulty score from 0 (easiest) to 1 (hardest).
    Uses weighted combination of features.
    """
    score = 0.0

    # Readability: lower Flesch = harder (scale: 0-100, invert and normalize)
    flesch = features["flesch_score"]
    if flesch > 80:
        score += 0.0
    elif flesch > 60:
        score += 0.1
    elif flesch > 40:
        score += 0.2
    elif flesch > 20:
        score += 0.3
    else:
        score += 0.4

    # Sentence length: longer = harder
    wc = features["word_count"]
    if wc <= 10:
        score += 0.0
    elif wc <= 20:
        score += 0.1
    elif wc <= 30:
        score += 0.15
    else:
        score += 0.2

    # Clauses: more = harder
    cc = features["clause_count"]
    score += min(cc * 0.08, 0.2)

    # Question word complexity
    score += features["q_word_complexity"] * 0.15

    # Answer complexity: multi-word answers = harder
    if features["answer_word_count"] > 3:
        score += 0.05

    # Normalize to 0-1
    return min(max(score, 0.0), 1.0)


def _question_word_difficulty(q_word: str) -> float:
    """Assign a complexity score to a question word."""
    difficulty_map = {
        "Who": 0.2,
        "What": 0.4,
        "Where": 0.3,
        "When": 0.2,
        "Which": 0.5,
        "How": 0.6,
        "How many": 0.5,
        "How much": 0.5,
        "Why": 0.8,
    }
    return difficulty_map.get(q_word, 0.4)


def get_difficulty_stats(questions: list) -> dict:
    """Get distribution of difficulty levels across questions."""
    counts = {DifficultyLevel.EASY: 0, DifficultyLevel.MEDIUM: 0, DifficultyLevel.HARD: 0}
    for q in questions:
        difficulty = classify_difficulty(q)
        counts[difficulty] += 1
    total = len(questions) if questions else 1
    return {
        "counts": counts,
        "percentages": {k: round(v / total * 100, 1) for k, v in counts.items()},
    }
