"""
Bloom's Taxonomy Classifier
🔥 Innovation #2 — Categorizes generated questions into Bloom's Taxonomy levels:
  - Remember (Knowledge)
  - Understand (Comprehension)
  - Apply (Application)
  - Analyze (Analysis)
  - Evaluate
  - Create
"""

import re
from .preprocessor import get_nlp


class BloomsLevel:
    REMEMBER = "Remember"
    UNDERSTAND = "Understand"
    APPLY = "Apply"
    ANALYZE = "Analyze"
    EVALUATE = "Evaluate"
    CREATE = "Create"

    ALL = [REMEMBER, UNDERSTAND, APPLY, ANALYZE, EVALUATE, CREATE]

    DESCRIPTIONS = {
        REMEMBER: "Recall facts and basic concepts",
        UNDERSTAND: "Explain ideas or concepts",
        APPLY: "Use information in new situations",
        ANALYZE: "Draw connections among ideas",
        EVALUATE: "Justify a stand or decision",
        CREATE: "Produce new or original work",
    }

    COLORS = {
        REMEMBER: "#4CAF50",       # Green
        UNDERSTAND: "#2196F3",     # Blue
        APPLY: "#FF9800",          # Orange
        ANALYZE: "#9C27B0",        # Purple
        EVALUATE: "#F44336",       # Red
        CREATE: "#E91E63",         # Pink
    }


# Question word → likely Bloom's level mapping
QUESTION_WORD_BLOOM = {
    "who": BloomsLevel.REMEMBER,
    "what": BloomsLevel.REMEMBER,      # Can be Understand depending on context
    "when": BloomsLevel.REMEMBER,
    "where": BloomsLevel.REMEMBER,
    "which": BloomsLevel.REMEMBER,
    "how many": BloomsLevel.REMEMBER,
    "how much": BloomsLevel.REMEMBER,
    "why": BloomsLevel.UNDERSTAND,
    "how": BloomsLevel.UNDERSTAND,     # Can be Apply/Analyze
}

# Verb patterns that indicate higher Bloom's levels
BLOOM_VERB_PATTERNS = {
    BloomsLevel.REMEMBER: [
        "define", "list", "name", "identify", "state", "describe",
        "recall", "recognize", "label", "select", "match",
    ],
    BloomsLevel.UNDERSTAND: [
        "explain", "summarize", "interpret", "classify", "compare",
        "contrast", "discuss", "distinguish", "illustrate", "paraphrase",
        "predict", "infer",
    ],
    BloomsLevel.APPLY: [
        "apply", "demonstrate", "use", "solve", "implement",
        "calculate", "execute", "show", "operate", "compute",
    ],
    BloomsLevel.ANALYZE: [
        "analyze", "differentiate", "examine", "compare", "contrast",
        "investigate", "categorize", "deconstruct", "distinguish",
        "relate", "organize",
    ],
    BloomsLevel.EVALUATE: [
        "evaluate", "justify", "assess", "critique", "judge",
        "argue", "defend", "support", "recommend", "prioritize",
    ],
    BloomsLevel.CREATE: [
        "create", "design", "develop", "formulate", "construct",
        "produce", "compose", "generate", "plan", "propose",
    ],
}


def classify_blooms(question) -> str:
    """
    Classify a generated question into a Bloom's Taxonomy level.

    Uses:
    1. Question word mapping
    2. Verb pattern matching in the question text
    3. Question structure analysis
    """
    q_text = question.question.lower() if isinstance(question, object) and hasattr(question, 'question') else str(question).lower()
    q_word = question.question_word.lower() if hasattr(question, 'question_word') else ""

    # Step 1: Check verb patterns (highest priority)
    for level in reversed(BloomsLevel.ALL):  # Check higher levels first
        if level in BLOOM_VERB_PATTERNS:
            for verb in BLOOM_VERB_PATTERNS[level]:
                if verb in q_text:
                    return level

    # Step 2: Check question phrase patterns
    if _matches_analysis_pattern(q_text):
        return BloomsLevel.ANALYZE
    if _matches_understanding_pattern(q_text):
        return BloomsLevel.UNDERSTAND

    # Step 3: Fall back to question word mapping
    # Check multi-word question starts first
    if q_text.startswith("how many") or q_text.startswith("how much"):
        return BloomsLevel.REMEMBER
    if q_text.startswith("how"):
        return BloomsLevel.UNDERSTAND
    if q_text.startswith("why"):
        return BloomsLevel.UNDERSTAND

    # Single question word
    if q_word in QUESTION_WORD_BLOOM:
        return QUESTION_WORD_BLOOM[q_word]

    # Default
    first_word = q_text.split()[0] if q_text.split() else ""
    return QUESTION_WORD_BLOOM.get(first_word, BloomsLevel.REMEMBER)


def _matches_analysis_pattern(q_text: str) -> bool:
    """Check if question matches analysis-level patterns."""
    patterns = [
        r"what is the (difference|relationship|connection) between",
        r"how (does|do|did) .+ (relate|connect|affect|influence|impact)",
        r"what (causes?|caused|effects?|resulted?)",
        r"(compare|contrast|distinguish)",
    ]
    return any(re.search(p, q_text) for p in patterns)


def _matches_understanding_pattern(q_text: str) -> bool:
    """Check if question matches understanding-level patterns."""
    patterns = [
        r"what (does|do|did) .+ mean",
        r"(explain|describe) (how|why|what)",
        r"in what way",
        r"what is the (purpose|significance|importance|meaning)",
    ]
    return any(re.search(p, q_text) for p in patterns)


def get_blooms_stats(questions: list) -> dict:
    """Get distribution of Bloom's taxonomy levels across questions."""
    counts = {level: 0 for level in BloomsLevel.ALL}
    for q in questions:
        level = classify_blooms(q)
        counts[level] += 1
    total = len(questions) if questions else 1
    return {
        "counts": counts,
        "percentages": {k: round(v / total * 100, 1) for k, v in counts.items()},
    }
