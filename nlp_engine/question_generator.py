"""
Question Generator
Generates questions from text using rule-based transformations,
template-based approaches, and dependency-parse-driven methods.
No LLMs — purely classical NLP.
"""

import re
from .preprocessor import get_nlp
from .answer_extractor import (
    extract_answers_from_text,
    AnswerCandidate,
    ENTITY_QUESTION_MAP,
)


class GeneratedQuestion:
    """Represents a generated question with metadata."""

    def __init__(self, question: str, answer: str, source_sentence: str,
                 question_type: str, question_word: str,
                 method: str = "rule", confidence: float = 0.5):
        self.question = question
        self.answer = answer
        self.source_sentence = source_sentence
        self.question_type = question_type   # 'factual', 'entity', 'descriptive'
        self.question_word = question_word   # Who, What, Where, When, How, etc.
        self.method = method                 # 'rule', 'template', 'dependency'
        self.confidence = confidence

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "source_sentence": self.source_sentence,
            "question_type": self.question_type,
            "question_word": self.question_word,
            "method": self.method,
            "confidence": round(self.confidence, 3),
        }

    def __repr__(self):
        return f"Q: {self.question} | A: {self.answer}"


# ============================================================
# AUXILIARY VERB HELPERS
# ============================================================

AUX_VERBS = {"is", "are", "was", "were", "has", "have", "had", "do", "does", "did",
             "can", "could", "will", "would", "shall", "should", "may", "might", "must"}

BE_VERBS = {"is", "are", "was", "were", "am", "be", "been", "being"}


def _find_auxiliary(sent_doc):
    """Find the auxiliary verb in a sentence."""
    for token in sent_doc:
        if token.dep_ == "aux" and token.text.lower() in AUX_VERBS:
            return token
        if token.dep_ == "auxpass":
            return token
    return None


def _find_root(sent_doc):
    """Find the root verb of a sentence."""
    for token in sent_doc:
        if token.dep_ == "ROOT":
            return token
    return None


def _find_subject(sent_doc):
    """Find the subject of a sentence."""
    root = _find_root(sent_doc)
    if root:
        for child in root.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                # Get the full subject span (including modifiers)
                subtree_tokens = list(child.subtree)
                start = subtree_tokens[0].i
                end = subtree_tokens[-1].i + 1
                subject_span = sent_doc[start:end]
                return subject_span
    return None


def _get_do_form(root_token):
    """Get the appropriate 'do' auxiliary for subject-aux inversion."""
    tag = root_token.tag_
    if tag == "VBD":  # past tense
        return "did"
    elif tag == "VBZ":  # 3rd person singular present
        return "does"
    else:
        return "do"


def _get_base_form(token):
    """Get the base form of a verb."""
    return token.lemma_


# ============================================================
# RULE-BASED QUESTION GENERATION
# ============================================================

def _generate_entity_replacement_question(sentence: str, answer: AnswerCandidate,
                                          sent_doc) -> GeneratedQuestion | None:
    """
    Generate a question by replacing the answer span with a wh-word
    and applying subject-auxiliary inversion.

    E.g., "Albert Einstein was born in Germany" →
          "Where was Albert Einstein born?" (replacing "Germany")
    """
    nlp = get_nlp()
    q_word = answer.question_word
    answer_text = answer.text

    root = _find_root(sent_doc)
    if not root:
        return None

    subject = _find_subject(sent_doc)
    subject_text = subject.text if subject else ""
    aux = _find_auxiliary(sent_doc)

    # Determine if the answer IS the subject
    answer_is_subject = False
    if subject and answer_text.lower() in subject_text.lower():
        answer_is_subject = True

    if answer_is_subject:
        # === SUBJECT QUESTION ===
        # "Albert Einstein was born in Germany" → "Who was born in Germany?"
        # Replace the subject with the question word
        question = re.sub(
            re.escape(answer_text),
            q_word,
            sentence,
            count=1,
            flags=re.IGNORECASE,
        )
    else:
        # === NON-SUBJECT QUESTION — needs auxiliary inversion ===
        # "Einstein was born in Germany" → "Where was Einstein born?"
        # Remove the answer from the sentence
        remaining = re.sub(
            r'\s*' + re.escape(answer_text) + r'\s*',
            ' ',
            sentence,
            count=1,
            flags=re.IGNORECASE,
        ).strip()

        # Clean up prepositions left dangling
        remaining = re.sub(r'\s+in\s*$', '', remaining)
        remaining = re.sub(r'\s+on\s*$', '', remaining)
        remaining = re.sub(r'\s+at\s*$', '', remaining)
        remaining = re.sub(r'\s+from\s*$', '', remaining)
        remaining = re.sub(r'\s+to\s*$', '', remaining)
        remaining = re.sub(r'\s+by\s*$', '', remaining)
        remaining = re.sub(r'\s+of\s*$', '', remaining)
        remaining = re.sub(r'\s+for\s*$', '', remaining)

        if aux:
            # Move auxiliary before subject
            # Remove the aux from its current position
            remaining_no_aux = remaining.replace(f" {aux.text} ", " ", 1)
            if remaining_no_aux == remaining:
                remaining_no_aux = remaining.replace(f" {aux.text}", "", 1)

            question = f"{q_word} {aux.text} {remaining_no_aux.strip()}"
        elif root and root.text.lower() in BE_VERBS:
            # Root is a be-verb — move it before subject
            remaining_no_root = remaining.replace(f" {root.text} ", " ", 1)
            if remaining_no_root == remaining:
                remaining_no_root = remaining.replace(f"{root.text} ", "", 1)

            question = f"{q_word} {root.text} {remaining_no_root.strip()}"
        elif root:
            # No auxiliary — use do/does/did
            do_form = _get_do_form(root)
            base_verb = _get_base_form(root)
            remaining_with_base = remaining.replace(root.text, base_verb, 1)
            question = f"{q_word} {do_form} {remaining_with_base.strip()}"
        else:
            question = f"{q_word} {remaining.strip()}"

    # Clean up the question
    question = _clean_question(question)

    return GeneratedQuestion(
        question=question,
        answer=answer_text,
        source_sentence=sentence,
        question_type="entity" if answer.answer_type == "entity" else "factual",
        question_word=q_word,
        method="rule",
        confidence=answer.confidence * 0.85,
    )


# ============================================================
# TEMPLATE-BASED QUESTION GENERATION
# ============================================================

TEMPLATES = {
    "definition": {
        "pattern": r"^(.+?)\s+(?:is|are|was|were)\s+(?:a|an|the)\s+(.+)$",
        "template": "What {verb} {subject}?",
    },
    "location": {
        "pattern": r"(.+?)\s+(?:is|are|was|were)\s+(?:located|situated|found)\s+(?:in|at|on)\s+(.+)",
        "template": "Where {verb} {subject} located?",
    },
    "year_event": {
        "pattern": r"[Ii]n\s+(\d{4}),?\s+(.+)",
        "template": "When did {event_stripped}?",
    },
}


def _generate_template_questions(sentence: str, sent_doc) -> list[GeneratedQuestion]:
    """Generate questions using predefined templates."""
    questions = []

    root = _find_root(sent_doc)
    subject = _find_subject(sent_doc)

    # === Template 1: Definition pattern — "X is a Y" → "What is X?" ===
    match = re.match(TEMPLATES["definition"]["pattern"], sentence, re.IGNORECASE)
    if match:
        subj = match.group(1).strip()
        verb = "is"
        for token in sent_doc:
            if token.dep_ == "ROOT" and token.text.lower() in BE_VERBS:
                verb = token.text.lower()
                break

        question = f"What {verb} {subj}?"
        answer = match.group(2).strip().rstrip(".")
        questions.append(GeneratedQuestion(
            question=question,
            answer=answer,
            source_sentence=sentence,
            question_type="definition",
            question_word="What",
            method="template",
            confidence=0.8,
        ))

    # === Template 2: Year-event pattern — "In YYYY, X happened" → "When did X happen?" ===
    match = re.match(TEMPLATES["year_event"]["pattern"], sentence)
    if match:
        year = match.group(1)
        event = match.group(2).strip().rstrip(".")

        # Convert to past tense question
        event_doc = get_nlp()(event)
        event_root = _find_root(event_doc)

        if event_root:
            base = _get_base_form(event_root)
            event_stripped = event.replace(event_root.text, base, 1)
            question = f"When did {event_stripped}?"
        else:
            question = f"When did {event}?"

        questions.append(GeneratedQuestion(
            question=_clean_question(question),
            answer=year,
            source_sentence=sentence,
            question_type="temporal",
            question_word="When",
            method="template",
            confidence=0.75,
        ))

    # === Template 3: Subject did action → "What did Subject do?" ===
    if root and subject and root.pos_ == "VERB" and root.text.lower() not in BE_VERBS:
        # Check if there's a direct object
        dobj = None
        for child in root.children:
            if child.dep_ == "dobj":
                subtree = list(child.subtree)
                dobj_span = sent_doc[subtree[0].i:subtree[-1].i + 1]
                dobj = dobj_span.text
                break

        if dobj:
            do_form = _get_do_form(root)
            base = _get_base_form(root)
            question = f"What {do_form} {subject.text} {base}?"
            questions.append(GeneratedQuestion(
                question=_clean_question(question),
                answer=dobj,
                source_sentence=sentence,
                question_type="action",
                question_word="What",
                method="template",
                confidence=0.65,
            ))

    return questions


# ============================================================
# MAIN GENERATION FUNCTION
# ============================================================

def generate_questions(text: str, max_questions: int = 20) -> list[GeneratedQuestion]:
    """
    Generate questions from input text using all available methods.

    Args:
        text: Input passage/paragraph
        max_questions: Maximum number of questions to return

    Returns:
        List of GeneratedQuestion objects, sorted by confidence
    """
    nlp = get_nlp()
    doc = nlp(text)
    all_questions = []
    seen_questions = set()

    for sent in doc.sents:
        sent_text = sent.text.strip()

        # Skip very short sentences
        if len(sent_text.split()) < 5:
            continue

        sent_doc = nlp(sent_text)

        # --- Method 1: Entity-replacement questions ---
        from .answer_extractor import extract_answers_from_sentence
        candidates = extract_answers_from_sentence(sent_doc, sent_text)

        for candidate in candidates:
            q = _generate_entity_replacement_question(sent_text, candidate, sent_doc)
            if q and _is_valid_question(q.question) and q.question.lower() not in seen_questions:
                all_questions.append(q)
                seen_questions.add(q.question.lower())

        # --- Method 2: Template-based questions ---
        template_qs = _generate_template_questions(sent_text, sent_doc)
        for q in template_qs:
            if _is_valid_question(q.question) and q.question.lower() not in seen_questions:
                all_questions.append(q)
                seen_questions.add(q.question.lower())

    # Sort by confidence and return top-K
    all_questions.sort(key=lambda q: q.confidence, reverse=True)
    return all_questions[:max_questions]


# ============================================================
# CLEANING AND VALIDATION
# ============================================================

def _clean_question(question: str) -> str:
    """Clean up a generated question — handles grammatical artifacts."""
    # Remove extra whitespace
    question = re.sub(r'\s+', ' ', question).strip()

    # Fix dangling prepositions mid-sentence: "born in, Germany" → "born in Germany"
    question = re.sub(r'\s+in\s+,', ' in', question)
    question = re.sub(r'\s+on\s+,', ' on', question)
    question = re.sub(r'\s+at\s+,', ' at', question)
    question = re.sub(r',\s*,', ',', question)  # double commas

    # Fix "in in" or "in ," patterns from answer removal
    question = re.sub(r'\bin\s+in\b', 'in', question)
    question = re.sub(r'\bin\s*\?', '?', question)
    question = re.sub(r'\bon\s*\?', '?', question)
    question = re.sub(r'\bat\s*\?', '?', question)

    # Fix dangling prep + period/comma at end before ?
    question = re.sub(r'\s+in\s*\.\s*\?', '?', question)

    # Remove trailing prepositions before question mark
    question = re.sub(r'\s+(in|on|at|from|to|by|of|for|with)\s*\?$', '?', question)

    # Replace pronouns like "He/She" at start with the proper question word context
    # e.g. "When did He move" → "When did he move" (lowercase)
    question = re.sub(r'^(Who|What|Where|When|Why|How\s?\w*)\s+(did|does|do|was|were|is|are|has|have|had)\s+He\b',
                      lambda m: f"{m.group(1)} {m.group(2)} he", question)
    question = re.sub(r'^(Who|What|Where|When|Why|How\s?\w*)\s+(did|does|do|was|were|is|are|has|have|had)\s+She\b',
                      lambda m: f"{m.group(1)} {m.group(2)} she", question)

    # Ensure it ends with a question mark
    question = question.rstrip(".,;:!?")
    question = question + "?"

    # Capitalize first letter
    if question:
        question = question[0].upper() + question[1:]

    # Remove double question marks
    question = question.replace("??", "?")

    # Fix common artifacts
    question = question.replace(" ,", ",")
    question = re.sub(r'\s+', ' ', question).strip()

    return question


def _is_valid_question(question: str) -> bool:
    """Check if a generated question is valid and well-formed."""
    if not question or len(question) < 10:
        return False

    words = question.split()
    if len(words) < 3:
        return False

    # Must start with a question word or auxiliary
    valid_starts = {"who", "what", "where", "when", "why", "how", "which",
                    "is", "are", "was", "were", "do", "does", "did",
                    "can", "could", "will", "would", "has", "have", "had"}
    if words[0].lower().rstrip(",?") not in valid_starts:
        return False

    # Must end with question mark
    if not question.endswith("?"):
        return False

    return True
