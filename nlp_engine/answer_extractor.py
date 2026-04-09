"""
Answer Candidate Extractor
Identifies potential answer spans from text using NER, noun phrases,
dependency parsing, and pattern matching.
"""

from .preprocessor import process_text, process_sentence, get_nlp


# Entity type to question word mapping
ENTITY_QUESTION_MAP = {
    "PERSON": "Who",
    "NORP": "What",        # nationalities, religious/political groups
    "FAC": "What",         # buildings, airports, highways
    "ORG": "What",         # organizations
    "GPE": "Where",        # countries, cities, states
    "LOC": "Where",        # non-GPE locations
    "PRODUCT": "What",     # objects, vehicles, foods
    "EVENT": "What",       # named events
    "WORK_OF_ART": "What", # titles of works
    "LAW": "What",         # named documents
    "LANGUAGE": "What",    # languages
    "DATE": "When",        # dates
    "TIME": "When",        # times
    "PERCENT": "How much", # percentages
    "MONEY": "How much",   # monetary values
    "QUANTITY": "How many", # measurements
    "ORDINAL": "Which",    # ordinal numbers (first, second)
    "CARDINAL": "How many", # cardinal numbers
}


class AnswerCandidate:
    """Represents a potential answer extracted from text."""

    def __init__(self, text: str, answer_type: str, entity_label: str = "",
                 question_word: str = "", sentence: str = "",
                 start_char: int = -1, confidence: float = 1.0):
        self.text = text
        self.answer_type = answer_type       # 'entity', 'noun_phrase', 'number', 'subject'
        self.entity_label = entity_label     # NER label (PERSON, GPE, etc.)
        self.question_word = question_word   # Who, What, Where, When, etc.
        self.sentence = sentence             # Source sentence
        self.start_char = start_char
        self.confidence = confidence

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "answer_type": self.answer_type,
            "entity_label": self.entity_label,
            "question_word": self.question_word,
            "sentence": self.sentence,
            "confidence": self.confidence,
        }

    def __repr__(self):
        return f"AnswerCandidate('{self.text}', type={self.answer_type}, qword={self.question_word})"


def extract_answers_from_sentence(sent_doc, sentence_text: str) -> list[AnswerCandidate]:
    """
    Extract answer candidates from a single sentence.

    Strategies:
    1. Named Entity Recognition — highest confidence
    2. Noun phrase subjects/objects — medium confidence
    3. Numerical values — medium confidence
    4. Key verbs/actions for 'What did X do?' style questions
    """
    candidates = []
    seen_texts = set()

    # === Strategy 1: Named Entities ===
    for ent in sent_doc.ents:
        if ent.text.strip() and ent.text.lower() not in seen_texts:
            question_word = ENTITY_QUESTION_MAP.get(ent.label_, "What")
            candidates.append(AnswerCandidate(
                text=ent.text.strip(),
                answer_type="entity",
                entity_label=ent.label_,
                question_word=question_word,
                sentence=sentence_text,
                start_char=ent.start_char,
                confidence=0.9,
            ))
            seen_texts.add(ent.text.lower())

    # === Strategy 2: Noun Phrases ===
    for chunk in sent_doc.noun_chunks:
        chunk_text = chunk.text.strip()
        if chunk_text.lower() not in seen_texts and len(chunk_text.split()) >= 1:
            # Determine question word based on dependency role
            if chunk.root.dep_ in ("nsubj", "nsubjpass"):
                question_word = "Who" if _is_person_like(chunk) else "What"
                confidence = 0.75
            elif chunk.root.dep_ in ("dobj", "pobj", "attr"):
                question_word = "What"
                confidence = 0.7
            else:
                question_word = "What"
                confidence = 0.5

            # Skip pronouns and very short stopword-only chunks
            if chunk.root.pos_ == "PRON":
                continue
            if all(t.is_stop for t in chunk):
                continue

            candidates.append(AnswerCandidate(
                text=chunk_text,
                answer_type="noun_phrase",
                entity_label="",
                question_word=question_word,
                sentence=sentence_text,
                start_char=chunk.start_char,
                confidence=confidence,
            ))
            seen_texts.add(chunk_text.lower())

    # === Strategy 3: Numbers and Quantities ===
    for token in sent_doc:
        if token.like_num and token.text.lower() not in seen_texts:
            # Check context for the number
            question_word = "How many"
            if token.head.text.lower() in ("year", "years", "century", "centuries", "date"):
                question_word = "When"
            elif token.head.text.lower() in ("dollar", "dollars", "pound", "euro", "rupee"):
                question_word = "How much"

            candidates.append(AnswerCandidate(
                text=token.text,
                answer_type="number",
                entity_label="CARDINAL",
                question_word=question_word,
                sentence=sentence_text,
                start_char=token.idx,
                confidence=0.6,
            ))
            seen_texts.add(token.text.lower())

    return candidates


def extract_answers_from_text(text: str) -> list[AnswerCandidate]:
    """Extract answer candidates from a full text passage."""
    nlp = get_nlp()
    doc = nlp(text)
    all_candidates = []

    for sent in doc.sents:
        sent_text = sent.text.strip()
        if len(sent_text.split()) < 5:  # Skip very short sentences
            continue

        # Process sentence individually for better NLP
        sent_doc = nlp(sent_text)
        candidates = extract_answers_from_sentence(sent_doc, sent_text)
        all_candidates.extend(candidates)

    return all_candidates


def _is_person_like(chunk) -> bool:
    """Heuristic to determine if a noun chunk refers to a person."""
    for token in chunk:
        if token.ent_type_ == "PERSON":
            return True
        if token.pos_ == "PROPN" and token.dep_ in ("nsubj", "nsubjpass"):
            return True
    return False
