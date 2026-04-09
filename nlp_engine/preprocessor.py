"""
NLP Preprocessor
Handles text processing using spaCy: sentence segmentation, POS tagging,
NER, dependency parsing, and noun chunk extraction.
"""

import spacy
from typing import Optional


# Load spaCy model (singleton)
_nlp = None


def get_nlp():
    """Get or initialize the spaCy NLP pipeline."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


class ProcessedText:
    """Container for processed text with NLP annotations."""

    def __init__(self, doc):
        self.doc = doc
        self.text = doc.text

    @property
    def sentences(self) -> list:
        """Get list of sentence spans."""
        return list(self.doc.sents)

    @property
    def entities(self) -> list[dict]:
        """Get named entities."""
        return [
            {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            }
            for ent in self.doc.ents
        ]

    @property
    def noun_chunks(self) -> list[dict]:
        """Get noun phrases."""
        return [
            {
                "text": chunk.text,
                "root": chunk.root.text,
                "root_dep": chunk.root.dep_,
                "root_head": chunk.root.head.text,
            }
            for chunk in self.doc.noun_chunks
        ]

    def get_sentence_data(self) -> list[dict]:
        """
        Get detailed data for each sentence.

        Returns list of dicts with:
            - text: sentence text
            - tokens: list of token dicts
            - entities: named entities in this sentence
            - noun_chunks: noun phrases in this sentence
            - root: root verb of the sentence
            - subject: subject of the sentence (if found)
        """
        sentences = []
        for sent in self.doc.sents:
            tokens = []
            for token in sent:
                tokens.append({
                    "text": token.text,
                    "lemma": token.lemma_,
                    "pos": token.pos_,
                    "tag": token.tag_,
                    "dep": token.dep_,
                    "head": token.head.text,
                    "is_stop": token.is_stop,
                })

            # Find root verb
            root = None
            for token in sent:
                if token.dep_ == "ROOT":
                    root = token
                    break

            # Find subject
            subject = None
            if root:
                for child in root.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        subject = child
                        break

            # Get entities in this sentence
            sent_ents = [
                {"text": ent.text, "label": ent.label_}
                for ent in sent.ents
            ]

            # Get noun chunks in this sentence
            sent_chunks = []
            for chunk in self.doc.noun_chunks:
                if chunk.start >= sent.start and chunk.end <= sent.end:
                    sent_chunks.append({
                        "text": chunk.text,
                        "root": chunk.root.text,
                    })

            sentences.append({
                "text": sent.text.strip(),
                "span": sent,
                "tokens": tokens,
                "entities": sent_ents,
                "noun_chunks": sent_chunks,
                "root": {
                    "text": root.text,
                    "lemma": root.lemma_,
                    "pos": root.pos_,
                    "tag": root.tag_,
                } if root else None,
                "subject": {
                    "text": subject.text,
                    "dep": subject.dep_,
                } if subject else None,
            })

        return sentences


def process_text(text: str) -> ProcessedText:
    """Process text through the NLP pipeline."""
    nlp = get_nlp()
    doc = nlp(text)
    return ProcessedText(doc)


def process_sentence(sentence: str):
    """Process a single sentence and return the spaCy doc."""
    nlp = get_nlp()
    return nlp(sentence)
