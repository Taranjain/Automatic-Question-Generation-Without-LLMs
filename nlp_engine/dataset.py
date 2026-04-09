"""
SQuAD 2.0 Dataset Loader
Loads and preprocesses the SQuAD 2.0 JSON files for training and evaluation.
"""

import json
import os
from typing import Optional


class SQuADDataset:
    """Loader for SQuAD 2.0 dataset."""

    def __init__(self, train_path: str, dev_path: str):
        self.train_path = train_path
        self.dev_path = dev_path
        self.train_data = None
        self.dev_data = None

    def load(self):
        """Load both train and dev datasets."""
        self.train_data = self._load_file(self.train_path)
        self.dev_data = self._load_file(self.dev_path)
        return self

    def _load_file(self, path: str) -> dict:
        """Load a single SQuAD JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def extract_qa_pairs(self, split: str = "train", answerable_only: bool = True,
                         limit: Optional[int] = None) -> list[dict]:
        """
        Extract question-answer pairs from the dataset.

        Returns list of dicts with keys:
            - context: the passage text
            - question: the question string
            - answers: list of answer dicts with 'text' and 'answer_start'
            - is_impossible: whether the question is unanswerable
            - id: unique question id
            - title: article title
        """
        data = self.train_data if split == "train" else self.dev_data
        if data is None:
            raise ValueError(f"Dataset not loaded. Call load() first.")

        qa_pairs = []
        for article in data["data"]:
            title = article["title"]
            for paragraph in article["paragraphs"]:
                context = paragraph["context"]
                for qa in paragraph["qas"]:
                    is_impossible = qa.get("is_impossible", False)

                    if answerable_only and is_impossible:
                        continue

                    qa_pairs.append({
                        "context": context,
                        "question": qa["question"],
                        "answers": qa.get("answers", []),
                        "is_impossible": is_impossible,
                        "id": qa["id"],
                        "title": title,
                    })

                    if limit and len(qa_pairs) >= limit:
                        return qa_pairs

        return qa_pairs

    def extract_contexts(self, split: str = "train",
                         limit: Optional[int] = None) -> list[dict]:
        """
        Extract unique contexts (passages) from the dataset.

        Returns list of dicts with keys:
            - context: the passage text
            - title: article title
            - questions: list of associated questions
        """
        data = self.train_data if split == "train" else self.dev_data
        if data is None:
            raise ValueError(f"Dataset not loaded. Call load() first.")

        contexts = []
        for article in data["data"]:
            title = article["title"]
            for paragraph in article["paragraphs"]:
                context = paragraph["context"]
                questions = []
                for qa in paragraph["qas"]:
                    if not qa.get("is_impossible", False):
                        questions.append({
                            "question": qa["question"],
                            "answers": qa.get("answers", []),
                        })

                if questions:
                    contexts.append({
                        "context": context,
                        "title": title,
                        "questions": questions,
                    })

                if limit and len(contexts) >= limit:
                    return contexts

        return contexts

    def get_stats(self, split: str = "train") -> dict:
        """Get dataset statistics."""
        data = self.train_data if split == "train" else self.dev_data
        if data is None:
            raise ValueError("Dataset not loaded. Call load() first.")

        total_articles = len(data["data"])
        total_paragraphs = 0
        total_questions = 0
        answerable = 0
        unanswerable = 0

        for article in data["data"]:
            for paragraph in article["paragraphs"]:
                total_paragraphs += 1
                for qa in paragraph["qas"]:
                    total_questions += 1
                    if qa.get("is_impossible", False):
                        unanswerable += 1
                    else:
                        answerable += 1

        return {
            "split": split,
            "version": data.get("version", "unknown"),
            "total_articles": total_articles,
            "total_paragraphs": total_paragraphs,
            "total_questions": total_questions,
            "answerable": answerable,
            "unanswerable": unanswerable,
        }


def get_default_dataset() -> SQuADDataset:
    """Get dataset with default paths."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, "train-v2.0.json")
    dev_path = os.path.join(base_dir, "dev-v2.0.json")
    return SQuADDataset(train_path, dev_path)
