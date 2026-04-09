"""
Evaluator
Evaluates generated questions against SQuAD ground-truth questions
using BLEU, ROUGE, and other metrics.
"""

import sys
import os
import json
from collections import defaultdict

import nltk
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from .dataset import get_default_dataset
from .question_generator import generate_questions
from .difficulty import classify_difficulty, get_difficulty_stats
from .blooms_taxonomy import classify_blooms, get_blooms_stats
from .question_ranker import rank_questions, filter_questions


def compute_bleu(reference: str, hypothesis: str) -> float:
    """Compute BLEU score between a reference and hypothesis question."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if not ref_tokens or not hyp_tokens:
        return 0.0

    smoothie = SmoothingFunction().method1
    try:
        score = sentence_bleu(
            [ref_tokens],
            hyp_tokens,
            weights=(0.5, 0.5, 0.0, 0.0),  # BLEU-2
            smoothing_function=smoothie,
        )
        return score
    except Exception:
        return 0.0


def compute_rouge(reference: str, hypothesis: str) -> dict:
    """Compute ROUGE scores between a reference and hypothesis question."""
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


def evaluate_on_squad(sample_size: int = 50, split: str = "dev") -> dict:
    """
    Evaluate the question generation system on SQuAD data.

    For each context in the dataset:
    1. Generate questions using our pipeline
    2. Compare with ground-truth SQuAD questions
    3. Compute BLEU and ROUGE scores

    Returns aggregate metrics.
    """
    dataset = get_default_dataset()
    dataset.load()

    contexts = dataset.extract_contexts(split=split, limit=sample_size)

    all_bleu_scores = []
    all_rouge1_scores = []
    all_rouge2_scores = []
    all_rougeL_scores = []
    question_type_counts = defaultdict(int)
    total_generated = 0
    total_reference = 0

    results = []

    for i, ctx in enumerate(contexts):
        context_text = ctx["context"]
        reference_questions = [q["question"] for q in ctx["questions"]]
        total_reference += len(reference_questions)

        # Generate questions
        generated = generate_questions(context_text, max_questions=15)
        generated = filter_questions(generated)
        generated = rank_questions(generated, top_k=10)
        total_generated += len(generated)

        # Compute metrics — for each generated question, find best matching reference
        for gen_q in generated:
            best_bleu = 0.0
            best_rouge = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

            for ref_q in reference_questions:
                bleu = compute_bleu(ref_q, gen_q.question)
                rouge = compute_rouge(ref_q, gen_q.question)

                if bleu > best_bleu:
                    best_bleu = bleu
                if rouge["rouge1"] > best_rouge["rouge1"]:
                    best_rouge = rouge

            all_bleu_scores.append(best_bleu)
            all_rouge1_scores.append(best_rouge["rouge1"])
            all_rouge2_scores.append(best_rouge["rouge2"])
            all_rougeL_scores.append(best_rouge["rougeL"])

            # Count question types
            question_type_counts[gen_q.question_word] += 1

        # Store per-context result
        results.append({
            "context_preview": context_text[:100] + "...",
            "title": ctx["title"],
            "reference_count": len(reference_questions),
            "generated_count": len(generated),
            "generated_questions": [q.to_dict() for q in generated],
        })

        # Progress feedback
        if (i + 1) % 10 == 0:
            print(f"  Evaluated {i + 1}/{len(contexts)} contexts...")

    # Aggregate metrics
    avg_bleu = sum(all_bleu_scores) / len(all_bleu_scores) if all_bleu_scores else 0
    avg_rouge1 = sum(all_rouge1_scores) / len(all_rouge1_scores) if all_rouge1_scores else 0
    avg_rouge2 = sum(all_rouge2_scores) / len(all_rouge2_scores) if all_rouge2_scores else 0
    avg_rougeL = sum(all_rougeL_scores) / len(all_rougeL_scores) if all_rougeL_scores else 0

    # Get full question list for Bloom's and difficulty stats
    all_generated = []
    for ctx in contexts:
        gen = generate_questions(ctx["context"], max_questions=5)
        all_generated.extend(gen)

    metrics = {
        "sample_size": len(contexts),
        "total_reference_questions": total_reference,
        "total_generated_questions": total_generated,
        "avg_questions_per_context": round(total_generated / len(contexts), 2) if contexts else 0,
        "bleu": {
            "average": round(avg_bleu, 4),
            "max": round(max(all_bleu_scores), 4) if all_bleu_scores else 0,
            "min": round(min(all_bleu_scores), 4) if all_bleu_scores else 0,
        },
        "rouge": {
            "rouge1": round(avg_rouge1, 4),
            "rouge2": round(avg_rouge2, 4),
            "rougeL": round(avg_rougeL, 4),
        },
        "question_type_distribution": dict(question_type_counts),
        "difficulty_distribution": get_difficulty_stats(all_generated) if all_generated else {},
        "blooms_distribution": get_blooms_stats(all_generated) if all_generated else {},
    }

    return {
        "metrics": metrics,
        "sample_results": results[:10],  # First 10 for review
    }


def generate_report(metrics: dict) -> str:
    """Generate a human-readable evaluation report."""
    m = metrics["metrics"]
    report = []
    report.append("=" * 60)
    report.append("  AUTOMATIC QUESTION GENERATION — EVALUATION REPORT")
    report.append("=" * 60)
    report.append("")
    report.append(f"📊 Dataset: SQuAD 2.0 ({m['sample_size']} contexts evaluated)")
    report.append(f"📝 Total reference questions: {m['total_reference_questions']}")
    report.append(f"🤖 Total generated questions: {m['total_generated_questions']}")
    report.append(f"📈 Avg questions per context: {m['avg_questions_per_context']}")
    report.append("")
    report.append("─── BLEU Scores ───")
    report.append(f"  Average BLEU-2:  {m['bleu']['average']:.4f}")
    report.append(f"  Max BLEU-2:      {m['bleu']['max']:.4f}")
    report.append(f"  Min BLEU-2:      {m['bleu']['min']:.4f}")
    report.append("")
    report.append("─── ROUGE Scores ───")
    report.append(f"  ROUGE-1:  {m['rouge']['rouge1']:.4f}")
    report.append(f"  ROUGE-2:  {m['rouge']['rouge2']:.4f}")
    report.append(f"  ROUGE-L:  {m['rouge']['rougeL']:.4f}")
    report.append("")
    report.append("─── Question Type Distribution ───")
    for qtype, count in m.get("question_type_distribution", {}).items():
        report.append(f"  {qtype}: {count}")
    report.append("")

    if m.get("difficulty_distribution"):
        report.append("─── Difficulty Distribution ───")
        diff = m["difficulty_distribution"]
        for level, pct in diff.get("percentages", {}).items():
            count = diff.get("counts", {}).get(level, 0)
            report.append(f"  {level.capitalize():8s}: {count:4d} ({pct:.1f}%)")
        report.append("")

    if m.get("blooms_distribution"):
        report.append("─── Bloom's Taxonomy Distribution ───")
        bloom = m["blooms_distribution"]
        for level, pct in bloom.get("percentages", {}).items():
            count = bloom.get("counts", {}).get(level, 0)
            report.append(f"  {level:12s}: {count:4d} ({pct:.1f}%)")

    report.append("")
    report.append("=" * 60)
    return "\n".join(report)


# === CLI entry point ===
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate AQG system on SQuAD")
    parser.add_argument("--sample", type=int, default=50, help="Number of contexts to evaluate")
    parser.add_argument("--split", type=str, default="dev", help="Dataset split (train/dev)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    args = parser.parse_args()

    print(f"🚀 Starting evaluation on {args.sample} contexts from {args.split} split...")
    results = evaluate_on_squad(sample_size=args.sample, split=args.split)
    report = generate_report(results)
    print(report)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n📁 Results saved to {args.output}")
