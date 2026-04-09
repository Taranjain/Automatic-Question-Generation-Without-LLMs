# Survey: Automatic Question Generation from Textbooks Without LLMs

**Course Project — NLP**
**Author:** Taran Jain
**Dataset:** SQuAD 2.0 (Stanford Question Answering Dataset)

---

## Table of Contents
1. [Introduction](#1-introduction)
2. [Problem Statement](#2-problem-statement)
3. [Applications & Motivation](#3-applications--motivation)
4. [Dataset: SQuAD 2.0](#4-dataset-squad-20)
5. [Literature Review — Approaches to Question Generation](#5-literature-review)
6. [Key Challenges in AQG](#6-key-challenges-in-aqg)
7. [Research Gap & Our Proposed Approach](#7-research-gap--our-proposed-approach)
8. [Why This Model and Not Others](#8-why-this-model-and-not-others)
9. [Survey-to-Implementation Mapping](#9-survey-to-implementation-mapping)
10. [References](#10-references)

---

## 1. Introduction

**Automatic Question Generation (AQG)** is the task of automatically generating natural language questions from a given input text (passage, paragraph, or document). It is a sub-field of Natural Language Processing (NLP) that has applications in education, assessment, information retrieval, and conversational AI.

Traditionally, creating questions for exams, quizzes, and study materials has been a manual, labor-intensive process done by educators. AQG systems seek to automate this, enabling on-demand question creation from any textbook or article content.

This survey reviews the landscape of AQG research, with focus on **classical NLP approaches that do not rely on Large Language Models (LLMs).** We examine rule-based, template-based, syntax-based, and statistical methods, and identify the design decisions that shaped our implementation.

---

## 2. Problem Statement

> **Given an input text paragraph from a textbook or educational material, automatically generate a set of relevant, grammatically correct, and pedagogically meaningful questions along with their corresponding answers — without using any Large Language Model (LLM) or neural text generation model.**

### Formal Definition

Let `C` be a context passage consisting of sentences `{s₁, s₂, ..., sₙ}`. For each sentence `sᵢ`, the system must:

1. **Identify answer candidates** `A = {a₁, a₂, ..., aₖ}` — spans of text that are meaningful answer targets (named entities, noun phrases, numerical values).
2. **Generate questions** `Q = {q₁, q₂, ..., qₖ}` — each `qⱼ` is a natural language question whose answer is `aⱼ`, derived from the information in `sᵢ`.
3. **Rank questions** by grammaticality, relevance, and pedagogical value.
4. **Classify questions** by difficulty level and cognitive taxonomy (Bloom's).

### Sub-Problems Addressed
| Sub-problem | Description |
|------------|-------------|
| Answer Extraction | Identifying what to ask about |
| Question Formulation | Transforming declarative sentences into interrogative form |
| Question Ranking | Filtering low-quality outputs |
| Difficulty Estimation | Classifying Easy / Medium / Hard |
| Cognitive Level Mapping | Categorizing by Bloom's Taxonomy levels |

---

## 3. Applications & Motivation

### 3.1 Educational Applications
- **Automated Quiz Generation:** Teachers can generate quizzes from textbook chapters instantly.
- **Adaptive Learning:** Systems can generate questions at appropriate difficulty levels for each student.
- **Reading Comprehension Practice:** Students get practice questions from any study material.
- **Exam Preparation:** Automated generation of practice tests from course content.

### 3.2 Information Retrieval & Knowledge Management
- **FAQ Generation:** Automatically generate FAQ pages from documentation.
- **Knowledge Base QA:** Enhance search systems with question-answer pairs.
- **Content Summarization:** Questions highlight the key information in a passage.

### 3.3 Conversational AI
- **Chatbot Training:** Generate question-answer pairs for training dialogue systems.
- **Conversational Assessment:** Tutoring bots that can quiz students interactively.

### 3.4 Why This Problem Matters
- The education sector generates millions of assessments annually — manual creation is costly and slow.
- Each textbook chapter could yield hundreds of potential questions — only a fraction are ever written by humans.
- Automated QG ensures broader coverage of content and consistent quality.
- During COVID-19, the demand for online assessment tools surged — AQG systems became critical for remote education platforms.

---

## 4. Dataset: SQuAD 2.0

### 4.1 Overview

**SQuAD (Stanford Question Answering Dataset)** is one of the most widely used benchmarks in NLP for reading comprehension and question answering. Created by Pranav Rajpurkar et al. at Stanford University, it provides a standardized way to train and evaluate systems that work with text-based questions and answers.

| Property | SQuAD 1.1 | SQuAD 2.0 |
|----------|-----------|-----------|
| Release Year | 2016 | 2018 |
| Total Questions | ~100,000 | ~150,000 |
| Answerable Questions | ~100,000 | ~100,000 |
| Unanswerable Questions | 0 | ~50,000 |
| Source Text | Wikipedia | Wikipedia |
| Paper | Rajpurkar et al. (2016) [1] | Rajpurkar et al. (2018) [2] |
| Website | [rajpurkar.github.io/SQuAD-explorer](https://rajpurkar.github.io/SQuAD-explorer/) | Same |

### 4.2 Why SQuAD 2.0 Was Given for This Project

SQuAD 2.0 was specifically chosen for this project because:

1. **Rich question-answer pairs:** It contains human-written questions and their exact answer spans within text — perfect for learning question patterns and evaluating generated questions.
2. **Diverse domains:** The passages come from 500+ Wikipedia articles spanning history, science, geography, sports, arts, and more — enabling a domain-independent QG system.
3. **Unanswerable questions (v2.0):** SQuAD 2.0 adds 50K+ unanswerable questions, which teaches us to distinguish between what CAN be asked about a passage and what cannot.
4. **Standardized benchmark:** Results can be compared against published baselines from hundreds of research papers.
5. **Extractive answers:** Each answer is an exact span from the passage (not generative), which aligns perfectly with our extractive answer candidate approach.

### 4.3 Dataset Structure

```json
{
  "version": "v2.0",
  "data": [
    {
      "title": "Article Title",
      "paragraphs": [
        {
          "context": "The passage text...",
          "qas": [
            {
              "question": "Human-written question?",
              "id": "unique_id",
              "answers": [{"text": "answer span", "answer_start": 42}],
              "is_impossible": false
            }
          ]
        }
      ]
    }
  ]
}
```

### 4.4 Dataset Statistics (Our Loaded Data)

| Metric | Train Set | Dev Set |
|--------|-----------|---------|
| Total Articles | 442 | 35 |
| Total Paragraphs | 18,891 | 1,204 |
| Total Questions | 130,319 | 11,873 |
| Answerable | 86,821 | 5,928 |
| Unanswerable | 43,498 | 5,945 |
| File Size | 42.1 MB | 4.4 MB |

### 4.5 How We Use SQuAD in Our Project

| Usage | Description |
|-------|-------------|
| **Pattern Learning** | Study the mapping between entity types and question words (PERSON→Who, GPE→Where, etc.) |
| **Template Extraction** | Analyze sentence structures that frequently lead to questions |
| **Evaluation** | Generate questions from SQuAD passages and compare against human-written questions using BLEU and ROUGE |
| **Sample Data** | Use SQuAD contexts in the web app's "Load Sample" feature |

> **Important:** We do NOT train a neural model on SQuAD. We use it for rule extraction and evaluation only.

### 4.6 SQuAD 2.0 Paper

**"Know What You Don't Know: Unanswerable Questions for SQuAD"**
Pranav Rajpurkar, Robin Jia, Percy Liang — Stanford University (ACL 2018) [2]

Key contribution: Added 50K+ adversarially written unanswerable questions to the original SQuAD. Systems must now determine when a question CANNOT be answered from the passage — a critical real-world capability. This forced the QA community to move beyond simple keyword matching toward deeper text understanding.

---

## 5. Literature Review — Approaches to Question Generation

### 5.1 Rule-Based Approaches

#### 5.1.1 Heilman & Smith (2010) — "Good Question! Statistical Ranking for Question Generation" [3]

**Seminal paper** in rule-based AQG. Introduced the **"overgenerate and rank"** paradigm:

1. Parse input sentences using a syntactic parser
2. Apply hand-coded transformation rules to generate candidate questions:
   - Subject-auxiliary inversion
   - Wh-movement (replacing answer with question word)
   - Do-insertion (adding do/does/did when no auxiliary exists)
3. Generate many candidates (overgenerate)
4. Use a logistic regression ranker trained on human quality judgments to select the best questions

**Relevance to our project:** Our question generation pipeline directly follows this paradigm. We perform NER-based answer extraction, apply subject-auxiliary inversion rules, and use a multi-feature ranker. Our approach modernizes Heilman & Smith's work by using spaCy's neural NER and dependency parser instead of the Stanford parser.

#### 5.1.2 Heilman (2011) — "Automatic Factual Question Generation from Text" [4]

PhD dissertation extending the 2010 paper with:
- Improved sentence simplification
- Better handling of complex sentences with subordinate clauses
- Analysis of question type distribution (Who/What/Where/When)

**Relevance:** Our template-based patterns (definition, temporal, action) address the same challenge of handling diverse sentence structures.

#### 5.1.3 Chali & Hasan (2015) — "Towards Topic-to-Question Generation" [5]

Proposed generating questions from topics rather than single sentences. Used syntactic dependency trees to identify key concepts and produce diverse question types.

**Relevance:** Informed our use of dependency parsing tree analysis for extracting subjects, objects, and prepositional phrases as answer candidates.

### 5.2 Template-Based Approaches

#### 5.2.1 Lindberg et al. (2013) — "Generating Natural Language Questions to Support Learning On-Line" [6]

Used predefined templates with slots filled from the source text:
- "What is [DEFINITION_TERM]?"
- "In what year did [EVENT] occur?"
- "Who [ACTION] [OBJECT]?"

Templates were matched using POS patterns and semantic role labeling.

**Relevance:** Our template module implements similar patterns — definition template ("X is a Y" → "What is X?"), temporal template ("In YYYY, event" → "When did event happen?"), and action template ("Subject verb object" → "What did Subject verb?").

#### 5.2.2 Ali et al. (2010) — "Automatic Question Generation" [7]

Used Named Entity Recognition to determine question types and simple sentence rewriting rules. Demonstrated that NER-based question type selection achieves high accuracy for factual questions.

**Relevance:** Our entity-to-question-word mapping (PERSON→Who, GPE→Where, DATE→When) is directly inspired by this approach.

### 5.3 Syntax-Based Approaches

#### 5.3.1 Mazidi & Nielsen (2014) — "Linguistic Considerations in Automatic Question Generation" [8]

Analyzed the grammatical transformations needed for high-quality QG:
- Subject-auxiliary inversion rules
- Do-support insertion
- Wh-movement constraints
- Handling of passive voice vs. active voice

**Relevance:** Our `_generate_entity_replacement_question()` function implements all four transformations identified in this paper: subject vs. non-subject answer handling, auxiliary movement, do/does/did insertion, and verb base-form conversion.

#### 5.3.2 Yao et al. (2012) — "Semantics-based Question Generation and Implementation" [9]

Combined semantic role labeling with syntactic parsing to generate deeper questions. Used FrameNet and PropBank for semantic analysis.

**Relevance:** While we don't use semantic role labeling, our dependency parse-based approach achieves similar goals by leveraging the `nsubj`, `dobj`, `pobj`, and `ROOT` dependency relations.

### 5.4 Neural / Deep Learning Approaches (For Comparison)

#### 5.4.1 Du et al. (2017) — "Learning to Ask: Neural Question Generation for Reading Comprehension" [10]

First work to apply sequence-to-sequence (Seq2Seq) neural models with attention to QG. Trained on SQuAD, the model learns to generate questions end-to-end.

- **Advantage:** Generates more natural-sounding questions
- **Disadvantage:** Requires GPU training, can hallucinate, is a black box

#### 5.4.2 Zhao et al. (2018) — "Paragraph-Level Neural Question Generation" [11]

Improved on Du et al. by incorporating paragraph-level context and gated attention, achieving better coherence and relevance.

#### 5.4.3 LLM-Based Approaches (GPT, T5, BART)

Modern systems fine-tune large pre-trained models (T5, BART, GPT-2) on SQuAD for QG. While these achieve state-of-the-art BLEU scores, they:
- Require massive compute (GPU/TPU)
- Are not interpretable (black box)
- Can hallucinate (generate facts not in the source)
- Are overkill for structured, factual question generation

**Our approach deliberately avoids these** in favor of interpretable, lightweight, classical methods.

### 5.5 Evaluation Approaches

#### 5.5.1 BLEU — Papineni et al. (2002) [12]

"BLEU: a Method for Automatic Evaluation of Machine Translation" — Originally for machine translation, now standard for any text generation task. Measures n-gram precision between generated and reference text.

#### 5.5.2 ROUGE — Lin (2004) [13]

"ROUGE: A Package for Automatic Evaluation of Summaries" — Measures recall-oriented overlap. We use ROUGE-1 (unigram), ROUGE-2 (bigram), and ROUGE-L (longest common subsequence).

### 5.6 Difficulty Classification and Bloom's Taxonomy

#### 5.6.1 Bloom et al. (1956) — "Taxonomy of Educational Objectives" [14]

The foundational framework for classifying learning objectives into six cognitive levels: Remember, Understand, Apply, Analyze, Evaluate, Create. Widely used in education to design balanced assessments.

#### 5.6.2 Stasaski & Hearst (2017) — "Multiple Choice Question Generation Utilizing an Ontology" [15]

Demonstrated that question difficulty can be estimated from linguistic features of the source sentence (complexity, abstractness, syntactic depth). Informed our difficulty classification approach.

---

## 6. Key Challenges in AQG

| Challenge | Description | How We Address It |
|-----------|-------------|-------------------|
| **Grammaticality** | Generated questions must be grammatically correct | Subject-auxiliary inversion rules, do-support, verb base-form conversion |
| **Relevance** | Questions should be about important content, not trivial details | NER-based answer selection prioritizes named entities (high information content) |
| **Answer Faithfulness** | The answer must actually be present in the source text | Extractive approach — answers are always exact spans from the input |
| **Diversity** | Avoid generating repetitive questions | Deduplication by answer, diversity in question types (Who/What/Where/When) |
| **Question Type Selection** | Choosing the right question word for each answer | Entity type → question word mapping (data-driven from SQuAD analysis) |
| **Ambiguity** | Natural language is inherently ambiguous | Multiple NLP features (POS, NER, dependency) reduce ambiguity |
| **Difficulty Calibration** | Questions should have varying difficulty levels | Multi-feature difficulty classifier (Innovation #1) |
| **Domain Independence** | Should work for any subject/topic | No domain-specific rules — relies on general linguistic features |

---

## 7. Research Gap & Our Proposed Approach

### 7.1 Identified Gaps in Literature

1. **Most modern AQG systems are neural** — they require GPUs, are opaque, and can hallucinate. There is a gap in **modern, well-engineered classical NLP systems** that leverage contemporary NLP tools (spaCy) with traditional methods.

2. **Difficulty estimation is rare** — most AQG papers generate questions but don't classify difficulty. Educators need questions at multiple levels.

3. **Bloom's Taxonomy integration is minimal** — very few systems map questions to cognitive levels. This is a key gap for educational technology.

4. **Web-based interactive interfaces are uncommon** — most AQG systems are offline scripts without user-friendly interfaces for educators.

### 7.2 Our Proposed System

We propose a system that bridges these gaps by combining:

| Component | Approach | Inspired By |
|-----------|----------|-------------|
| Answer Extraction | NER + Noun Phrases + Numbers | Ali et al. (2010) [7] |
| Question Generation | Rule-based transformations + Templates | Heilman & Smith (2010) [3] |
| Linguistic Analysis | spaCy dependency parsing | Mazidi & Nielsen (2014) [8] |
| Question Ranking | 7-feature heuristic scorer | Heilman (2011) [4] |
| Difficulty Classification | Multi-feature linguistic classifier | Stasaski & Hearst (2017) [15] |
| Bloom's Taxonomy | Pattern-based cognitive level mapping | Bloom et al. (1956) [14] |
| Web Interface | Interactive quiz mode with scoring | Novel contribution |

---

## 8. Why This Model and Not Others

### 8.1 Why spaCy's `en_core_web_sm` ?

| Reason | Explanation |
|--------|-------------|
| **Not an LLM** | It's a statistical NLP pipeline — does analysis, not generation. Satisfies the "without LLMs" constraint. |
| **Accurate NER** | 85%+ F1 on standard NER benchmarks (OntoNotes). Correctly identifies PERSON, GPE, DATE, ORG, etc. |
| **Fast** | Processes 10,000+ words/second on CPU. No GPU needed. |
| **Dependency parsing** | Provides full syntactic tree — essential for subject-auxiliary inversion and wh-movement. |
| **Industry standard** | Used in production at companies like Airbnb, Uber, and Bloomberg. |
| **Lightweight** | Only ~12MB — runs on any laptop. |
| **Python ecosystem** | Integrates seamlessly with NLTK, Flask, scikit-learn. |

### 8.2 Why Rule-Based and Not Neural?

| Factor | Rule-Based (Ours) | Neural (Seq2Seq / LLM) |
|--------|-------------------|----------------------|
| **Explainability** | ✅ Can explain why each question was generated | ❌ Black box |
| **Faithfulness** | ✅ Answers always from source text | ❌ Can hallucinate |
| **Hardware** | ✅ Runs on any CPU laptop | ❌ Needs GPU (CUDA) |
| **Training data** | ✅ No training needed | ❌ Needs 100K+ examples |
| **Speed** | ✅ Milliseconds | ❌ Seconds per question |
| **Quality** | ⚠️ Good for factual Qs | ✅ Better for complex Qs |
| **Naturalness** | ⚠️ Sometimes awkward phrasing | ✅ More fluent |

**For an academic project in classical NLP, the rule-based approach is stronger** because:
- Demonstrates deeper understanding of linguistic principles
- Shows mastery of NLP fundamentals (POS, NER, dependency parsing)
- Is fully interpretable and debuggable
- Doesn't rely on pre-trained text generators

### 8.3 Why Not BERT / T5 / GPT for QG?

- **BERT:** A masked language model — designed for understanding, not generation. Not suitable for QG directly.
- **T5:** A sequence-to-sequence model that can do QG, but requires fine-tuning on GPU and is an LLM — violates our project constraint.
- **GPT:** A generative LLM — exactly what we're told NOT to use. Also hallucination-prone.

---

## 9. Survey-to-Implementation Mapping

This section directly connects what we learned from the survey to what we implemented in our project:

| Survey Finding | What We Learned | How We Implemented It |
|---------------|----------------|----------------------|
| Heilman & Smith's "overgenerate and rank" [3] | Generate many candidates → rank by quality | `question_generator.py` generates 20+ candidates → `question_ranker.py` scores and returns top-K |
| NER determines question type (Ali et al.) [7] | PERSON→Who, GPE→Where, DATE→When mapping | `answer_extractor.py` with `ENTITY_QUESTION_MAP` dictionary mapping 18 entity types to question words |
| Subject-auxiliary inversion (Mazidi & Nielsen) [8] | Questions require moving auxiliary before subject | `question_generator.py`: `_find_auxiliary()`, `_find_subject()`, inversion logic in `_generate_entity_replacement_question()` |
| Do-support insertion [8] | When no auxiliary exists, insert do/does/did | `_get_do_form()` determines correct form based on verb tense tag (VBD→did, VBZ→does) |
| Template matching (Lindberg et al.) [6] | Common patterns produce reliable questions | Three templates in `question_generator.py`: definition ("X is Y"), temporal ("In YYYY"), action ("S V O") |
| Difficulty from linguistics (Stasaski & Hearst) [15] | Sentence complexity → question difficulty | `difficulty.py`: 7 features including Flesch readability, clause count, sentence length |
| Bloom's Taxonomy (Bloom, 1956) [14] | Questions should be categorized by cognitive level | `blooms_taxonomy.py`: verb pattern matching + question word mapping to 6 levels |
| BLEU evaluation (Papineni et al.) [12] | Standard metric for generated text quality | `evaluator.py`: BLEU-2 using NLTK's `sentence_bleu()` |
| ROUGE evaluation (Lin, 2004) [13] | Recall-oriented metric for text overlap | `evaluator.py`: ROUGE-1, ROUGE-2, ROUGE-L using `rouge-score` library |
| SQuAD dataset (Rajpurkar et al.) [2] | Gold standard for QA research | `dataset.py`: loads and parses SQuAD 2.0 JSON for evaluation and samples |

---

## 10. References

[1] Rajpurkar, P., Zhang, J., Lopyrev, K., & Liang, P. (2016). **"SQuAD: 100,000+ Questions for Machine Comprehension of Text."** *Proceedings of EMNLP 2016.* https://aclanthology.org/D16-1264/

[2] Rajpurkar, P., Jia, R., & Liang, P. (2018). **"Know What You Don't Know: Unanswerable Questions for SQuAD."** *Proceedings of ACL 2018.* https://aclanthology.org/P18-2124/

[3] Heilman, M., & Smith, N. A. (2010). **"Good Question! Statistical Ranking for Question Generation."** *Proceedings of NAACL-HLT 2010.* https://aclanthology.org/N10-1086/

[4] Heilman, M. (2011). **"Automatic Factual Question Generation from Text."** *PhD Thesis, Carnegie Mellon University.* http://www.cs.cmu.edu/~ark/mheilman/questions/papers/heilman-question-generation-dissertation.pdf

[5] Chali, Y., & Hasan, S. A. (2015). **"Towards Topic-to-Question Generation."** *Computational Linguistics, 41(1).* https://aclanthology.org/J15-1001/

[6] Lindberg, D., Popowich, F., Nesbit, J., & Winne, P. (2013). **"Generating Natural Language Questions to Support Learning On-Line."** *Proceedings of ENLG 2013.* https://aclanthology.org/W13-2114/

[7] Ali, H., Chali, Y., & Hasan, S. A. (2010). **"Automatic Question Generation from Sentences."** *Proceedings of QG 2010.* https://aclanthology.org/W10-4233/

[8] Mazidi, K., & Nielsen, R. D. (2014). **"Linguistic Considerations in Automatic Question Generation."** *Proceedings of ACL 2014 (Student Research Workshop).* https://aclanthology.org/P14-3017/

[9] Yao, X., Bouma, G., & Zhang, Y. (2012). **"Semantics-based Question Generation and Implementation."** *Dialogue & Discourse, 3(2).* https://journals.uic.edu/ojs/index.php/dad/article/view/3667

[10] Du, X., Shao, J., & Cardie, C. (2017). **"Learning to Ask: Neural Question Generation for Reading Comprehension."** *Proceedings of ACL 2017.* https://aclanthology.org/P17-1123/

[11] Zhao, Y., Ni, X., Ding, Y., & Ke, Q. (2018). **"Paragraph-level Neural Question Generation with Maxout Pointer and Gated Self-attention Networks."** *Proceedings of EMNLP 2018.* https://aclanthology.org/D18-1424/

[12] Papineni, K., Roukos, S., Ward, T., & Zhu, W. (2002). **"BLEU: a Method for Automatic Evaluation of Machine Translation."** *Proceedings of ACL 2002.* https://aclanthology.org/P02-1040/

[13] Lin, C.-Y. (2004). **"ROUGE: A Package for Automatic Evaluation of Summaries."** *Text Summarization Branches Out, ACL 2004.* https://aclanthology.org/W04-1013/

[14] Bloom, B. S. (1956). **"Taxonomy of Educational Objectives: The Classification of Educational Goals."** *Handbook I: Cognitive Domain. Longman.*

[15] Stasaski, K., & Hearst, M. A. (2017). **"Multiple Choice Question Generation Utilizing An Ontology."** *Proceedings of BEA Workshop 2017.* https://aclanthology.org/W17-5034/

---

*This survey establishes the theoretical foundation for our Automatic Question Generation system and demonstrates that our implementation is grounded in established NLP research while introducing novel innovations in difficulty estimation, Bloom's Taxonomy classification, and interactive web-based quiz delivery.*
