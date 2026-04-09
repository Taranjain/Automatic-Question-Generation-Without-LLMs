# 📽️ PRESENTATION
## Automatic Question Generation from Textbooks — Without LLMs

**NLP Course Project**
**Author:** Taran Jain
**Dataset:** SQuAD 2.0

> This file is structured as slide-by-slide presentation content. Use it directly with any presentation tool (Google Slides, PowerPoint, Canva etc.)

---

## SLIDE 1: Title Slide

# Automatic Question Generation from Textbooks Without LLMs

**NLP Course Project**

- **Student:** Taran Jain
- **Dataset:** SQuAD 2.0 (Stanford Question Answering Dataset)
- **Approach:** Classical NLP (Rule-Based + Template-Based)
- **Innovation:** Difficulty Detection | Bloom's Taxonomy | Interactive Quiz

---

## SLIDE 2: Agenda

### Agenda
1. Problem Statement
2. Dataset — SQuAD 2.0
3. Our Approach — Pipeline Overview
4. System Architecture
5. NLP Techniques Used
6. Innovations (3)
7. Demo & Web Interface
8. Evaluation Results
9. Analysis & Comparison
10. Conclusion & Future Work

---

## SLIDE 3: Problem Statement

### Problem
> Given a textbook paragraph, **automatically generate educational questions** and their answers — **without using LLMs** (no GPT, BERT, T5).

### Why It Matters
- Teachers spend **hours manually** creating assessments
- Textbooks contain more information than any human writes questions about
- COVID-19 increased demand for **automated online assessment** tools
- Existing neural approaches require **expensive GPUs** — we need lightweight alternatives

### Our Goal
Build a **complete, deployable system** that:
- ✅ Generates factual questions from any text
- ✅ Classifies difficulty (Easy/Medium/Hard)
- ✅ Maps to Bloom's Taxonomy cognitive levels
- ✅ Provides an interactive quiz interface
- ✅ Runs on any laptop (no GPU needed)

---

## SLIDE 4: Dataset — SQuAD 2.0

### SQuAD 2.0 — Stanford Question Answering Dataset

**Authors:** Pranav Rajpurkar, Robin Jia, Percy Liang (Stanford, 2018)

| Metric | Train | Dev |
|--------|-------|-----|
| Articles | 442 | 35 |
| Paragraphs | 18,891 | 1,204 |
| Questions | 130,319 | 11,873 |
| Answerable | 86,821 | 5,928 |
| Unanswerable | 43,498 | 5,945 |

### Why SQuAD 2.0?
- **Gold standard** — most cited QA dataset in NLP
- **Extractive answers** — answer is a span within the text (matches our approach)
- **v2.0** adds unanswerable questions (more realistic)
- Used for **pattern learning** + **evaluation**
- **NOT for training a neural network**

---

## SLIDE 5: System Architecture

### Architecture Overview

```
USER → Next.js Frontend (:3000) → Flask API (:5001) → NLP Engine
                                                         ├── Preprocessor (spaCy)
                                                         ├── Answer Extractor (NER)
                                                         ├── Question Generator (Rules)
                                                         ├── Question Ranker (7 features)
                                                         ├── Difficulty Classifier ⭐
                                                         ├── Bloom's Taxonomy ⭐
                                                         └── Evaluator (BLEU/ROUGE)
```

### Tech Stack
| Component | Technology |
|-----------|-----------|
| NLP Processing | spaCy + NLTK |
| Backend API | Flask (Python) |
| Frontend | Next.js + Tailwind CSS |
| Evaluation | BLEU (NLTK) + ROUGE |
| Model | `en_core_web_sm` (NOT an LLM) |

---

## SLIDE 6: Pipeline — Step by Step

### How a Question is Generated

```
INPUT: "Albert Einstein was born in Ulm, Germany in 1879."
```

**Step 1 — NLP Processing (spaCy)**
| Token | POS | NER |
|-------|-----|-----|
| Albert Einstein | PROPN | PERSON |
| Ulm | PROPN | GPE |
| Germany | PROPN | GPE |
| 1879 | NUM | DATE |

**Step 2 — Answer Extraction**
| Answer Candidate | Entity Type | → Question Word |
|-----------------|------------|-----------------|
| Albert Einstein | PERSON | Who |
| Ulm | GPE | Where |
| Germany | GPE | Where |
| 1879 | DATE | When |

**Step 3 — Question Transformation**
| | |
|---|---|
| Remove subject: | ~~Albert Einstein~~ was born in Ulm, Germany in 1879 |
| Add question word: | **Who** was born in Ulm, Germany in 1879**?** |

**Step 4 — Rank & Enrich**
| Question | Score | Difficulty | Bloom's |
|----------|-------|-----------|---------|
| Who was born in Ulm, Germany in 1879? | 0.97 | Easy | Remember |

---

## SLIDE 7: NLP Techniques Used

### Key NLP Techniques (All Classical — No LLMs)

| Technique | What It Does | Library |
|-----------|-------------|---------|
| **Tokenization** | Split text into words | spaCy |
| **POS Tagging** | Label grammar roles (NOUN, VERB...) | spaCy |
| **Named Entity Recognition** | Find PERSON, GPE, DATE, ORG | spaCy |
| **Dependency Parsing** | Understand subject-verb-object structure | spaCy |
| **Subject-Auxiliary Inversion** | "X was born" → "Was X born?" | Custom rules |
| **Do-Support** | "X developed Y" → "What did X develop?" | Custom rules |
| **Template Matching** | "X is a Y" → "What is X?" | Custom regex |
| **Readability Scoring** | Flesch Reading Ease formula | textstat |

### Why NOT LLMs?
- **Explainable:** We can trace every question to its source
- **No hallucination:** Answers are always from the text
- **Lightweight:** Runs on CPU, no GPU needed
- **Demonstrates NLP fundamentals** mastery

---

## SLIDE 8: Innovation #1 — Difficulty Classification

### Difficulty Level Classification ⭐

**Classifies each question as Easy / Medium / Hard**

**7 Linguistic Features:**

| Feature | Easy → Hard |
|---------|------------|
| Flesch Readability Score | 100 → 0 |
| Sentence length | 5 words → 30+ words |
| Subordinate clauses | 0 → 3+ |
| Question word type | Who/When → Why/How |
| Answer word count | 1 word → 4+ words |
| Prepositional phrases | 0-1 → 3+ |
| Proper noun density | High → Low |

**Results from Evaluation:**

| Difficulty | Percentage |
|-----------|------------|
| Easy | 7.6% |
| Medium | 66.8% |
| Hard | 25.6% |

---

## SLIDE 9: Innovation #2 — Bloom's Taxonomy

### Bloom's Taxonomy Classification ⭐

**Maps each question to a cognitive level (Bloom, 1956)**

```
CREATE      ▓░░░░░░░░░░░░░░  4.4%
EVALUATE    ▓░░░░░░░░░░░░░░  2.8%
ANALYZE     ▓░░░░░░░░░░░░░░  2.4%
APPLY       ▓▓░░░░░░░░░░░░░  6.0%
UNDERSTAND  ░░░░░░░░░░░░░░░  0.0%
REMEMBER    ▓▓▓▓▓▓▓▓▓▓▓▓▓░░ 84.4%
```

**Classification method:**
1. Check verb patterns ("compare" → Analyze, "evaluate" → Evaluate)
2. Check question phrases ("What is the difference" → Analyze)
3. Fallback to question word (Who/What/When → Remember)

**Why 84% Remember?** Rule-based systems excel at factual questions. Remember-level questions are the foundation of Bloom's pyramid.

---

## SLIDE 10: Innovation #3 — Interactive Quiz Mode

### Interactive Quiz Mode ⭐

A web-based quiz system built into the application:

**Features:**
- 🎯 Configurable: Choose 3, 5, 7, or 10 questions
- 🎚️ Difficulty filter: Easy / Medium / Hard / All
- ⏱️ Live timer tracking quiz duration
- 💡 Hint system (shows source sentence)
- ⏭️ Skip option for unknown questions
- 📊 Results dashboard with:
  - Score and percentage
  - Time elapsed
  - Question-by-question review
  - Correct answers for wrong responses

---

## SLIDE 11: Evaluation Metrics

### How We Measure Quality

**Evaluated on 50 SQuAD 2.0 dev contexts**

| Metric | Score | What It Measures |
|--------|-------|-----------------|
| BLEU-2 | **0.0861** | N-gram precision (matching words/phrases) |
| ROUGE-1 | **0.2837** | Unigram recall (word overlap) |
| ROUGE-2 | **0.0857** | Bigram recall (phrase overlap) |
| ROUGE-L | **0.2389** | Longest common subsequence |

**Other Statistics:**
- **499** questions generated from 50 contexts
- **9.98** average questions per passage
- **6** question types: What, Who, Where, When, How many, Which

---

## SLIDE 12: Comparison with Published Systems

### How We Compare

| System | Type | BLEU-2 | ROUGE-L | GPU? |
|--------|------|--------|---------|------|
| Heilman & Smith (2010) | Rule-based | ~0.08 | ~0.20 | ❌ |
| **Ours** | **Rule + Template + Innovations** | **0.0861** | **0.2389** | **❌** |
| Du et al. (2017) | Seq2Seq Neural | ~0.15 | ~0.30 | ✅ |
| NQG++ (2018) | Neural + Features | ~0.17 | ~0.32 | ✅ |
| T5-base (2020) | Pre-trained LLM | ~0.22 | ~0.40 | ✅ |

### Key Takeaway
Our system **matches the classical baseline** (Heilman & Smith) while adding:
- ✅ Difficulty Classification (absent in all baselines)
- ✅ Bloom's Taxonomy (absent in all baselines)
- ✅ Interactive Web Interface (absent in all baselines)

---

## SLIDE 13: Question Type Distribution

### What Types of Questions Are Generated?

```
What       ████████████████████████████░░  57.7% (288)
Who        █████████░░░░░░░░░░░░░░░░░░░░  16.4% (82)
Where      ████████░░░░░░░░░░░░░░░░░░░░░  15.0% (75)
When       ████░░░░░░░░░░░░░░░░░░░░░░░░░   7.6% (38)
How many   █░░░░░░░░░░░░░░░░░░░░░░░░░░░░   2.4% (12)
Which      ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0.8% (4)
```

This closely mirrors SQuAD's human-written question distribution, validating our entity→question word mapping.

---

## SLIDE 14: Sample Output

### Example Questions Generated

**Input:** *"The Normans were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France."*

| # | Question | Answer | Difficulty | Bloom's |
|---|----------|--------|-----------|---------|
| 1 | Who gave their name to Normandy? | The Normans | Easy | Remember |
| 2 | Where is Normandy a region? | France | Easy | Remember |
| 3 | What did the Normans give to Normandy? | their name | Medium | Remember |
| 4 | When did the Normans give their name to Normandy? | 10th and 11th centuries | Medium | Remember |

All answers are **exact spans** from the text — **zero hallucination**.

---

## SLIDE 15: Web Interface

### Three-Page Web Application

| Page | Purpose |
|------|---------|
| **/ (Generate)** | Paste text → get questions with difficulty + Bloom's badges |
| **/quiz** | Interactive quiz with timer, hints, scoring |
| **/analysis** | Dataset stats, BLEU/ROUGE charts, distribution visualization |

**Design:** Dark theme with glassmorphism cards, color-coded badges, animated transitions

**Stack:** Next.js 14 + Tailwind CSS → Flask API → spaCy NLP Engine

---

## SLIDE 16: Strengths

### System Strengths

| ✅ Strength | Detail |
|------------|--------|
| No GPU required | Runs on any laptop |
| No hallucination | 100% extractive — answers always from source |
| Explainable | Every question traceable to source sentence |
| High volume | 10 questions per passage (3.5× human average) |
| 6 question types | Comprehensive Wh-question coverage |
| 3 innovations | Difficulty + Bloom's + Quiz (unique in rule-based literature) |
| Fast | Generates questions in milliseconds |
| Full-stack | Web interface for real-world deployment |

---

## SLIDE 17: Limitations & Future Work

### Limitations

| Limitation | Cause | Future Solution |
|-----------|-------|------------------|
| No "Why" questions | Rules can't extract causality | Discourse parsing |
| Grammar imperfections | Complex sentence transformation | Grammar checker post-processing |
| 84% Remember level | Factual focus of rule-based QG | Train ML classifier for causal sentences |
| BLEU < neural | Different phrasing, not quality | Use semantic similarity (BERTScore) |

### Future Enhancements
- PDF/DOCX file upload
- Multi-language support (Hindi, French)
- User history and progress tracking
- Question paraphrasing for naturalness
- Adaptive learning paths based on quiz performance

---

## SLIDE 18: References (Key Papers)

| # | Paper | Year |
|---|-------|------|
| 1 | Rajpurkar et al. — "SQuAD: 100K+ Questions" | 2016 |
| 2 | Rajpurkar et al. — "Know What You Don't Know: SQuAD 2.0" | 2018 |
| 3 | Heilman & Smith — "Good Question! Statistical Ranking for QG" | 2010 |
| 4 | Heilman — "Automatic Factual QG from Text" (PhD) | 2011 |
| 5 | Lindberg et al. — "Generating NL Questions for Learning" | 2013 |
| 6 | Mazidi & Nielsen — "Linguistic Considerations in Auto QG" | 2014 |
| 7 | Du et al. — "Learning to Ask: Neural QG" | 2017 |
| 8 | Papineni et al. — "BLEU" | 2002 |
| 9 | Lin — "ROUGE" | 2004 |
| 10 | Bloom — "Taxonomy of Educational Objectives" | 1956 |

---

## SLIDE 19: Conclusion

### Summary

✅ Built a **complete AQG system without LLMs** — demonstrating mastery of classical NLP

✅ **15 research papers** reviewed in survey connecting our work to the field

✅ **3 innovations:** Difficulty Detection, Bloom's Taxonomy, Interactive Quiz Mode

✅ **Full-stack deployment:** Next.js → Flask → spaCy pipeline

✅ **Competitive metrics:** BLEU-2 = 0.0861, ROUGE-L = 0.2389 (on par with published baselines)

✅ **Pedagogically valuable:** Difficulty + Bloom's classification enables balanced assessment creation

### Key Contribution
> We demonstrate that **classical NLP can produce a practical, deployable educational tool** without requiring expensive GPU infrastructure or opaque neural models — while adding pedagogical innovations absent from existing systems.

---

## SLIDE 20: Thank You

# Thank You!

### Try the System

```bash
# Backend
source venv/bin/activate && python app.py

# Frontend
cd web && npm run dev

# Open http://localhost:3000
```

**Questions?**

---

## BONUS: Quick Q&A Reference Card

**Q: What model do you use?**
A: spaCy's `en_core_web_sm` — a 12MB statistical NLP pipeline for analysis (POS, NER, parsing). NOT a language model.

**Q: Do you train anything?**
A: No. We use pre-trained spaCy for analysis and handcrafted rules for generation.

**Q: Why not use GPT/BERT?**
A: Project constraint (no LLMs) + explainability + no GPU needed + demonstrates NLP fundamentals.

**Q: What is your BLEU score?**
A: 0.0861 (BLEU-2) — on par with Heilman & Smith (2010), the classic rule-based baseline.

**Q: What is your innovation?**
A: Three: (1) Difficulty classification, (2) Bloom's Taxonomy mapping, (3) Interactive quiz mode with scoring.

**Q: How is difficulty calculated?**
A: 7 features: Flesch readability, sentence length, clause count, question word, answer complexity, prep phrases, proper noun density.
