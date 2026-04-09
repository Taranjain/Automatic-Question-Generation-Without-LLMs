# 🎓 Complete Project Explanation
## Automatic Question Generation from Textbooks — Without LLMs

> **Read this entire document and you'll be able to explain every part of this project and answer ANY question about it.**

---

## 📌 Table of Contents
1. [What This Project Does (Simple English)](#1-what-this-project-does)
2. [What is SQuAD 2.0 Dataset](#2-what-is-squad-20-dataset)
3. [What "Without LLMs" Means](#3-what-without-llms-means)
4. [The Core NLP Concepts Used](#4-the-core-nlp-concepts-used)
5. [How the Pipeline Works (Step by Step)](#5-how-the-pipeline-works-step-by-step)
6. [Each Module Explained](#6-each-module-explained-in-detail)
7. [The 3 Innovations](#7-the-3-innovations)
8. [Evaluation Metrics (How We Measure Quality)](#8-evaluation-metrics)
9. [How the Website Works](#9-how-the-website-works)
10. [Architecture Diagram](#10-architecture-diagram)
11. [Common Viva / Interview Questions & Answers](#11-common-viva-questions--answers)

---

## 1. What This Project Does

**In one line:** You paste a paragraph from any textbook, and the system automatically generates educational questions from it — like a teacher would.

**Example:**

**Input text:**
> "Albert Einstein was born in Ulm, Germany in 1879. He developed the theory of relativity."

**Output questions:**
| # | Question | Answer | Difficulty | Bloom's Level |
|---|----------|--------|------------|---------------|
| 1 | Who was born in Ulm, Germany in 1879? | Albert Einstein | Easy | Remember |
| 2 | Where was Albert Einstein born? | Ulm, Germany | Easy | Remember |
| 3 | When was Albert Einstein born? | 1879 | Easy | Remember |
| 4 | What did he develop? | the theory of relativity | Medium | Remember |

**The key constraint:** We do NOT use any Large Language Models (GPT, BERT, T5, etc.). We use only classical NLP techniques.

---

## 2. What is SQuAD 2.0 Dataset

**SQuAD** = **Stanford Question Answering Dataset**
- Created by Stanford University researchers (Pranav Rajpurkar et al.)
- Website: https://rajpurkar.github.io/SQuAD-explorer/

### What's inside the dataset?

It's a JSON file containing Wikipedia articles broken into:
- **Articles** (e.g., "Normans", "Albert Einstein")
  - **Paragraphs** (text passages from the article)
    - **Questions** written by humans about that paragraph
    - **Answers** (the exact text span in the paragraph that answers the question)
    - **is_impossible** flag (SQuAD 2.0 added unanswerable questions)

### Our dataset files:
| File | Size | Contents |
|------|------|----------|
| `train-v2.0.json` | 42 MB | ~130,000 QA pairs for training |
| `dev-v2.0.json` | 4.4 MB | ~12,000 QA pairs for evaluation |

### What does the JSON look like?

```json
{
  "version": "v2.0",
  "data": [
    {
      "title": "Normans",
      "paragraphs": [
        {
          "context": "The Normans were the people who...",
          "qas": [
            {
              "question": "In what country is Normandy located?",
              "id": "56ddde6b9a695914005b9628",
              "answers": [
                {
                  "text": "France",
                  "answer_start": 159
                }
              ],
              "is_impossible": false
            }
          ]
        }
      ]
    }
  ]
}
```

### How do we use this dataset?

We use it in **two ways**:

1. **Learning patterns:** We study how real questions relate to text (what kind of entity → what question word). The dataset teaches us: if the answer is a PERSON, the question usually starts with "Who". If the answer is a PLACE, the question starts with "Where", etc.

2. **Evaluation:** We generate questions from SQuAD paragraphs and compare our generated questions against the human-written questions to measure quality (using BLEU and ROUGE scores).

> ⚠️ **Important: We do NOT train a neural network on this dataset.** We use it to learn patterns (rules) and to evaluate our output. This is what makes our approach "without LLMs."

---

## 3. What "Without LLMs" Means

### What are LLMs?
LLMs (Large Language Models) are AI models like GPT-4, BERT, T5 that have billions of parameters and are trained on massive amounts of text data. They can generate human-like text.

### Why avoid LLMs?
- **Academic challenge:** It's more impressive to show you understand NLP fundamentals
- **Interpretable:** You can explain exactly WHY each question was generated (no black box)
- **Lightweight:** Runs on any laptop without GPU
- **Novel approach:** Most modern papers use LLMs, so this is a refreshing academic contrast

### What we use instead:
| ❌ NOT used | ✅ Used |
|-------------|---------|
| GPT, ChatGPT | spaCy NLP library |
| BERT, RoBERTa | Named Entity Recognition (NER) |
| T5, BART | Part-of-Speech (POS) tagging |
| Any transformer model | Dependency parsing |
| Any neural network for QG | Rule-based transformations |
| | Template matching |
| | Heuristic scoring |

---

## 4. The Core NLP Concepts Used

### 4.1 Tokenization
**What:** Breaking text into individual words (tokens).
```
"Albert Einstein was born" → ["Albert", "Einstein", "was", "born"]
```

### 4.2 Part-of-Speech (POS) Tagging
**What:** Labeling each word with its grammatical role.
```
Albert    → PROPN  (Proper Noun)
Einstein  → PROPN  (Proper Noun)
was       → AUX    (Auxiliary verb)
born      → VERB   (Verb)
in        → ADP    (Adposition/Preposition)
Germany   → PROPN  (Proper Noun)
```
**Why we need it:** Helps us understand sentence structure for question transformation.

### 4.3 Named Entity Recognition (NER)
**What:** Identifying real-world entities in text and categorizing them.
```
"Albert Einstein was born in Germany in 1879"

Entities found:
  Albert Einstein → PERSON
  Germany         → GPE (Geo-Political Entity = country/city)
  1879            → DATE
```
**Why we need it:** Entities become answer candidates. The entity TYPE tells us which question word to use:
- PERSON → "Who?"
- GPE/LOC → "Where?"
- DATE/TIME → "When?"
- ORG → "What organization?"
- MONEY → "How much?"

### 4.4 Dependency Parsing
**What:** Understanding the grammatical relationships between words in a sentence.
```
"Einstein received the Nobel Prize"

Einstein  ──nsubj──→  received (Einstein is the SUBJECT)
received  ──ROOT───→  (main verb)
Prize     ──dobj───→  received (Prize is the DIRECT OBJECT)
Nobel     ──amod───→  Prize    (Nobel modifies Prize)
the       ──det────→  Prize    (the is a determiner)
```

**Key relationships we use:**
| Dependency | Meaning | Example |
|-----------|---------|---------|
| `nsubj` | Subject | "**Einstein** received" |
| `dobj` | Direct Object | "received **the Nobel Prize**" |
| `ROOT` | Main verb | "Einstein **received** the Prize" |
| `aux` | Auxiliary verb | "Einstein **was** awarded" |
| `prep` | Preposition | "born **in** Germany" |
| `pobj` | Object of preposition | "in **Germany**" |

**Why we need it:** To know what's the subject, what's the object, and how to rearrange the sentence into a question.

### 4.5 Noun Chunks
**What:** Groups of words that form a noun phrase.
```
"The Nobel Prize in Physics" → one noun chunk
"Albert Einstein" → one noun chunk
```
**Why we need it:** These are additional answer candidates beyond named entities.

### 4.6 spaCy
**What:** The NLP library we use. It provides all of the above (tokenization, POS, NER, dependency parsing) in one package.
```python
import spacy
nlp = spacy.load("en_core_web_sm")  # Load English model
doc = nlp("Albert Einstein was born in Germany")  # Process text
```

`en_core_web_sm` is a **pre-trained statistical model** (NOT an LLM) — it's a small pipeline (~12MB) trained on web text to do POS tagging, NER, and dependency parsing. It does NOT generate text.

---

## 5. How the Pipeline Works (Step by Step)

Here is exactly what happens when you paste text and click "Generate":

```
Step 1: INPUT TEXT
"Albert Einstein was born in Ulm, Germany in 1879."

        ↓

Step 2: SENTENCE SEGMENTATION
Split into individual sentences:
  Sentence 1: "Albert Einstein was born in Ulm, Germany in 1879."

        ↓

Step 3: NLP PROCESSING (spaCy)
For each sentence, extract:
  - POS tags: Albert(PROPN) Einstein(PROPN) was(AUX) born(VERB) in(ADP) Ulm(PROPN) ...
  - Named Entities: Albert Einstein(PERSON), Ulm(GPE), Germany(GPE), 1879(DATE)
  - Dependency tree: Einstein→nsubj, was→auxpass, born→ROOT, Germany→pobj
  - Noun chunks: "Albert Einstein", "Ulm", "Germany"

        ↓

Step 4: ANSWER EXTRACTION
Pick potential answers from the sentence:
  Candidate 1: "Albert Einstein" — type: PERSON → question word: "Who"
  Candidate 2: "Ulm"             — type: GPE    → question word: "Where"
  Candidate 3: "Germany"         — type: GPE    → question word: "Where"
  Candidate 4: "1879"            — type: DATE   → question word: "When"

        ↓

Step 5: QUESTION GENERATION
For each answer candidate, transform the sentence into a question:

  Answer: "Albert Einstein" (PERSON → Who)
    Original: "Albert Einstein was born in Ulm, Germany in 1879."
    Replace subject with "Who": "Who was born in Ulm, Germany in 1879?"
    ✅ Result: "Who was born in Ulm, Germany in 1879?"

  Answer: "Germany" (GPE → Where)
    Original: "Albert Einstein was born in Ulm, Germany in 1879."
    Remove "Germany", add "Where" + move auxiliary:
    ✅ Result: "Where was Albert Einstein born in Ulm, in 1879?"

        ↓

Step 6: RANKING & FILTERING
Score each question on quality (0 to 1):
  - Is the answer NOT in the question? (+)
  - Does it start with a proper question word? (+)
  - Is it the right length (4-15 words)? (+)
  - Does it contain a verb? (+)
  - Are there repeated words? (-)

Remove duplicates. Keep top-K questions.

        ↓

Step 7: ENRICHMENT (Innovations)
For each question, classify:
  - Difficulty: Easy / Medium / Hard
  - Bloom's Taxonomy: Remember / Understand / Apply / Analyze

        ↓

Step 8: OUTPUT
Return the final list of questions with all metadata.
```

---

## 6. Each Module Explained in Detail

### 6.1 `dataset.py` — SQuAD 2.0 Loader

**What it does:** Reads the JSON files and extracts data in a usable format.

**Key functions:**
- `load()` — reads both train and dev JSON files
- `extract_qa_pairs()` — returns list of `{context, question, answer, is_impossible}`
- `extract_contexts()` — returns unique paragraphs with their questions
- `get_stats()` — returns counts (articles, paragraphs, questions, answerable, unanswerable)

**When it's used:** During evaluation (comparing our generated questions vs. real SQuAD questions) and in the website's "Load Sample" feature.

---

### 6.2 `preprocessor.py` — spaCy NLP Pipeline

**What it does:** Takes raw text and runs it through spaCy to get all NLP annotations.

**Key function:** `process_text(text)` → returns a `ProcessedText` object with:
- `.sentences` — list of sentences
- `.entities` — named entities with labels
- `.noun_chunks` — noun phrases
- `.get_sentence_data()` — detailed per-sentence breakdown (tokens, POS, deps, root verb, subject)

**The spaCy model:** `en_core_web_sm` — a small English model (~12MB) that can:
- Tokenize text
- POS tag words
- Recognize named entities (PERSON, GPE, DATE, ORG, etc.)
- Parse dependency trees
- Extract noun chunks

> **This is NOT a language generation model.** It only analyzes text structure. It cannot write or generate text.

---

### 6.3 `answer_extractor.py` — Finding Potential Answers

**What it does:** Given a sentence, identifies what words/phrases could be answers to questions.

**Three strategies:**

| Strategy | How it works | Confidence | Example |
|----------|-------------|------------|---------|
| Named Entities | Uses spaCy NER to find PERSON, GPE, DATE, etc. | 0.9 (highest) | "Albert Einstein" → PERSON |
| Noun Phrases | Uses spaCy noun chunks for subjects/objects | 0.7 | "the Nobel Prize" |
| Numbers | Finds numeric tokens | 0.6 | "1879", "$5 million" |

**Entity → Question Word mapping:**
```
PERSON    → "Who"
GPE, LOC  → "Where"
DATE,TIME → "When"
ORG       → "What"
MONEY     → "How much"
CARDINAL  → "How many"
PERCENT   → "How much"
ORDINAL   → "Which"
```

---

### 6.4 `question_generator.py` — The Core Engine

This is the most important file. It generates questions using **two methods**:

#### Method 1: Rule-Based Entity Replacement

**Logic:**
1. Take a declarative sentence
2. Identify the answer to remove
3. Determine if the answer is the SUBJECT or not
4. Apply the correct transformation:

**If answer IS the subject:**
```
Sentence: "Albert Einstein was born in Germany"
Answer:   "Albert Einstein" (PERSON → Who)

Simply replace the subject with the question word:
Result:   "Who was born in Germany?"
```

**If answer is NOT the subject (needs auxiliary inversion):**
```
Sentence: "Einstein was born in Germany"
Answer:   "Germany" (GPE → Where)

Steps:
  1. Remove "Germany" from sentence
  2. Find auxiliary verb ("was")
  3. Move auxiliary before subject (subject-auxiliary inversion)
  4. Add question word at the start
Result:   "Where was Einstein born?"
```

**If there's no auxiliary verb (use do/does/did):**
```
Sentence: "Einstein developed the theory of relativity"
Answer:   "the theory of relativity" (→ What)

Steps:
  1. Remove the answer
  2. Determine tense: "developed" = past → use "did"
  3. Convert verb to base form: "developed" → "develop"
Result:   "What did Einstein develop?"
```

#### Method 2: Template-Based Generation

Uses regex patterns to match common sentence structures:

| Template | Pattern | Example |
|----------|---------|---------|
| Definition | "X is a Y" | "A planet is a celestial body" → "What is a planet?" |
| Temporal | "In YYYY, event" | "In 1879, Einstein was born" → "When was Einstein born?" |
| Action | "Subject verb object" | "Einstein developed relativity" → "What did Einstein develop?" |

---

### 6.5 `question_ranker.py` — Quality Scoring

**What it does:** Scores each question from 0 to 1 based on quality features, then returns only the best ones.

**7 Features used for scoring:**

| # | Feature | Good | Bad |
|---|---------|------|-----|
| 1 | Length | 4-15 words | <4 or >20 words |
| 2 | Starts with question word | "Who was..." | "The was..." |
| 3 | Word uniqueness | All different words | "Who who who?" |
| 4 | Has proper nouns | More specific | Too generic |
| 5 | Answer NOT in question | Clean question | Self-answering |
| 6 | Has a verb | Grammatical | Fragment |
| 7 | Method bonus | Template slightly preferred | — |

**Also does deduplication:** If two questions have the same answer, keep only the highest-scored one.

---

### 6.6 `evaluator.py` — Measuring Quality

**What it does:** Compare our generated questions against real SQuAD questions to see how good our system is.

**Process:**
1. Take a paragraph from SQuAD
2. Generate questions using our pipeline
3. Compare each generated question against ALL reference questions
4. Take the best-matching reference for each generated question
5. Compute BLEU and ROUGE scores (explained in Section 8)

---

## 7. The 3 Innovations

### 🔥 Innovation #1: Difficulty Level Classification (`difficulty.py`)

**What:** Classifies each generated question as **Easy**, **Medium**, or **Hard**.

**How it works:** Uses 7 linguistic features to compute a difficulty score (0 to 1):

| Feature | Easy (low score) | Hard (high score) |
|---------|-----------------|-------------------|
| Flesch Readability | Simple sentence (score > 80) | Complex sentence (score < 20) |
| Sentence length | ≤ 10 words | > 30 words |
| Number of clauses | 0 subordinate clauses | 3+ subordinate clauses |
| Question word | Who, When (factual) | Why, How (analytical) |
| Answer word count | 1 word | 4+ words |
| Prepositional phrases | 0-1 | 3+ |

**Scoring thresholds:**
- Score ≤ 0.33 → **Easy**
- Score 0.34 - 0.66 → **Medium**
- Score > 0.66 → **Hard**

**Flesch Reading Ease** is a formula that measures how easy text is to read:
```
Score = 206.835 - 1.015 × (words/sentences) - 84.6 × (syllables/words)

90-100 → Very easy (5th grade)
60-70  → Standard (8th grade)
0-30   → Very confusing (college graduate)
```

---

### 🔥 Innovation #2: Bloom's Taxonomy Classification (`blooms_taxonomy.py`)

**What is Bloom's Taxonomy?**  
It's a framework created by Benjamin Bloom (1956) that classifies learning objectives into 6 cognitive levels:

```
        ╔═══════════╗
        ║  CREATE   ║  ← Highest (produce new work)
        ╠═══════════╣
        ║ EVALUATE  ║  ← Judge, justify
        ╠═══════════╣
        ║  ANALYZE  ║  ← Compare, examine
        ╠═══════════╣
        ║   APPLY   ║  ← Use in new situations
        ╠═══════════╣
        ║ UNDERSTAND║  ← Explain, summarize
        ╠═══════════╣
        ║ REMEMBER  ║  ← Recall facts (lowest)
        ╚═══════════╝
```

**How we classify:**

| Level | Question indicators | Examples |
|-------|-------------------|----------|
| Remember | Who, What, When, Where, Which | "Who discovered gravity?" |
| Understand | Why, How, Explain, Describe | "Why does gravity exist?" |
| Apply | Apply, Demonstrate, Use, Solve | "How would you apply this formula?" |
| Analyze | Compare, Contrast, Examine, Differentiate | "What is the difference between X and Y?" |
| Evaluate | Evaluate, Justify, Assess, Critique | "Is this approach effective?" |
| Create | Create, Design, Develop, Formulate | "Design a new experiment for..." |

**Classification method:** We check:
1. **Verb patterns** in the question (highest priority) — e.g., "compare" → Analyze
2. **Question phrase patterns** — e.g., "What is the difference between" → Analyze
3. **Question word** (fallback) — e.g., "Who" → Remember

**Why this is innovative:** Most AQG systems just generate questions. We also CATEGORIZE them by cognitive level, which is extremely useful for educators who want to create balanced assessments.

---

### 🔥 Innovation #3: Interactive Quiz Mode (Website)

**What:** Users can take a timed quiz generated from any text they paste.

**Features:**
- Choose number of questions (3, 5, 7, 10)
- Filter by difficulty (Easy/Medium/Hard/All)
- Timer tracks how long you take
- Hint system (shows the source sentence)
- Skip questions
- Results page with:
  - Score and percentage
  - Time elapsed
  - Each question reviewed (correct ✓ / incorrect ✗)
  - Correct answers shown for wrong responses

---

## 8. Evaluation Metrics

### 8.1 BLEU Score (Bilingual Evaluation Understudy)

**What:** Measures how similar our generated question is to the reference (human-written) question by counting matching words/phrases.

**How it works (simplified):**
```
Reference: "In what country is Normandy located?"
Generated: "Where is Normandy located?"

Matching unigrams (single words): "is", "Normandy", "located" = 3 out of 4
Matching bigrams (word pairs): "is Normandy", "Normandy located" = 2 out of 3

BLEU = geometric mean of precision at different n-gram levels
```

**Score range:** 0 to 1 (1 = perfect match)

> We use **BLEU-2** (considers unigrams and bigrams, which is standard for short text like questions)

### 8.2 ROUGE Score (Recall-Oriented Understudy for Gisting Evaluation)

**What:** Measures overlap between generated and reference text, focusing on RECALL (how much of the reference is captured).

**Three variants we use:**
| Metric | What it measures |
|--------|-----------------|
| ROUGE-1 | Overlap of individual words (unigrams) |
| ROUGE-2 | Overlap of word pairs (bigrams) |
| ROUGE-L | Longest Common Subsequence (word order) |

**Example:**
```
Reference: "Who was the leader of the Normans?"
Generated: "Who was the leader?"

ROUGE-1 = matched words / reference words = 4/7 = 0.57
ROUGE-L = longest common subsequence = "Who was the leader" = 4/7 = 0.57
```

### 8.3 Why both BLEU and ROUGE?

- **BLEU** measures PRECISION (of what we generated, how much is correct)
- **ROUGE** measures RECALL (of the reference, how much did we capture)
- Together they give a complete picture of similarity

### 8.4 Expected Scores

For a rule-based system (no LLMs), typical scores are:
- BLEU-2: **0.05 - 0.20** (this is normal — even neural systems rarely exceed 0.30)
- ROUGE-1: **0.15 - 0.35**
- ROUGE-L: **0.10 - 0.30**

> Low scores are expected because questions can be phrased many different ways. "Where was Einstein born?" and "In what city was Einstein born?" convey the same meaning but have low word overlap.

---

## 9. How the Website Works

### Architecture:

```
┌──────────────────────────────────┐
│   BROWSER (http://localhost:3000)│
│   Next.js React App              │
│                                  │
│   Pages:                         │
│   ├─ / (Generate questions)      │
│   ├─ /quiz (Take a quiz)        │
│   └─ /analysis (See metrics)    │
└──────────────┬───────────────────┘
               │ HTTP (fetch)
               ↓
┌──────────────────────────────────┐
│   FLASK API (http://localhost:5001)│
│   Python Backend                  │
│                                   │
│   Endpoints:                      │
│   ├─ POST /api/generate          │
│   ├─ POST /api/quiz              │
│   ├─ POST /api/evaluate          │
│   ├─ GET  /api/stats             │
│   ├─ GET  /api/sample            │
│   └─ GET  /api/health            │
└──────────────┬────────────────────┘
               │ (function calls)
               ↓
┌──────────────────────────────────┐
│   NLP ENGINE (nlp_engine/)        │
│   Pure Python + spaCy             │
│                                   │
│   Pipeline:                       │
│   Text → Preprocess → Extract     │
│   → Generate → Rank → Enrich     │
└──────────────────────────────────┘
```

### How data flows when you click "Generate":

1. **Frontend (Next.js):** User types text in textarea → clicks "Generate" → JavaScript `fetch()` sends POST request to `http://localhost:5001/api/generate` with `{"text": "..."}`

2. **API (Flask):** Receives the request → calls `generate_questions(text)` → calls `filter_questions()` → calls `rank_questions()` → for each question, adds difficulty and Bloom's level → returns JSON response

3. **Frontend:** Receives JSON → renders question cards with badges

### Tech Stack:
| Layer | Technology | Why |
|-------|-----------|-----|
| Frontend | **Next.js 14** (React framework) | Modern, fast, great for academic projects |
| Styling | **Tailwind CSS** | Utility-first CSS for rapid beautiful UI |
| Backend API | **Flask** (Python) | Lightweight, perfect for serving Python NLP |
| CORS | **flask-cors** | Allows frontend (port 3000) to call backend (port 5001) |
| NLP | **spaCy** (`en_core_web_sm`) | Industry-standard NLP pipeline |
| Text Analysis | **textstat** | Readability scores for difficulty classification |
| Evaluation | **nltk** (BLEU), **rouge-score** (ROUGE) | Standard NLP evaluation metrics |

---

## 10. Architecture Diagram

```
                    ┌─────────────────────────────────────────┐
                    │             USER'S BROWSER              │
                    │                                         │
                    │  ┌───────┐ ┌──────┐ ┌──────────┐      │
                    │  │Generate│ │ Quiz │ │ Analysis │      │
                    │  │ Page  │ │ Page │ │   Page   │      │
                    │  └───┬───┘ └──┬───┘ └────┬─────┘      │
                    └──────┼────────┼──────────┼─────────────┘
                           │        │          │
                    ───────┼────────┼──────────┼──── HTTP ────
                           │        │          │
                    ┌──────▼────────▼──────────▼─────────────┐
                    │           FLASK API (:5001)             │
                    │                                         │
                    │  /api/generate  /api/quiz  /api/evaluate│
                    │  /api/stats     /api/sample /api/health │
                    └──────────────────┬──────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────┐
                    │           NLP ENGINE                     │
                    │                                         │
                    │  ┌────────────┐  ┌──────────────────┐  │
                    │  │ Preprocessor│  │ Answer Extractor  │  │
                    │  │  (spaCy)   │  │ (NER, nouns, nums)│  │
                    │  └─────┬──────┘  └────────┬─────────┘  │
                    │        │                   │            │
                    │  ┌─────▼───────────────────▼─────────┐ │
                    │  │      Question Generator            │ │
                    │  │  (Rules + Templates + Dep. Parse)  │ │
                    │  └─────┬─────────────────────────────┘ │
                    │        │                                │
                    │  ┌─────▼─────┐ ┌──────────┐ ┌───────┐ │
                    │  │  Ranker   │ │Difficulty│ │Bloom's│ │
                    │  │(7 features)│ │Classifier│ │Taxon. │ │
                    │  └───────────┘ └──────────┘ └───────┘ │
                    │                                         │
                    │  ┌──────────────────────────┐          │
                    │  │ Evaluator (BLEU + ROUGE)  │          │
                    │  └──────────────────────────┘          │
                    └─────────────────────────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────┐
                    │         SQuAD 2.0 Dataset               │
                    │  train-v2.0.json  │  dev-v2.0.json      │
                    └─────────────────────────────────────────┘
```

---

## 11. Common Viva Questions & Answers

### Q: What model have you used?
**A:** We use **spaCy's `en_core_web_sm`** — a pre-trained statistical NLP model (not a language model) for POS tagging, Named Entity Recognition, and dependency parsing. We do NOT use any language generation model. The question generation is purely rule-based and template-based.

### Q: Did you train any model?
**A:** No, we did not train any neural network or machine learning model for question generation. The spaCy model is pre-trained (downloaded as-is). Our system uses **handcrafted rules** and **linguistic patterns** to generate questions. However, our question **ranker** uses heuristic feature scoring (which could be extended to a trained classifier).

### Q: What is the difference between your approach and using GPT/BERT?
**A:** 
| Aspect | Our Approach (Rule-Based) | LLM Approach (GPT/BERT) |
|--------|--------------------------|------------------------|
| Training | No training needed | Needs fine-tuning on GPU |
| Explainability | Can explain exactly why each question was generated | Black box |
| Speed | Instant (milliseconds) | Slower (seconds per question) |
| Hardware | Runs on any laptop | Needs GPU/cloud |
| Quality | Good for factual questions | Better for complex questions |
| Creativity | Limited to patterns/rules | Can generate novel phrasings |

### Q: Why did you choose SQuAD 2.0?
**A:** SQuAD 2.0 is the gold standard dataset for question answering research. It contains 130K+ human-written questions about Wikipedia passages, with exact answer spans. Version 2.0 also includes **unanswerable questions** (50K+), making it more realistic. We use it for pattern learning and evaluation.

### Q: What is subject-auxiliary inversion?
**A:** It's the grammatical rule that in English questions, the auxiliary verb moves before the subject:
- Statement: "Einstein **was** born in Germany" (subject → aux → verb)
- Question: "Where **was** Einstein born?" (aux → subject → verb)

If there's no auxiliary, we insert do/does/did:
- Statement: "Einstein **developed** relativity"
- Question: "What **did** Einstein **develop**?" (inserted "did", verb → base form)

### Q: How accurate is your system?
**A:** For factual questions (Who/What/Where/When), our system generates grammatically correct questions with high accuracy. BLEU-2 scores are typically 0.05-0.20, which is standard for rule-based systems. The questions are always grounded in the source text (no hallucination, unlike LLMs).

### Q: What are the limitations?
**A:**
1. Cannot generate "Why" or "How" questions well (requires reasoning)
2. Depends on spaCy's NER accuracy (may miss some entities)
3. Grammar can be imperfect for complex sentences
4. Cannot handle ambiguous text or figurative language
5. Limited to English

### Q: What is the innovation in your project?
**A:** Three innovations:
1. **Difficulty Classification** — Uses 7 linguistic features (Flesch readability, clause count, etc.) to classify questions as Easy/Medium/Hard
2. **Bloom's Taxonomy Mapping** — Categorizes questions into cognitive levels (Remember, Understand, Apply, Analyze) using verb and question-word pattern matching
3. **Interactive Quiz Mode** — A web-based quiz system with timer, hints, difficulty filtering, and automatic scoring

### Q: How does difficulty classification work?
**A:** We compute a score from 0 (easiest) to 1 (hardest) using:
- **Flesch Reading Ease** of the source sentence (complex text → harder question)
- **Sentence length** (longer → harder)
- **Number of subordinate clauses** (more clauses → harder)
- **Question word** (Who/When = easy, Why/How = hard)
- **Answer complexity** (multi-word answers = harder)

Score ≤ 0.33 = Easy, 0.34-0.66 = Medium, > 0.66 = Hard

### Q: What is Flesch Reading Ease?
**A:** A formula that gives a readability score from 0-100:
```
Score = 206.835 - 1.015 × (total words / total sentences) - 84.6 × (total syllables / total words)
```
90-100 = 5th grader can read, 0-30 = graduate level. We use this to determine if the source sentence is complex.

### Q: Can this system work for subjects other than English?
**A:** Currently, it's English-only because we use spaCy's English model. But the architecture is language-agnostic — with a different spaCy model (e.g., `de_core_web_sm` for German), the same pipeline could work for other languages with minor rule adjustments.

### Q: What is CORS and why do you need it?
**A:** CORS (Cross-Origin Resource Sharing) is a browser security feature. Our frontend runs on `localhost:3000` and backend on `localhost:5001` — different "origins." Without CORS, the browser blocks the frontend from calling the backend. `flask-cors` adds the necessary headers to allow cross-origin requests.

### Q: Why Flask and not Django?
**A:** Flask is lightweight and perfect for API-only backends. Django is full-featured (ORM, admin panel, auth) which we don't need. Flask lets us focus on the NLP logic with minimal boilerplate.

### Q: What is the Flask API doing?
**A:** It wraps our Python NLP engine as a REST API with JSON endpoints. When the frontend sends text via HTTP POST, Flask receives it, runs the NLP pipeline, and returns the questions as JSON. This separates the NLP logic (Python) from the UI (JavaScript/React).

### Q: What would you improve if you had more time?
**A:**
1. Add constituency parsing for better question structure
2. Train a small ML model (Random Forest) for question ranking
3. Add support for "Why" questions using causal relation extraction
4. Implement paraphrase detection to compare generated vs reference questions semantically (not just word overlap)
5. Add more question templates for different sentence patterns
6. Support PDF/DOCX file upload on the website

---

## Quick Reference Card

```
PROJECT:     Automatic Question Generation from Textbooks (Without LLMs)
DATASET:     SQuAD 2.0 (Stanford Question Answering Dataset)
APPROACH:    Rule-Based + Template-Based NLP (No neural networks for QG)
NLP TOOL:    spaCy (en_core_web_sm) — POS, NER, Dependency Parsing
METRICS:     BLEU-2, ROUGE-1, ROUGE-2, ROUGE-L
INNOVATIONS: Difficulty Detection, Bloom's Taxonomy, Quiz Mode
FRONTEND:    Next.js 14 + Tailwind CSS
BACKEND:     Flask (Python)
RUNS ON:     Any laptop (no GPU needed)
```
