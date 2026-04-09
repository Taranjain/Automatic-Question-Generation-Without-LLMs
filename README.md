# Automatic Question Generation from Textbooks — Without LLMs

A complete NLP-powered question generation system built using classical NLP techniques (no LLMs). Generates educational questions from any text passage with difficulty levels, Bloom's Taxonomy categorization, and interactive quiz mode.

## 🏗️ Architecture

```
┌─────────────────────┐     ┌──────────────────────┐
│   Next.js Frontend  │────▶│   Flask API (Python)  │
│   (Port 3000)       │     │   (Port 5000)         │
└─────────────────────┘     └──────────┬───────────┘
                                       │
                            ┌──────────▼───────────┐
                            │   NLP Engine          │
                            │  ├─ Dataset Loader    │
                            │  ├─ Preprocessor      │
                            │  ├─ Answer Extractor  │
                            │  ├─ Question Generator│
                            │  ├─ Question Ranker   │
                            │  ├─ Difficulty Clf    │
                            │  ├─ Bloom's Taxonomy  │
                            │  └─ Evaluator         │
                            └──────────────────────┘
```

## 🚀 Quick Start

### 1. Setup Python Backend
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies (already done if following setup)
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Start the API server
python app.py
```

### 2. Setup Next.js Frontend
```bash
cd web
npm install
npm run dev
```

### 3. Open in Browser
Visit `http://localhost:3000`

## 📊 Dataset

- **SQuAD 2.0** (Stanford Question Answering Dataset)
- `train-v2.0.json` — Training set (~130K QA pairs)
- `dev-v2.0.json` — Dev set (~12K QA pairs)

## 🧠 How It Works (Without LLMs)

1. **Text Processing** — spaCy for tokenization, POS tagging, NER, dependency parsing
2. **Answer Extraction** — Identifies answer candidates using named entities, noun phrases, and numbers
3. **Question Generation** — Rule-based transformations with subject-auxiliary inversion
4. **Template Matching** — Predefined linguistic templates for definition, temporal, and action questions
5. **Question Ranking** — Heuristic scoring based on grammaticality, length, and relevance
6. **Quality Filtering** — Removes duplicates, fragments, and self-answering questions

## 🔥 Innovations

### 1. Difficulty Level Detection
Classifies questions as **Easy / Medium / Hard** using:
- Flesch readability score
- Syntactic clause depth
- Answer type complexity
- Question word difficulty

### 2. Bloom's Taxonomy Categorization
Maps questions to cognitive levels: **Remember → Understand → Apply → Analyze → Evaluate → Create**

### 3. Interactive Quiz Mode
Generates timed quizzes from any text with scoring and difficulty filtering.

## 📈 Evaluation Metrics

Run evaluation against SQuAD ground-truth:
```bash
source venv/bin/activate
python -m nlp_engine.evaluator --sample 50 --split dev
```

Metrics computed: **BLEU-2**, **ROUGE-1/2/L**, question type distribution, difficulty distribution, Bloom's distribution.

## 🛠️ Tech Stack

- **NLP Engine**: Python, spaCy, NLTK
- **API**: Flask
- **Frontend**: Next.js 14, Tailwind CSS
- **Database**: Prisma + SQLite
- **Evaluation**: ROUGE, BLEU (nltk)

## 📁 Project Structure

```
nlp_project/
├── app.py                     # Flask API
├── requirements.txt           # Python dependencies
├── nlp_engine/                # Core NLP engine
│   ├── __init__.py
│   ├── dataset.py             # SQuAD 2.0 loader
│   ├── preprocessor.py        # spaCy NLP pipeline
│   ├── answer_extractor.py    # Answer candidate extraction
│   ├── question_generator.py  # Rule & template-based QG
│   ├── question_ranker.py     # Quality ranking
│   ├── difficulty.py          # Difficulty classification
│   ├── blooms_taxonomy.py     # Bloom's taxonomy
│   └── evaluator.py           # BLEU/ROUGE evaluation
├── web/                       # Next.js frontend
├── train-v2.0.json            # SQuAD training data
└── dev-v2.0.json              # SQuAD dev data
```
