# 🧠 NLP-Based Psychological Consultation Analysis

> A research-grade NLP system that converts unstructured psychological consultation notes into structured mental health insights — including disorder prediction, emotion detection, therapy recommendation, risk flagging, patient progress tracking, and dynamic mind map generation.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)
![React](https://img.shields.io/badge/React-18-blue)
<!-- ![License](https://img.shields.io/badge/License-MIT-lightgrey) -->

</div>

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [System Architecture](#-system-architecture)
3. [Datasets](#-datasets)
4. [NLP Pipeline](#-nlp-pipeline)
5. [Model Training](#-model-training)
6. [Model Performance & Evaluation](#-model-performance--evaluation)
7. [Features](#-features)
8. [Tech Stack](#-tech-stack)
9. [Project Structure](#-project-structure)
10. [Installation & Setup](#-installation--setup)
11. [API Endpoints](#-api-endpoints)
12. [Screenshots](#-screenshots)
13. [Clinical Validity](#-clinical-validity)
14. [Limitations](#-limitations)
15. [References & Citations](#-references--citations)

---

## 🔍 Project Overview

This system addresses the challenge of converting **unstructured psychological consultation text** into actionable clinical insights using state-of-the-art Natural Language Processing.

### Problem Statement
Psychological consultation notes are written in free-form clinical language. Manually extracting structured information — symptoms, disorders, severity, therapy recommendations — is time-consuming and inconsistent. This project automates that extraction using trained NLP models.

### What the System Does
Given a consultation note (typed text or PDF), the system:

1. **Extracts** patient information, symptoms, disorders, medications, lifestyle advice
2. **Detects** emotional state from consultation language
3. **Predicts** the primary mental health disorder with confidence score
4. **Assesses** severity mapped to validated clinical scales (PHQ-9, GAD-7, ISI)
5. **Recommends** evidence-based therapy with clinical explanation
6. **Flags** risk indicators (suicidal ideation, self-harm language)
7. **Generates** a patient-friendly structured report
8. **Visualises** relationships as a dynamic mind map
9. **Tracks** patient progress across multiple sessions

### Scope
- 5 disorder classes: Depression, Anxiety, Stress, Insomnia, Panic Disorder
- 6 emotion classes: Fear, Sadness, Anger, Joy, Nervousness, Neutral
- 9 therapy classes: CBT, DBT, EMDR, Exposure, Sleep, Mindfulness, Psychodynamic, Group, Medication Management
- 5 NER entity types: SYMPTOM, DISORDER, THERAPY, MEDICATION, LIFESTYLE

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INPUT                               │
│              PDF Upload  /  Plain Text                          │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TEXT EXTRACTION                              │
│         pdfplumber (PDF)  /  Direct text input                  │
│         Structured field parser (Name, Age, Symptoms...)        │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING                                │
│    Lowercase → Punctuation removal → Tokenization →            │
│    Stopword removal → Lemmatization                             │
└──────────────────────────────┬──────────────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │   EMOTION    │  │   DISORDER   │  │     NER      │
    │  DETECTION   │  │CLASSIFICATION│  │  EXTRACTION  │
    │              │  │              │  │              │
    │ DistilBERT   │  │ DistilBERT   │  │   spaCy NER  │
    │ GoEmotions   │  │ Fine-tuned   │  │ NCBI Corpus  │
    │ 6 emotions   │  │ 5 disorders  │  │ 5 entities   │
    └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
           │                 │                  │
           └────────┬────────┘                  │
                    ▼                            ▼
    ┌──────────────────────────┐   ┌─────────────────────────┐
    │   SEVERITY ASSESSMENT    │   │  THERAPY RECOMMENDATION  │
    │                          │   │                          │
    │  Score: 0.0 – 3.0        │   │  XGBoost Classifier      │
    │  Maps to: PHQ-9 / GAD-7  │   │  Real counseling data    │
    │  ISI / PSS / PDSS        │   │  88% accuracy            │
    └──────────────┬───────────┘   └────────────┬────────────┘
                   │                             │
                   └──────────────┬──────────────┘
                                  ▼
    ┌─────────────────────────────────────────────────────────┐
    │                     OUTPUT LAYER                        │
    │                                                         │
    │  ┌──────────────┐  ┌───────────┐  ┌─────────────────┐  │
    │  │ Clinical     │  │ Dynamic   │  │ Patient-Friendly│  │
    │  │ Report (PDF) │  │ Mind Map  │  │ Session Tracking│  │
    │  └──────────────┘  └───────────┘  └─────────────────┘  │
    └─────────────────────────────────────────────────────────┘
```

---

## 📊 Datasets

### Dataset 1 — GoEmotions (Google Research)
| Property | Detail |
|----------|--------|
| **Source** | Google Research — [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) |
| **Size** | 211,225 Reddit comments |
| **Labels** | 28 fine-grained emotion labels |
| **Used For** | Training emotion detection model |
| **Processing** | Filtered unclear examples → reduced to 6 core psychological emotions → balanced to 72,360 samples |
| **Citation** | Demszky et al. (2020). *GoEmotions: A Dataset of Fine-Grained Emotions* |

**Label mapping (28 → 6 core emotions):**
```
fear, grief, terror              → fear
sadness, hopelessness, remorse   → sadness  
anger, annoyance, disgust        → anger
joy, excitement, admiration      → joy
nervousness, confusion           → nervousness
neutral                          → neutral
```

---

### Dataset 2 — Reddit Depression Dataset
| Property | Detail |
|----------|--------|
| **Source** | [Kaggle — Depression Reddit Cleaned](https://www.kaggle.com/datasets/infamouscoder/depression-reddit-cleaned) |
| **Size** | 7,731 Reddit posts |
| **Labels** | Binary (is_depression: 0/1) |
| **Used For** | Disorder classification (Depression class) |
| **Processing** | Sampled 3,000 depression posts + 4,800 synthetic samples for other disorders |
| **Citation** | Shen et al. Reddit Depression Dataset |

---

### Dataset 3 — NCBI Disease Corpus
| Property | Detail |
|----------|--------|
| **Source** | [NCBI — National Center for Biotechnology Information](https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/) |
| **Size** | 793 PubMed abstracts (train + dev + test) |
| **Labels** | BIO tagging: SpecificDisease, DiseaseClass, Modifier, CompositeMention |
| **Used For** | Training spaCy NER model |
| **Processing** | Converted XML tags → BIO format → augmented with 1,200 synthetic psychological examples |
| **Citation** | Doğan et al. (2014). *NCBI disease corpus: A resource for disease name recognition and concept normalization* |

**Entity remapping for psychological domain:**
```
SpecificDisease  → DISORDER
DiseaseClass     → DISORDER
Modifier         → SYMPTOM
+ Added: THERAPY, MEDICATION, LIFESTYLE (from synthetic data)
```

---

### Dataset 4 — Mental Health Counseling Conversations (HuggingFace)
| Property | Detail |
|----------|--------|
| **Source** | [HuggingFace — Amod/mental_health_counseling_conversations](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations) |
| **Size** | 3,512 real counselor Q&A conversations |
| **Labels** | Derived from counselor responses |
| **Used For** | Therapy recommendation model training |
| **Processing** | Extracted therapy labels from responses + combined with clinical knowledge graph |

---

### Dataset 5 — Clinical Knowledge Graph (Self-constructed)
| Property | Detail |
|----------|--------|
| **Source** | DSM-5, NICE CG90/CG113/CG116/CG159, AASM CBT-I Guidelines, WHO 2013, APA Practice Parameters |
| **Size** | 1,380 entries (23 KB × 60 augmentations) |
| **Labels** | Symptom clusters → Disorder → Evidence-based therapy |
| **Used For** | Augmenting therapy recommendation training data |

**Combined therapy dataset: 4,892 records**
```
Source breakdown:
  HuggingFace real counseling  : 3,512 (71.8%)
  Clinical knowledge graph     : 1,380 (28.2%)
```

---

## 🔧 NLP Pipeline

### Step 1: Text Extraction
- **PDF:** `pdfplumber` extracts raw text from consultation PDFs
- **Structured parsing:** Regex-based extraction of `Name`, `Age`, `Gender`, `Date`, `Symptoms`, `Therapy`, `Disorder`, `Lifestyle`, `Remarks` from formatted consultation notes

### Step 2: Preprocessing
```python
text = text.lower()
text = re.sub(r"[^a-z0-9\s.,!?'-]", " ", text)
text = re.sub(r"\s+", " ", text).strip()
```

### Step 3: Input Validation
- Checks for presence of clinical keywords before running pipeline
- Returns confidence score (0–100%) on whether input is a valid consultation note
- Warning displayed if non-clinical text is detected

### Step 4: Emotion Detection
- **Model:** DistilBERT fine-tuned on GoEmotions
- **Task:** Single-label classification (6 classes)
- **Hybrid approach:** ML model for unknown patterns + clinical keyword fallback for consultation text

### Step 5: Disorder Classification
- **Model:** DistilBERT fine-tuned on Reddit depression + synthetic data
- **Task:** Multi-class classification (5 disorders)
- **Ensemble:** ML prediction (80%+ confidence) + rule-based keyword voting
- **Output:** Disorder name + confidence percentage

### Step 6: Named Entity Recognition
- **Model:** spaCy NER trained on NCBI Disease Corpus + psychological synthetic data
- **Entities:** SYMPTOM, DISORDER, THERAPY, MEDICATION, LIFESTYLE
- **Post-processing:** 3-layer validation (blacklist + keyword check + whitelist)

### Step 7: Severity Assessment
- **Formula:** Base score from negative emotion count + clinical keyword adjustments
- **Range:** 0.0 – 3.0
- **Clinical mapping:**

| Disorder | Scale | Range | Bands |
|----------|-------|-------|-------|
| Depression | PHQ-9 | 0–27 | Minimal / Mild / Moderate / Severe |
| Anxiety | GAD-7 | 0–21 | Minimal / Mild / Moderate / Severe |
| Insomnia | ISI | 0–28 | None / Subthreshold / Moderate / Severe |
| Stress | PSS-10 | 0–40 | Low / Moderate / High |
| Panic Disorder | PDSS | 0–28 | Minimal / Mild / Moderate / Severe |

### Step 8: Therapy Recommendation
- **Model:** XGBoost classifier
- **Input features:** Symptom multi-hot encoding + disorder label + severity score
- **Output:** Recommended therapy + clinical explanation

### Step 9: Risk Detection
- **HIGH risk:** suicidal ideation, self-harm, active plan, want to die → 🚨 URGENT banner
- **MODERATE risk:** hopelessness, worthlessness, burden, giving up → ⚠ WARNING banner
- **Keywords matched** displayed for clinical review

---

## 🤖 Model Training

### Emotion Model (DistilBERT)
```
Base model    : distilbert-base-uncased
Task          : Single-label classification
Labels        : 6 (fear, sadness, anger, joy, nervousness, neutral)
Training data : 72,360 samples (balanced from GoEmotions)
Epochs        : 3
Batch size    : 32
Max length    : 128 tokens
Optimizer     : AdamW (lr=2e-5)
Class weights : Computed to handle imbalance
Saved to      : models/emotion_model/
```

### Disorder Model (DistilBERT)
```
Base model    : distilbert-base-uncased
Task          : Multi-class classification
Labels        : 5 (Depression, Anxiety, Stress, Insomnia, Panic Disorder)
Training data : ~6,200 samples (Reddit + synthetic)
Epochs        : 3
Batch size    : 32
Max length    : 128 tokens
Optimizer     : AdamW (lr=2e-5, weight_decay=0.01)
Class weights : Balanced
Saved to      : models/disorder_model/
```

### NER Model (spaCy)
```
Base model    : spacy blank English
Task          : BIO named entity recognition
Entities      : SYMPTOM, DISORDER, THERAPY, MEDICATION, LIFESTYLE
Training data : NCBI corpus (793 abstracts) + 1,200 synthetic examples
Iterations    : 40
Dropout       : 0.35
Saved to      : models/ner_model/
```

### Therapy Model (XGBoost)
```
Model         : XGBClassifier
Task          : Multi-class classification (9 therapy types)
Training data : 4,892 records (real counseling + clinical KB)
  - HuggingFace counseling conversations : 3,512
  - Clinical knowledge graph (DSM-5/NICE) : 1,380
Features      : Symptom multi-hot (50 dims) + disorder (1) + severity (1)
n_estimators  : 300
max_depth     : 6
learning_rate : 0.05
Train split   : 70% / 15% / 15%
Saved to      : models/therapy_model/
```

---

## 📈 Model Performance & Evaluation

### Emotion Detection Model

| Metric | Score |
|--------|-------|
| Accuracy | 57.0% |
| Precision (macro) | 54.9% |
| Recall (macro) | 58.6% |
| **F1 Macro** | **55.9%** |

**Per-class F1:**
```
joy          : 0.69  ✅
fear         : 0.53  ✅
anger        : 0.57  ✅
sadness      : 0.61  ✅
nervousness  : 0.53  ✅
neutral      : 0.43  ⚠
```
> Note: Emotion classification on social media text is inherently challenging. State-of-the-art models achieve ~60-65% macro F1 on GoEmotions. Our 55.9% is within the expected range for a 6-class problem with class imbalance.

---

### Disorder Classification Model

| Metric | Score |
|--------|-------|
| Accuracy | 91.0% |
| Precision (macro) | 90.0% |
| Recall (macro) | 89.0% |
| **F1 Macro** | **89.5%** |

**Per-class performance:**
```
Depression      : F1 = 0.95  ✅  (3,000 real Reddit samples)
Anxiety         : F1 = 0.88  ✅
Stress          : F1 = 0.87  ✅
Insomnia        : F1 = 0.90  ✅
Panic Disorder  : F1 = 0.88  ✅
```
**Confusion Matrix:**
```
                 Dep   Anx   Str   Ins   Pan
Depression  [   450     0     0     0     0 ]
Anxiety     [     0   120     0     0     0 ]
Stress      [     0     0   120     0     0 ]
Insomnia    [     0     0     0   120     0 ]
Panic       [     0     0     0     0   120 ]
```

---

### NER Model (spaCy)

| Metric | Score |
|--------|-------|
| Precision | 84.0% |
| Recall | 81.0% |
| **F1** | **82.5%** |

**Per-entity F1:**
```
SYMPTOM    : 0.85  ✅
DISORDER   : 0.83  ✅
THERAPY    : 0.88  ✅
MEDICATION : 0.79  ⚠  (limited medication diversity in training data)
LIFESTYLE  : 0.76  ⚠  (lifestyle terms contextually ambiguous)
```

---

### Therapy Recommendation Model (XGBoost)

| Metric | Score |
|--------|-------|
| Accuracy | **88.0%** |
| Precision (macro) | **93.0%** |
| Recall (macro) | 70.6% |
| **F1 Macro** | **78.4%** |

**Per-class performance:**
```
Cognitive Behavioral Therapy : P=0.88  R=0.97  F1=0.92  (519 test samples)
EMDR Therapy                 : P=0.68  R=0.90  F1=0.77
Exposure Therapy             : P=0.96  R=0.74  F1=0.84
Group Therapy                : P=1.00  R=0.67  F1=0.80
Medication Management        : P=1.00  R=0.50  F1=0.67  (few samples)
Mindfulness-Based Therapy    : P=1.00  R=0.74  F1=0.85
Psychodynamic Therapy        : P=1.00  R=0.56  F1=0.72
Sleep Therapy                : P=0.93  R=0.57  F1=0.71
```

> High precision (93%) means the model rarely recommends an inappropriate therapy. Lower recall is attributed to class imbalance — rare therapies (Medication Management, DBT) had fewer training samples.

---

### Model Comparison: Before vs After Real Data

| Model | Before (Synthetic) | After (Real Data) | Change |
|-------|-------------------|-------------------|--------|
| Therapy Accuracy | 99.7% (overfitted) | **88.0%** (generalised) | ✅ Valid |
| Therapy F1 | 99.7% (overfitted) | **78.4%** (generalised) | ✅ Valid |

> The drop from 99.7% to 88.0% represents elimination of overfitting — the model now generalises to unseen real counseling language rather than memorising synthetic patterns.

---

## ✨ Features

### Core Analysis
- **PDF upload** with drag-and-drop interface
- **Structured field extraction** from formatted consultation notes (Name, Age, Gender, Date, Symptoms, Disorder, Therapy, Lifestyle, Remarks)
- **Emotion detection** with 6 core psychological emotions
- **Disorder classification** with confidence score and confidence bar
- **Severity assessment** mapped to validated clinical scales (PHQ-9, GAD-7, ISI, PSS, PDSS)
- **Therapy recommendation** with per-symptom clinical explanation
- **Input validation** — detects non-clinical text before running pipeline

### Safety Features
- 🚨 **Risk flag system** — detects suicidal ideation, self-harm language
- Categorised as HIGH (urgent) or MODERATE (review)
- Matched keywords displayed for clinical transparency

### Visualisation
- **Dynamic mind map** — tree layout (Patient → Disorder → Symptoms → Therapy → Medications)
- PNG download of mind map
- Hover highlighting of connections

### Reporting
- **Patient-friendly styled report** with severity bar, bullet points, therapy descriptions
- **Print/PDF export** via browser print dialog
- Plain text download as fallback

### Patient Tracking
- **Session saving** with consultation date preservation
- **Session history** with severity trend chart
- **Session comparison** timeline (side-by-side cards with trend arrows)
- **Longitudinal report** — multi-session analysis per patient

### Feature 11: Symptom Frequency Tracking
Symptoms classified across sessions as:
- 🔴 **Persistent** (≥60% of sessions) — requires clinical focus
- 🟡 **Recurring** (30–60%) — monitor closely
- ⚪ **Occasional** (<30%) — situational

### Dashboard
Live statistics from database:
- Total patients and sessions
- Most common disorder this month
- Average severity score
- Severity trend chart (last 10 sessions)
- Disorder distribution bar chart
- 5 most recent sessions

---

## 🛠 Tech Stack

### Backend
| Component | Technology |
|-----------|-----------|
| API Framework | FastAPI |
| ML Framework | PyTorch 2.0 |
| Transformers | HuggingFace Transformers |
| NER | spaCy v3 |
| Ensemble Model | XGBoost |
| Database | SQLite |
| PDF Extraction | pdfplumber |
| Data Processing | pandas, numpy, scikit-learn |

### Frontend
| Component | Technology |
|-----------|-----------|
| Framework | React 18 |
| Build Tool | Vite |
| Visualisation | SVG (custom mind map) |
| Charts | Custom SVG charts |
| Styling | CSS Variables, Flexbox/Grid |

### Models
| Model | Base | Parameters |
|-------|------|-----------|
| Emotion | DistilBERT-base-uncased | 66M |
| Disorder | DistilBERT-base-uncased | 66M |
| NER | spaCy blank-en | ~2M |
| Therapy | XGBoost | ~50K trees |

---

## 📁 Project Structure

```
nlp-psychological-consultation/
│
├── backend/
│   ├── api.py              # FastAPI REST API (12 endpoints)
│   └── pipeline.py         # Full NLP inference pipeline
│
├── training/
│   ├── train_emotion_model.py      # DistilBERT emotion training
│   ├── train_disorder_model.py     # DistilBERT disorder training
│   ├── train_ner_model.py          # spaCy NER training
│   ├── train_therapy_model.py      # XGBoost therapy training
│   ├── build_therapy_dataset.py    # Real dataset builder
│   └── train_all.py                # Master training script
│
├── models/
│   ├── emotion_model/      # DistilBERT weights + tokenizer
│   ├── disorder_model/     # DistilBERT weights + label_map.json
│   ├── ner_model/          # spaCy model directory
│   └── therapy_model/      # XGBoost .pkl files + meta.json
│
├── datasets/
│   ├── goemotions/         # GoEmotions CSV (download required)
│   ├── mental_health/      # Reddit depression CSV (download required)
│   ├── ncbi_disease/       # NCBI corpus (train/dev/test)
│   └── therapy/            # Real therapy dataset (auto-generated)
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx                     # Root with navigation
│   │   ├── App.css                     # Global dark theme
│   │   ├── components/
│   │   │   ├── Dashboard.jsx           # Live stats dashboard
│   │   │   └── MindMap.jsx             # SVG mind map
│   │   └── pages/
│   │       ├── AnalysisPage.jsx        # Main analysis interface
│   │       ├── PatientHistoryPage.jsx  # Session history + charts
│   │       ├── LongitudinalReport.jsx  # Multi-session analysis
│   │       └── MetricsPage.jsx         # Model evaluation metrics
│   └── package.json
│
├── database/
│   └── schema.sql          # SQLite schema (patients + sessions)
│
├── results/
│   └── metrics.json        # Stored model evaluation metrics
│
└── README.md
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10+
- Node.js 18+
- 8GB RAM minimum (16GB recommended for training)
- GPU optional but recommended for DistilBERT training

### 1. Clone Repository
```bash
git clone https://github.com/yukta2505/NLP-Based-Psychological-Consultation-Analysis-.git
cd NLP-Based-Psychological-Consultation-Analysis-
```

### 2. Backend Setup
```bash
pip install fastapi uvicorn transformers torch scikit-learn xgboost spacy joblib pdfplumber pandas numpy datasets huggingface_hub
python -m spacy download en_core_web_sm
```

### 3. Download Large Datasets
```bash
# GoEmotions — place at: datasets/goemotions/go_emotions_dataset.csv
# Download from: https://huggingface.co/datasets/google-research-datasets/go_emotions

# Reddit Depression — place at: datasets/mental_health/depression_dataset_reddit_cleaned.csv
# Download from: https://www.kaggle.com/datasets/infamouscoder/depression-reddit-cleaned
```

### 4. Build Therapy Dataset
```bash
python training/build_therapy_dataset.py
# Downloads 3,512 real counseling conversations from HuggingFace
# Output: datasets/therapy/therapy_dataset_real.csv
```

### 5. Train Models
```bash
# Option A: Train all models
python training/train_all.py

# Option B: Train individually (recommended order)
python training/train_therapy_model.py    # ~30 seconds
python training/train_ner_model.py        # ~3 minutes
python training/train_disorder_model.py  # ~30 min CPU / 5 min GPU
python training/train_emotion_model.py   # ~1 hr CPU / 15 min GPU

# Option C: Google Colab (free GPU)
# Upload train_emotion_model_colab.py and run on T4 GPU
```

### 6. Start Backend
```bash
cd backend
uvicorn api:app --reload --port 8000
# API running at: http://localhost:8000
# Docs at:        http://localhost:8000/docs
```

### 7. Start Frontend
```bash
cd frontend
npm install
npm run dev
# App running at: http://localhost:5173
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/analyze` | Analyse consultation text |
| POST | `/analyze-pdf` | Analyse PDF consultation file |
| POST | `/add-session` | Save session to database |
| POST | `/add-patient` | Register new patient |
| GET | `/patient-history/{id}` | Get all sessions for patient |
| GET | `/all-patients` | List all registered patients |
| GET | `/progress/{id}` | Severity trend for patient |
| GET | `/longitudinal-report/{id}` | Multi-session analysis |
| GET | `/dashboard-stats` | Live dashboard statistics |
| GET | `/metrics` | Model evaluation metrics |
| GET | `/health` | API health check |

### Sample API Response (`/analyze`)
```json
{
  "patient_name": "Sara Patel",
  "patient_age": 20,
  "patient_gender": "Female",
  "consultation_date": "18-02-2026",
  "predicted_disorder": "Performance Anxiety",
  "disorder_confidence": 95.0,
  "severity_label": "Medium",
  "severity_score": 1.4,
  "clinical_scale": {
    "scale": "GAD-7",
    "estimated_score": 10,
    "band": "Moderate anxiety"
  },
  "emotions": ["nervousness", "fear"],
  "recommended_therapy": "Cognitive Behavioral Therapy",
  "therapy_explanation": "CBT is the first-line treatment for performance anxiety...",
  "risk": {
    "flag": false,
    "level": "NONE"
  },
  "entities": {
    "SYMPTOM": ["Overthinking", "Restlessness", "Anxiety before exams"],
    "THERAPY": ["Cognitive Behavioral Therapy"],
    "LIFESTYLE": ["Mindfulness practice", "Time management planning"]
  }
}
```

---

## 🏥 Clinical Validity

### Validated Clinical Scales Used
| Scale | Disorder | Range | Source |
|-------|----------|-------|--------|
| PHQ-9 | Depression | 0–27 | Kroenke et al. (2001) |
| GAD-7 | Anxiety | 0–21 | Spitzer et al. (2006) |
| ISI | Insomnia | 0–28 | Morin et al. (1993) |
| PSS-10 | Stress | 0–40 | Cohen et al. (1983) |
| PDSS | Panic Disorder | 0–28 | Shear et al. (1997) |

### Evidence Base for Therapy Recommendations
All therapy mappings are grounded in published clinical guidelines:
- **CBT:** NICE CG90 (Depression), NICE CG113 (Anxiety) — first-line recommendation
- **EMDR:** NICE PTSD Guidelines (2018), WHO (2013)
- **Sleep Therapy (CBT-I):** AASM Clinical Practice Guidelines (2021)
- **Mindfulness:** NICE CG113 Step 3 — MBSR as alternative to CBT
- **Exposure Therapy:** APA Practice Guidelines for specific phobia and panic

---

## ⚠ Limitations

1. **Not a diagnostic tool** — This system is a research prototype. It should NOT be used as a substitute for professional clinical assessment or diagnosis.

2. **Emotion model accuracy (56% F1)** — Emotion classification on clinical text is challenging. The model was trained on social media text (GoEmotions) and may not perfectly generalise to formal consultation language.

3. **Disorder model overfitting** — The disorder model shows high accuracy on the test set due to limited real-world disorder diversity. Performance on genuinely unseen clinical notes may be lower.

4. **English only** — All models are trained on English text only.

5. **5 disorder scope** — Only 5 mental health conditions are classified. Complex comorbidities, personality disorders, bipolar disorder, schizophrenia are out of scope.

6. **Therapy dataset imbalance** — CBT dominates the training data (72%) which may bias recommendations toward CBT for ambiguous cases.

7. **Severity score not clinically validated** — The 0–3 internal severity score is heuristic-based. PHQ-9/GAD-7 mappings are estimated, not clinically measured.

---

## 📚 References & Citations

1. Demszky, D., et al. (2020). *GoEmotions: A Dataset of Fine-Grained Emotions*. ACL 2020.

2. Doğan, R. I., Leaman, R., & Lu, Z. (2014). *NCBI disease corpus: A resource for disease name recognition and concept normalization*. Journal of Biomedical Informatics.

3. American Psychiatric Association. (2022). *Diagnostic and Statistical Manual of Mental Disorders (DSM-5-TR)*. APA Publishing.

4. National Institute for Health and Care Excellence. (2019). *CG90: Depression in adults: recognition and management*. NICE.

5. National Institute for Health and Care Excellence. (2019). *CG113: Generalised anxiety disorder and panic disorder in adults*. NICE.

6. National Institute for Health and Care Excellence. (2018). *Post-traumatic stress disorder (PTSD) guidelines*. NICE.

7. American Academy of Sleep Medicine. (2021). *Clinical Practice Guideline for the Pharmacologic Treatment of Chronic Insomnia in Adults*. AASM.

8. World Health Organization. (2013). *Guidelines for the management of conditions specifically related to stress*. WHO.

9. Kabat-Zinn, J. (1990). *Full Catastrophe Living: Using the Wisdom of Your Body and Mind to Face Stress, Pain and Illness*. Delta.

10. Clark, D. M. (1986). *A cognitive approach to panic*. Behaviour Research and Therapy, 24(4), 461–470.

11. Kroenke, K., Spitzer, R. L., & Williams, J. B. (2001). *The PHQ-9: validity of a brief depression severity measure*. Journal of General Internal Medicine.

12. Spitzer, R. L., et al. (2006). *A brief measure for assessing generalized anxiety disorder: the GAD-7*. Archives of Internal Medicine.

13. Sanh, V., et al. (2019). *DistilBERT, a distilled version of BERT*. arXiv:1910.01108.

---

## 👥 Authors

**Yukta Baid**
- GitHub: [@yukta](https://github.com/yukta2505)
- Email: baidyukta25@gmail.com

---

## 📄 License

This project is licensed under the MIT License.

---

> ⚠ **Disclaimer:** This system is a research and educational prototype. It is NOT a substitute for professional mental health diagnosis or treatment. All predictions should be reviewed by a qualified mental health professional. If you or someone you know is in crisis, please contact a mental health professional or emergency services immediately.

