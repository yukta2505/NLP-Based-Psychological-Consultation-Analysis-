"""
backend/pipeline.py
Full NLP inference pipeline — works in FALLBACK MODE if models are not yet trained.
All imports are guarded. No hard crash on missing packages.
"""

import os
import re
import json
import datetime
import traceback
from typing import Optional, List, Dict, Any

# ──────────────────────────────────────────────
# Safe optional imports
# ──────────────────────────────────────────────
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import torch
    from transformers import (
        DistilBertTokenizerFast,
        DistilBertForSequenceClassification,
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

EMOTION_MODEL_PATH  = os.path.join(BASE, "../models/emotion_model")
DISORDER_MODEL_PATH = os.path.join(BASE, "../models/disorder_model")
NER_MODEL_PATH      = os.path.join(BASE, "../models/ner_model")
THERAPY_MODEL_PATH  = os.path.join(BASE, "../models/therapy_model")

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
EMOTION_LABELS = [
    "admiration","amusement","anger","annoyance","approval","caring",
    "confusion","curiosity","desire","disappointment","disapproval","disgust",
    "embarrassment","excitement","fear","gratitude","grief","joy","love",
    "nervousness","optimism","pride","realization","relief","remorse",
    "sadness","surprise","neutral"
]

DISORDER_LABELS = ["Depression", "Anxiety", "Stress", "Insomnia", "Panic Disorder"]

RULE_THERAPY_MAP = {
    "Depression":           "Cognitive Behavioral Therapy",
    "Anxiety":              "Cognitive Behavioral Therapy",
    "Stress":               "Mindfulness-Based Therapy",
    "Insomnia":             "Sleep Therapy",
    "Panic Disorder":       "Exposure Therapy",
    "PTSD":                 "EMDR Therapy",
    "Performance Anxiety":  "Cognitive Behavioral Therapy",
    "Bipolar Disorder":     "Dialectical Behavior Therapy",
    "OCD":                  "Exposure Therapy",
}

# Symptom → therapy reasoning map
SYMPTOM_THERAPY_REASONS = {
    "overthinking":              ("CBT", "addresses cognitive distortions and overthinking patterns"),
    "negative thoughts":         ("CBT", "restructures negative automatic thoughts"),
    "exam anxiety":              ("CBT", "targets performance anxiety through cognitive restructuring"),
    "performance anxiety":       ("CBT", "reduces anticipatory anxiety and self-critical thinking"),
    "restlessness":              ("CBT", "behavioural activation reduces restlessness"),
    "anxiety":                   ("CBT", "evidence-based first-line treatment for anxiety"),
    "worry":                     ("CBT", "worry management and cognitive restructuring"),
    "nervousness":               ("CBT", "systematic desensitisation of nervous triggers"),
    "hopelessness":              ("CBT", "challenges hopeless thinking through behavioural experiments"),
    "worthlessness":             ("CBT", "rebuilds self-worth through cognitive reframing"),
    "sadness":                   ("Psychodynamic Therapy", "explores underlying causes of persistent sadness"),
    "grief":                     ("Psychodynamic Therapy", "processes unresolved grief and loss"),
    "depression":                ("CBT", "structured approach proven effective for depression"),
    "low motivation":            ("CBT", "behavioural activation restores motivation"),
    "social withdrawal":         ("CBT", "gradual social reintegration through exposure tasks"),
    "insomnia":                  ("Sleep Therapy", "sleep restriction and stimulus control restore sleep"),
    "sleepless":                 ("Sleep Therapy", "sleep hygiene and sleep scheduling"),
    "fatigue":                   ("Sleep Therapy", "addresses root causes of fatigue through sleep regulation"),
    "panic attacks":             ("Exposure Therapy", "interoceptive exposure reduces panic frequency"),
    "chest tightness":           ("Exposure Therapy", "bodily sensation exposure reduces fear response"),
    "avoidance":                 ("Exposure Therapy", "gradual exposure hierarchy breaks avoidance cycle"),
    "flashbacks":                ("EMDR Therapy", "reprocesses traumatic memories causing flashbacks"),
    "nightmares":                ("EMDR Therapy", "trauma reprocessing reduces nightmare frequency"),
    "hypervigilance":            ("EMDR Therapy", "trauma-focused processing reduces hyperarousal"),
    "burnout":                   ("Mindfulness-Based Therapy", "mindfulness reduces stress reactivity and burnout"),
    "overwhelmed":               ("Mindfulness-Based Therapy", "present-moment focus reduces overwhelm"),
    "irritability":              ("DBT", "emotion regulation skills reduce irritability"),
    "anger":                     ("DBT", "distress tolerance and emotion regulation"),
    "impulsivity":               ("DBT", "impulse control through DBT skills training"),
}

def explain_therapy(therapy: str, symptoms: List[str], disorder: str) -> str:
    """Generate a clinical explanation for why this therapy was recommended."""
    reasons = []
    therapy_short = {
        "Cognitive Behavioral Therapy":   "CBT",
        "Dialectical Behavior Therapy":   "DBT",
        "Mindfulness-Based Therapy":      "Mindfulness",
        "Exposure Therapy":               "Exposure Therapy",
        "Sleep Therapy":                  "Sleep Therapy",
        "Psychodynamic Therapy":          "Psychodynamic Therapy",
        "EMDR Therapy":                   "EMDR",
        "Group Therapy":                  "Group Therapy",
        "Medication Management":          "Medication Management",
    }.get(therapy, therapy)

    # Match symptoms to reasons
    matched = []
    for sym in symptoms:
        sl = sym.lower()
        for key, (t, reason) in SYMPTOM_THERAPY_REASONS.items():
            if key in sl and therapy_short in (t, therapy):
                matched.append(f"{sym} ({reason})")
                break

    # Disorder-level reason
    disorder_reasons = {
        "Depression":          f"{therapy} is the first-line evidence-based treatment for depression",
        "Anxiety":             f"{therapy} is clinically proven effective for anxiety disorders",
        "Panic Disorder":      f"{therapy} targets panic cycle through exposure and cognitive work",
        "Insomnia":            f"{therapy} addresses sleep onset/maintenance without medication dependency",
        "Stress":              f"{therapy} builds resilience and reduces physiological stress response",
        "PTSD":                f"{therapy} processes traumatic memories safely using bilateral stimulation",
        "Performance Anxiety": f"{therapy} addresses cognitive distortions driving performance fear",
    }

    base_reason = disorder_reasons.get(disorder, f"{therapy} is recommended based on the presenting symptoms and disorder profile")

    if matched:
        symptom_str = "; ".join(matched[:3])
        return f"{base_reason}. Key symptom matches: {symptom_str}."
    return f"{base_reason}."

SEVERITY_WEIGHTS = {
    "fear": 0.9, "grief": 0.95, "sadness": 0.85, "anger": 0.7,
    "nervousness": 0.8, "remorse": 0.7, "disapproval": 0.5,
    "disgust": 0.6, "disappointment": 0.6, "annoyance": 0.4,
    "confusion": 0.3, "surprise": 0.3, "neutral": 0.0,
    "joy": 0.0, "admiration": 0.0, "gratitude": 0.0,
    "love": 0.0, "optimism": 0.1, "amusement": 0.0,
}

# ──────────────────────────────────────────────
# Cached model references
# ──────────────────────────────────────────────
_cache: Dict[str, Any] = {}


def _load_emotion_model():
    if "emotion" in _cache:
        return _cache["emotion"]
    result = (None, None)
    if TORCH_AVAILABLE and os.path.isdir(EMOTION_MODEL_PATH):
        try:
            tok = DistilBertTokenizerFast.from_pretrained(EMOTION_MODEL_PATH)
            mdl = DistilBertForSequenceClassification.from_pretrained(EMOTION_MODEL_PATH)
            mdl.eval()
            result = (mdl, tok)
            print("[pipeline] Emotion model loaded.")
        except Exception as e:
            print(f"[pipeline] Emotion model load failed: {e}")
    _cache["emotion"] = result
    return result


def _load_disorder_model():
    if "disorder" in _cache:
        return _cache["disorder"]
    result = (None, None)
    if TORCH_AVAILABLE and os.path.isdir(DISORDER_MODEL_PATH):
        try:
            tok = DistilBertTokenizerFast.from_pretrained(DISORDER_MODEL_PATH)
            mdl = DistilBertForSequenceClassification.from_pretrained(DISORDER_MODEL_PATH)
            mdl.eval()
            result = (mdl, tok)
            print("[pipeline] Disorder model loaded.")
        except Exception as e:
            print(f"[pipeline] Disorder model load failed: {e}")
    _cache["disorder"] = result
    return result


def _load_ner_model():
    if "ner" in _cache:
        return _cache["ner"]
    result = None
    if SPACY_AVAILABLE and os.path.isdir(NER_MODEL_PATH):
        try:
            result = spacy.load(NER_MODEL_PATH)
            print("[pipeline] NER model loaded.")
        except Exception as e:
            print(f"[pipeline] NER model load failed: {e}")
    _cache["ner"] = result
    return result


def _load_therapy_model():
    if "therapy" in _cache:
        return _cache["therapy"]
    result = (None, None, None, None, None)
    if JOBLIB_AVAILABLE and os.path.isdir(THERAPY_MODEL_PATH):
        try:
            xgb  = joblib.load(os.path.join(THERAPY_MODEL_PATH, "xgb_model.pkl"))
            mlb  = joblib.load(os.path.join(THERAPY_MODEL_PATH, "mlb_symptoms.pkl"))
            le_d = joblib.load(os.path.join(THERAPY_MODEL_PATH, "le_disorder.pkl"))
            le_t = joblib.load(os.path.join(THERAPY_MODEL_PATH, "le_therapy.pkl"))
            with open(os.path.join(THERAPY_MODEL_PATH, "meta.json")) as f:
                meta = json.load(f)
            result = (xgb, mlb, le_d, le_t, meta)
            print("[pipeline] Therapy model loaded.")
        except Exception as e:
            print(f"[pipeline] Therapy model load failed: {e}")
    _cache["therapy"] = result
    return result


# ──────────────────────────────────────────────
# Structured consultation PDF parser
# ──────────────────────────────────────────────
def parse_consultation_fields(text: str) -> Dict[str, Any]:
    """
    Extract structured fields from a consultation note PDF.
    Handles formats like:
        Name: Sara Patel
        Age: 20
        Gender: Female
        Symptoms Observed: ...
        Suggested Therapy: ...
    """
    fields = {
        "name":        None,
        "age":         None,
        "gender":      None,
        "date":        None,
        "symptoms":    [],
        "therapy":     [],
        "lifestyle":   [],
        "disorder":    [],
        "remarks":     None,
    }

    # Name
    m = re.search(r"(?i)name\s*[:\-]\s*(.+)", text)
    if m:
        fields["name"] = m.group(1).strip().split("\n")[0].strip()

    # Age
    m = re.search(r"(?i)age\s*[:\-]\s*(\d+)", text)
    if m:
        fields["age"] = int(m.group(1))

    # Gender
    m = re.search(r"(?i)gender\s*[:\-]\s*(male|female|other|non-binary)", text)
    if m:
        fields["gender"] = m.group(1).strip().title()

    # Date
    m = re.search(r"(?i)date[^:]*[:\-]\s*([\d\-/]+)", text)
    if m:
        fields["date"] = m.group(1).strip()

    # Symptoms — numbered list after "Symptoms"
    sym_block = re.search(
        r"(?i)symptoms?\s*(?:observed|reported|noted|presented)?[:\-]?\s*\n(.*?)(?=\n[A-Z][^\n]*:|$)",
        text, re.DOTALL
    )
    if sym_block:
        raw = sym_block.group(1)
        items = re.findall(r"(?:^|\n)\s*\d+[\.\)]\s*(.+)", raw)
        fields["symptoms"] = [s.strip() for s in items if s.strip()]

    # Therapy — numbered list after "Therapy"
    thy_block = re.search(
        r"(?i)(?:suggested\s+)?therapy\s*(?:approach)?[:\-]?\s*\n(.*?)(?=\n[A-Z][^\n]*:|$)",
        text, re.DOTALL
    )
    if thy_block:
        items = re.findall(r"(?:^|\n)\s*\d+[\.\)]\s*(.+)", thy_block.group(1))
        fields["therapy"] = [s.strip() for s in items if s.strip()]

    # Lifestyle suggestions
    life_block = re.search(
        r"(?i)lifestyle\s*(?:suggestion|advice|recommendation)?[:\-]?\s*\n(.*?)(?=\n[A-Z][^\n]*:|$)",
        text, re.DOTALL
    )
    if life_block:
        items = re.findall(r"(?:^|\n)\s*\d+[\.\)]\s*(.+)", life_block.group(1))
        fields["lifestyle"] = [s.strip() for s in items if s.strip()]

    # Disorder / Possibility
    dis_block = re.search(
        r"(?i)(?:possibility of the issue|diagnosis|disorder)[:\-]?\s*\n(.*?)(?=\n[A-Z][^\n]*:|$)",
        text, re.DOTALL
    )
    if dis_block:
        items = re.findall(r"(?:^|\n)\s*\d+[\.\)]\s*(.+)", dis_block.group(1))
        fields["disorder"] = [s.strip() for s in items if s.strip()]

    # Remarks
    rem = re.search(r"(?i)remarks?\s*[:\-]\s*(.+)", text, re.DOTALL)
    if rem:
        raw = rem.group(1).strip()
        if raw and raw[0] in ('"', "'"):
            raw = raw[1:]
        if raw and raw[-1] in ('"', "'"):
            raw = raw[:-1]
        fields["remarks"] = raw.strip()

    return fields


# ──────────────────────────────────────────────
# Text extraction
# ──────────────────────────────────────────────
def extract_text_from_pdf(path: str) -> str:
    if not PDF_AVAILABLE:
        raise ImportError("pdfplumber not installed. Run: pip install pdfplumber")
    with pdfplumber.open(path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


# ──────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s.,!?'\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ──────────────────────────────────────────────
# Emotion detection
# ──────────────────────────────────────────────

# Clinical keywords that override ML predictions for consultation text
CLINICAL_EMOTION_KEYWORDS = {
    "fear":        ["fear","afraid","scared","terror","frightened","phobia","dread"],
    "sadness":     ["sad","hopeless","worthless","cry","depressed","grief","miserable",
                    "empty","numb","loss of interest","no joy","meaningless"],
    "anger":       ["angry","furious","rage","irritated","frustrated","aggression"],
    "nervousness": ["nervous","anxious","anxiety","worry","overthink","overthinking",
                    "restless","tense","on edge","apprehensive","panic","racing thoughts",
                    "difficulty sleeping","concentration","social withdrawal"],
    "neutral":     [],
}

def detect_emotions(text: str, threshold: float = 0.35) -> List[str]:
    """
    Hybrid approach:
    1. Run rule-based check on clinical keywords first
    2. If strong clinical signals found → use rule-based result
    3. Otherwise → use ML model prediction
    This prevents the model from detecting 'admiration/approval' in clinical notes.
    """
    # Step 1: Check for strong clinical keyword signals
    rule_emotions = _rule_emotions_clinical(text)
    clinical_score = len([e for e in rule_emotions if e != "neutral"])

    # If strong clinical signals (2+ relevant emotions found), trust rule-based
    if clinical_score >= 2:
        print(f"[pipeline] Strong clinical signals found → using rule-based emotions: {rule_emotions}")
        return rule_emotions

    # Step 2: Try ML model
    model, tok = _load_emotion_model()
    if model is not None:
        try:
            import torch.nn.functional as F
            enc = tok(text, return_tensors="pt",
                      truncation=True, max_length=128, padding=True)
            with torch.no_grad():
                logits = model(**enc).logits

            label_map_path = os.path.join(EMOTION_MODEL_PATH, "label_map.json")
            if os.path.exists(label_map_path):
                with open(label_map_path) as f:
                    lm = json.load(f)
                id2label = lm.get("id2label", {})
                probs    = F.softmax(logits, dim=-1).squeeze().tolist()
                if isinstance(probs, float):
                    probs = [probs]
                ml_emotions = [id2label[str(i)] for i, p in enumerate(probs)
                               if p >= threshold and str(i) in id2label]

                # Filter out clinically irrelevant emotions from ML output
                irrelevant = {"admiration","amusement","approval","gratitude",
                              "love","excitement","pride","relief","curiosity",
                              "desire","optimism","caring","realization"}
                ml_emotions = [e for e in ml_emotions if e not in irrelevant]

                if ml_emotions:
                    # Merge ML + any rule-based clinical signals
                    merged = list(set(ml_emotions + rule_emotions))
                    merged = [e for e in merged if e != "neutral"] or ["neutral"]
                    return merged

        except Exception as e:
            print(f"[pipeline] Emotion inference error: {e}")

    # Fallback: rule-based
    return rule_emotions if rule_emotions else ["neutral"]


def _rule_emotions_clinical(text: str) -> List[str]:
    """Clinical keyword-based emotion detection optimised for consultation notes."""
    keyword_map = {
        "fear":        ["fear","afraid","scared","terror","frightened","phobia",
                        "dread","panic attack","chest tightness"],
        "sadness":     ["sad","hopeless","worthless","cry","depressed","grief",
                        "miserable","empty","numb","loss of interest","meaningless",
                        "no pleasure","anhedonia","tearful"],
        "anger":       ["angry","furious","rage","irritated","frustrated",
                        "aggressive","hostility","irritability"],
        "nervousness": ["nervous","anxious","anxiety","worry","overthink",
                        "overthinking","restless","tense","apprehensive",
                        "panic","racing thoughts","difficulty sleeping",
                        "concentration difficulties","social withdrawal",
                        "insomnia","sleepless","unable to sleep"],
        "grief":       ["grief","bereavement","mourning","loss","trauma","ptsd"],
    }
    found = []
    for emotion, keywords in keyword_map.items():
        if any(kw in text for kw in keywords):
            found.append(emotion)
    return found if found else ["neutral"]


def _rule_emotions(text: str) -> List[str]:
    return _rule_emotions_clinical(text)


# ──────────────────────────────────────────────
# Severity score (0.0 – 3.0)
# ──────────────────────────────────────────────
# ── Severity keyword lists ─────────────────────────────────────
SEVERITY_HIGH_KEYWORDS = [
    ("suicidal ideation",        0.5),
    ("passive suicidal",         0.5),
    ("safety plan",              0.4),
    ("referral to psychiatry",   0.4),
    ("crisis",                   0.4),
    ("urgent referral",          0.4),
    ("hospitaliz",               0.5),
    ("psychomotor retardation",  0.4),
    ("inability to perform",     0.3),
    ("unable to function",       0.3),
    ("complete loss of",         0.3),
    ("recurrent",                0.25),
    ("3 to 4 times per week",    0.3),
    ("severe",                   0.3),
    ("significantly impacting",  0.3),
    ("severely impacting",       0.3),
    ("cannot perform basic",     0.3),
    ("persistent",               0.15),
    ("significant fatigue",      0.2),
    ("recurrent thoughts",       0.3),
    ("weight loss",              0.2),
    ("avoids",                   0.15),
    ("avoidance",                0.2),
    ("fear of dying",            0.4),
    ("chest tightness",          0.2),
    ("heart palpitation",        0.2),
    ("anticipatory anxiety",     0.2),
    ("flashback",                0.3),
    ("nightmare",                0.2),
    ("hypervigilance",           0.3),
    ("sleep severely",           0.25),
    ("disrupted",                0.15),
    ("impaired",                 0.2),
    ("poor concentration",       0.15),
    ("loss of interest",         0.2),
    ("loss of motivation",       0.2),
    ("worthless",                0.25),
    ("hopeless",                 0.25),
    ("overwhelmed",              0.2),
    ("burnout",                  0.2),
    ("unable to switch off",     0.2),
    ("excessive workload",       0.2),
    ("gastrointestinal",         0.15),
    ("physical symptoms",        0.1),
    ("difficulty sleeping",      0.15),
    ("concentration",            0.1),
    ("racing thoughts",          0.15),
]

SEVERITY_LOW_KEYWORDS = [
    ("no significant psychiatric", -0.8),
    ("no formal psychiatric",      -0.7),
    ("generally stable",           -0.7),
    ("stable mood",                -0.7),
    ("wellness check",             -0.8),
    ("routine check",              -0.7),
    ("follow-up in 6 months",      -0.6),
    ("mild stress",                -0.5),
    ("occasional",                 -0.4),
    ("sleep is adequate",          -0.5),
    ("appetite normal",            -0.3),
    ("no significant symptoms",    -0.6),
    ("improving",                  -0.3),
    ("well-managed",               -0.4),
    ("continue current",           -0.3),
]

def compute_severity(emotions: List[str], text: str = "") -> float:
    """
    Severity 0.0–3.0:
      Base from negative emotion count + keyword adjustments.
      Low < 1.0 | Medium 1.0–1.99 | High >= 2.0
    """
    clean = text.lower() if text else ""
    neg = [e for e in emotions if e not in ("neutral", "joy")]
    n   = len(neg)
    if n == 0:   base = 0.3
    elif n == 1: base = 1.0
    elif n == 2: base = 1.4
    else:        base = 1.7
    score = base
    for kw, d in SEVERITY_HIGH_KEYWORDS:
        if kw in clean: score += d
    for kw, d in SEVERITY_LOW_KEYWORDS:
        if kw in clean: score += d
    return round(min(max(score, 0.0), 3.0), 2)


def severity_label(score: float) -> str:
    if score < 1.0:
        return "Low"
    if score < 2.0:
        return "Medium"
    return "High"


def map_to_clinical_scales(score: float, disorder: str) -> Dict[str, Any]:
    """
    Map internal severity score (0-3) to clinically validated scales.
    PHQ-9 for Depression, GAD-7 for Anxiety, ISI for Insomnia,
    PSS for Stress, PDSS for Panic Disorder.
    """
    # Normalise disorder name — handle full names like "Generalized Anxiety Disorder"
    d = disorder.lower()
    if any(x in d for x in ["anxiety", "gad", "worry", "panic"]):
        disorder = "Anxiety"
    elif any(x in d for x in ["depress", "phq", "low mood"]):
        disorder = "Depression"
    elif any(x in d for x in ["insomnia", "sleep", "isi"]):
        disorder = "Insomnia"
    elif any(x in d for x in ["stress", "burnout", "pss"]):
        disorder = "Stress"
    elif any(x in d for x in ["panic disorder", "pdss"]):
        disorder = "Panic Disorder"
    elif any(x in d for x in ["ptsd", "trauma", "post-traumatic"]):
        disorder = "Stress"  # Use PSS as closest validated scale

    # Normalise 0-3 to 0-1
    norm = min(score / 3.0, 1.0)

    scales = {
        "Depression": {
            "scale": "PHQ-9",
            "range": "0–27",
            "estimated_score": round(norm * 27),
            "bands": [
                (0,  4,  "Minimal depression"),
                (5,  9,  "Mild depression"),
                (10, 14, "Moderate depression"),
                (15, 19, "Moderately severe depression"),
                (20, 27, "Severe depression"),
            ],
        },
        "Anxiety": {
            "scale": "GAD-7",
            "range": "0–21",
            "estimated_score": round(norm * 21),
            "bands": [
                (0,  4,  "Minimal anxiety"),
                (5,  9,  "Mild anxiety"),
                (10, 14, "Moderate anxiety"),
                (15, 21, "Severe anxiety"),
            ],
        },
        "Insomnia": {
            "scale": "ISI",
            "range": "0–28",
            "estimated_score": round(norm * 28),
            "bands": [
                (0,  7,  "No clinically significant insomnia"),
                (8,  14, "Subthreshold insomnia"),
                (15, 21, "Moderate clinical insomnia"),
                (22, 28, "Severe clinical insomnia"),
            ],
        },
        "Stress": {
            "scale": "PSS-10",
            "range": "0–40",
            "estimated_score": round(norm * 40),
            "bands": [
                (0,  13, "Low stress"),
                (14, 26, "Moderate stress"),
                (27, 40, "High perceived stress"),
            ],
        },
        "Panic Disorder": {
            "scale": "PDSS",
            "range": "0–28",
            "estimated_score": round(norm * 28),
            "bands": [
                (0,  7,  "Minimal panic symptoms"),
                (8,  14, "Mild panic disorder"),
                (15, 21, "Moderate panic disorder"),
                (22, 28, "Severe panic disorder"),
            ],
        },
    }

    # Default fallback
    default_scale = {
        "scale": "Severity Index",
        "range": "0–3",
        "estimated_score": round(score, 2),
        "bands": [
            (0.0, 1.0, "Low severity"),
            (1.0, 2.0, "Moderate severity"),
            (2.0, 3.0, "High severity"),
        ],
    }

    cfg = scales.get(disorder, default_scale)
    est = cfg["estimated_score"]

    # Find band
    band_label = cfg["bands"][-1][2]
    for lo, hi, label in cfg["bands"]:
        if lo <= est <= hi:
            band_label = label
            break

    return {
        "scale":           cfg["scale"],
        "range":           cfg["range"],
        "estimated_score": est,
        "band":            band_label,
        "interpretation":  f"{cfg['scale']} estimated score: {est} / {cfg['range'].split('–')[1]} — {band_label}",
    }


# ──────────────────────────────────────────────
# Disorder classification
# ──────────────────────────────────────────────
def classify_disorder_with_confidence(text: str) -> Dict[str, Any]:
    """
    Returns disorder prediction WITH confidence score and all class probabilities.
    Used internally so analyze() can expose confidence to the frontend.
    """
    rule_result = _rule_disorder(text)

    model, tok = _load_disorder_model()
    if model is not None:
        try:
            enc = tok(text, return_tensors="pt",
                      truncation=True, max_length=128, padding=True)
            with torch.no_grad():
                logits = model(**enc).logits

            import torch.nn.functional as F
            probs = F.softmax(logits, dim=-1).squeeze().tolist()
            if isinstance(probs, float):
                probs = [probs]

            label_map_path = os.path.join(DISORDER_MODEL_PATH, "label_map.json")
            if os.path.exists(label_map_path):
                with open(label_map_path) as f:
                    lm = json.load(f)
                id2label = lm.get("id2label", {})
            else:
                id2label = {str(i): l for i, l in enumerate(DISORDER_LABELS)}

            ml_result = id2label.get(str(int(torch.tensor(probs).argmax())),
                                     DISORDER_LABELS[0])
            ml_conf   = round(float(max(probs)) * 100, 1)

            # Build all-class probability map
            all_probs = {
                id2label.get(str(i), f"Class{i}"): round(float(p) * 100, 1)
                for i, p in enumerate(probs)
            }

            # Ensemble decision
            if ml_result == rule_result:
                final    = ml_result
                conf     = ml_conf
                source   = "ml+rule"
            elif float(max(probs)) > 0.80:
                final    = ml_result
                conf     = ml_conf
                source   = "ml"
            else:
                final    = rule_result
                # Rule-based confidence: ratio of matched keywords
                conf     = min(round(float(max(probs)) * 80, 1), 75.0)
                source   = "rule"

            print(f"[pipeline] Disorder → {final} ({conf}%) [{source}]")

            return {
                "disorder":    final,
                "confidence":  conf,
                "source":      source,
                "all_probs":   all_probs,
            }

        except Exception as e:
            print(f"[pipeline] Disorder inference error: {e}")

    # Rule-based fallback — estimate confidence from keyword density
    scores = {}
    keyword_map = {
        "Depression":    ["depress","hopeless","worthless","suicidal","grief"],
        "Anxiety":       ["anxi","worry","overthink","nervous","restless","panic"],
        "Insomnia":      ["insomnia","awake","sleepless","sleep onset","lying awake"],
        "Stress":        ["stress","overwhelm","burnout","pressure","tension"],
        "Panic Disorder":["panic attack","chest tight","heart palpitation","fear of dying"],
    }
    tl = text.lower()
    for d, kws in keyword_map.items():
        scores[d] = sum(1 for k in kws if k in tl)
    total = sum(scores.values()) or 1
    best_score = scores.get(rule_result, 1)
    rule_conf = round(min((best_score / total) * 100 + 30, 85), 1)

    return {
        "disorder":   rule_result,
        "confidence": rule_conf,
        "source":     "rule",
        "all_probs":  {d: round(s / total * 100, 1) for d, s in scores.items()},
    }


def classify_disorder(text: str) -> str:
    """Simple wrapper — returns just the disorder name."""
    return classify_disorder_with_confidence(text)["disorder"]


def _rule_disorder(text: str) -> str:
    keyword_map = {
        "Depression":     [
            "depress", "hopeless", "worthless", "suicidal", "empty", "numb",
            "grief", "loss of motivation", "loss of interest", "no pleasure",
            "psychomotor", "anhedonia", "tearful", "crying spells", "burden",
            "meaningless", "self-care", "weight loss", "early morning awakening"
        ],
        "Anxiety":        [
            "anxi", "worry", "overthink", "nervous", "restless", "panic",
            "fear", "dread", "social anxiety", "school refusal", "avoids",
            "embarrassment", "negative evaluation", "trembling", "nausea",
            "racing thoughts", "muscle tension", "cognitive restructuring",
            "fearing judgment", "anticipatory", "social withdrawal", "apprehensive"
        ],
        "Insomnia":       [
            "insomnia", "awake", "sleepless", "sleep onset", "waking up",
            "cannot sleep", "lying awake", "sleep schedule", "unrested",
            "sleep restriction", "sleep hygiene", "difficulty initiating sleep",
            "maintaining sleep", "melatonin"
        ],
        "Stress":         [
            "stress", "overwhelm", "burnout", "pressure", "tension", "overwork",
            "workload", "deadlines", "irritable", "gastrointestinal", "headaches",
            "neck tension", "unable to switch off", "delegating"
        ],
        "Panic Disorder": [
            "panic attack", "chest tightness", "heart palpitation", "dizzy",
            "sudden fear", "fear of dying", "interoceptive", "avoids public",
            "crowded places", "anticipatory anxiety about future attacks",
            "unexpected episodes", "alprazolam"
        ],
    }

    # Negative context words — if present, reduce score for that disorder
    negative_context = {
        "Insomnia": ["sleep is adequate", "sleep is good", "sleeping well",
                     "no sleep issues", "7 hours", "8 hours per night", "adequate sleep"],
        "Depression": ["no depression", "not depressed", "mood stable", "mood is stable"],
        "Anxiety": ["no anxiety", "not anxious", "anxiety resolved"],
    }

    scores = {}
    for disorder, keywords in keyword_map.items():
        score = sum(1 for kw in keywords if kw in text)
        # Penalise if negative context found
        for neg in negative_context.get(disorder, []):
            if neg in text:
                score = max(0, score - 3)
        scores[disorder] = score

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Anxiety"


# ──────────────────────────────────────────────
# Entity validation helpers
# ──────────────────────────────────────────────

# Common English words that are NOT medications, symptoms, etc.
# Used to filter out NER false positives
INVALID_ENTITIES = {
    # Common words falsely tagged as medications
    "MEDICATION": {
        "been", "have", "has", "had", "was", "were", "are", "is", "be",
        "will", "would", "could", "should", "may", "might", "can",
        "the", "a", "an", "to", "of", "and", "or", "but", "in", "on",
        "at", "by", "for", "with", "about", "into", "through", "during",
        "before", "after", "above", "below", "between", "each", "every",
        "this", "that", "these", "those", "not", "no", "nor", "so",
        "very", "too", "also", "just", "now", "then", "there", "here",
        "when", "where", "why", "how", "all", "both", "few", "more",
        "most", "other", "some", "such", "only", "own", "same", "than",
        "further", "once", "any", "due", "per", "well", "back", "still",
        "even", "however", "while", "since", "without", "along", "as",
    },
    # Common words falsely tagged as symptoms
    "SYMPTOM": {
        "the", "a", "an", "and", "or", "but", "is", "was", "are", "were",
        "has", "have", "had", "be", "been", "being", "will", "would",
        "patient", "person", "individual", "client", "report", "reports",
        "noted", "observed", "suggested", "recommended", "prescribed",
    },
    # Common words/phrases falsely tagged as therapies
    "THERAPY": {
        "the", "a", "an", "and", "or", "therapy", "treatment", "approach",
        "session", "sessions", "support", "care", "management",
        "difficulty trusting", "difficulty trusting others",
        "trusting", "trusting others", "difficulty", "others",
        "social support", "emotional support", "support system",
        "encouraged", "initiated", "recommended", "suggested",
        "techniques", "approach", "strategies", "methods",
    },
}

# Minimum character length for valid entities per type
MIN_ENTITY_LENGTH = {
    "MEDICATION": 4,
    "SYMPTOM":    4,
    "THERAPY":    4,
    "DISORDER":   4,
    "LIFESTYLE":  4,
}

# Known valid medication names (whitelist)
KNOWN_MEDICATIONS = {
    "sertraline", "fluoxetine", "escitalopram", "citalopram",
    "paroxetine", "fluvoxamine", "venlafaxine", "duloxetine",
    "bupropion", "mirtazapine", "amitriptyline", "nortriptyline",
    "alprazolam", "clonazepam", "lorazepam", "diazepam",
    "buspirone", "quetiapine", "aripiprazole", "risperidone",
    "lithium", "valproate", "lamotrigine", "melatonin",
    "zolpidem", "trazodone", "hydroxyzine", "propranolol",
}

def clean_entity(text: str, label: str) -> Optional[str]:
    """
    Validate and clean a single entity string.
    Returns None if the entity should be discarded.
    """
    val = text.strip()
    if not val:
        return None

    val_lower = val.lower()

    # Too short
    if len(val) < MIN_ENTITY_LENGTH.get(label, 3):
        return None

    # In the invalid set for this label
    if val_lower in INVALID_ENTITIES.get(label, set()):
        return None

    # For MEDICATION: must be in known list OR look like a drug name
    # (longer than 6 chars, not a common word)
    if label == "MEDICATION":
        if val_lower in KNOWN_MEDICATIONS:
            return val.title()
        # Reject if it is a common English word (contains only common letters pattern)
        # Simple check: medications are usually >= 6 chars with no spaces
        if len(val) < 6 or " " in val:
            # Allow multi-word only if it contains a known medication
            if not any(med in val_lower for med in KNOWN_MEDICATIONS):
                return None
        # Reject purely alphabetic short words that look like English words
        common_words = {
            "been", "have", "been", "with", "from", "this", "that",
            "they", "them", "then", "when", "what", "some", "been",
            "take", "took", "does", "done", "make", "made", "help",
        }
        if val_lower in common_words:
            return None
        return val.title()

    # THERAPY: must contain a known therapy keyword to be valid
    if label == "THERAPY":
        therapy_keywords = [
            "therapy", "cbt", "dbt", "emdr", "counseling", "counselling",
            "psychotherapy", "psychodynamic", "behavioral", "behaviour",
            "cognitive", "mindfulness", "exposure", "sleep", "group",
            "medication", "management", "intervention", "treatment",
        ]
        if not any(kw in val_lower for kw in therapy_keywords):
            return None

    # LIFESTYLE: must be an activity or practice, not a symptom description
    if label == "LIFESTYLE":
        lifestyle_keywords = [
            "exercise", "yoga", "meditation", "mindfulness", "journaling",
            "diet", "sleep", "hygiene", "management", "support", "breathing",
            "relaxation", "activity", "routine", "practice", "walk",
        ]
        if not any(kw in val_lower for kw in lifestyle_keywords):
            return None

    # SYMPTOM: must not be a therapy or lifestyle term
    if label == "SYMPTOM":
        not_symptom = [
            "therapy", "cbt", "dbt", "emdr", "counseling", "prescribed",
            "recommended", "suggested", "initiated", "exercise", "yoga",
            "meditation", "management", "mindfulness",
        ]
        if any(kw in val_lower for kw in not_symptom):
            return None

    # Capitalise first letter for display
    return val[0].upper() + val[1:] if val else None


def clean_entities(entities: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Clean all entity lists, removing false positives."""
    cleaned = {}
    for label, items in entities.items():
        seen = set()
        result_list = []
        for item in (items or []):
            clean = clean_entity(item, label)
            if clean and clean.lower() not in seen:
                seen.add(clean.lower())
                result_list.append(clean)
        cleaned[label] = result_list
    return cleaned


# ──────────────────────────────────────────────
# Named Entity Recognition
# ──────────────────────────────────────────────
def extract_entities(text: str) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {
        "SYMPTOM": [], "DISORDER": [], "THERAPY": [],
        "MEDICATION": [], "LIFESTYLE": []
    }
    ner = _load_ner_model()
    if ner is not None:
        try:
            doc = ner(text)
            for ent in doc.ents:
                label = ent.label_
                if label in result:
                    val = ent.text.strip()
                    if val and val not in result[label]:
                        result[label].append(val)
            # Clean NER output before returning
            return clean_entities(result)
        except Exception as e:
            print(f"[pipeline] NER inference error: {e}")
    return clean_entities(_rule_ner(text))


def _rule_ner(text: str) -> Dict[str, List[str]]:
    symptom_kw = [
        "overthinking", "restlessness", "anxiety", "panic", "hopelessness",
        "sadness", "fatigue", "insomnia", "nervousness", "fear", "irritability",
        "worthlessness", "grief", "concentration difficulties", "social withdrawal",
        "low self-esteem", "mood swings", "chest tightness", "night sweats"
    ]
    therapy_kw = [
        "cognitive behavioral therapy", "cognitive behavioural therapy",
        "cbt", "dbt", "dialectical behavior therapy", "dialectical behaviour therapy",
        "exposure therapy", "sleep therapy", "group therapy",
        "emdr therapy", "emdr", "psychodynamic therapy",
        "mindfulness-based therapy", "mindfulness based therapy",
        "medication management", "sleep restriction therapy",
        "interoceptive conditioning", "acceptance and commitment therapy",
        "act therapy", "trauma-focused therapy",
    ]
    disorder_kw = [
        "depression", "anxiety disorder", "insomnia", "panic disorder",
        "ptsd", "bipolar", "ocd", "obsessive compulsive disorder",
        "stress disorder", "major depressive disorder"
    ]
    med_kw = [
        "sertraline", "fluoxetine", "escitalopram", "alprazolam",
        "clonazepam", "lorazepam", "venlafaxine", "bupropion"
    ]
    life_kw = [
        "exercise regularly", "regular exercise", "physical activity",
        "meditation", "journaling", "diet improvement",
        "sleep hygiene", "mindfulness practice", "mindfulness techniques",
        "mindfulness exercises", "yoga", "breathing exercises",
        "time management", "time management planning",
        "social support", "relaxation techniques",
        "regular physical activity", "healthy lifestyle",
    ]

    def find_matches(keywords):
        return [kw for kw in keywords if kw in text]

    return {
        "SYMPTOM":    find_matches(symptom_kw),
        "DISORDER":   find_matches(disorder_kw),
        "THERAPY":    find_matches(therapy_kw),
        "MEDICATION": find_matches(med_kw),
        "LIFESTYLE":  find_matches(life_kw),
    }


# ──────────────────────────────────────────────
# Therapy recommendation
# ──────────────────────────────────────────────
def recommend_therapy(symptoms: List[str], disorder: str, severity_score: float) -> str:
    xgb, mlb, le_d, le_t, meta = _load_therapy_model()
    if xgb is not None and NUMPY_AVAILABLE:
        try:
            known_disorders = meta.get("all_disorders", [])
            all_symptoms    = meta.get("all_symptoms", [])
            disorder_clean  = disorder if disorder in known_disorders else "Anxiety"
            symp_clean      = [s.lower() for s in symptoms if s.lower() in all_symptoms]
            if not symp_clean:
                symp_clean = all_symptoms[:2]
            feat_sym = mlb.transform([symp_clean])
            feat_dis = le_d.transform([disorder_clean]).reshape(-1, 1)
            feat_sev = np.array([[min(int(severity_score) + 1, 3)]])
            X = np.hstack([feat_sym, feat_dis, feat_sev])
            pred_idx = xgb.predict(X)[0]
            return le_t.inverse_transform([pred_idx])[0]
        except Exception as e:
            print(f"[pipeline] Therapy inference error: {e}")
    return RULE_THERAPY_MAP.get(disorder, "Cognitive Behavioral Therapy")


# ──────────────────────────────────────────────
# Mind map
# ──────────────────────────────────────────────
def build_mind_map(
    patient_name: str,
    symptoms: List[str],
    disorder: str,
    therapy: str,
    medications: List[str],
    lifestyle: List[str],
) -> Dict[str, Any]:
    nodes = [{"id": "patient", "label": patient_name, "type": "patient"}]
    edges = []

    nodes.append({"id": "disorder", "label": disorder, "type": "disorder"})
    edges.append({"source": "patient", "target": "disorder"})

    for i, s in enumerate(symptoms[:6]):
        nid = f"symptom_{i}"
        nodes.append({"id": nid, "label": s, "type": "symptom"})
        edges.append({"source": "disorder", "target": nid})

    nodes.append({"id": "therapy", "label": therapy, "type": "therapy"})
    edges.append({"source": "disorder", "target": "therapy"})

    for i, m in enumerate(medications[:3]):
        nid = f"medication_{i}"
        nodes.append({"id": nid, "label": m, "type": "medication"})
        edges.append({"source": "therapy", "target": nid})

    for i, l in enumerate(lifestyle[:3]):
        nid = f"lifestyle_{i}"
        nodes.append({"id": nid, "label": l, "type": "lifestyle"})
        edges.append({"source": "patient", "target": nid})

    return {"nodes": nodes, "edges": edges}


# ──────────────────────────────────────────────
# Report generation
# ──────────────────────────────────────────────
def _dedup(items):
    """Deduplicate list case-insensitively, keep first occurrence capitalised."""
    seen, result = set(), []
    for item in (items or []):
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(item.strip().title())
    return result


def _bullet(items, fallback="None identified"):
    """Format list as bullet points."""
    clean = _dedup(items)
    if not clean:
        return f"  • {fallback}"
    return "\n".join(f"  • {i}" for i in clean)


def _severity_bar(score):
    """ASCII progress bar for severity."""
    filled = round((score / 3.0) * 20)
    bar    = "█" * filled + "░" * (20 - filled)
    return bar


def _severity_message(label, disorder):
    """Human-friendly message about severity."""
    messages = {
        "Low":    f"Your emotional state is relatively stable. While some stressors related to {disorder} are present, they are manageable with the right support.",
        "Medium": f"You are experiencing a moderate level of distress related to {disorder}. With consistent therapy and self-care, significant improvement is expected.",
        "High":   f"You are experiencing significant distress related to {disorder}. Professional support is strongly recommended and early intervention will be most effective.",
    }
    return messages.get(label, "")


def _therapy_description(therapy):
    """Plain-English description of recommended therapy."""
    descriptions = {
        "Cognitive Behavioral Therapy":   "A structured, evidence-based approach that helps identify and change negative thought patterns and behaviours.",
        "Dialectical Behavior Therapy":   "A skills-based therapy focusing on emotional regulation, mindfulness, and building healthy relationships.",
        "Mindfulness-Based Therapy":      "Techniques that cultivate present-moment awareness to reduce stress, anxiety, and emotional reactivity.",
        "Exposure Therapy":               "A gradual, structured approach to help face feared situations and reduce avoidance behaviours.",
        "Sleep Therapy":                  "Behavioural and cognitive strategies to restore healthy sleep patterns and improve sleep quality.",
        "Psychodynamic Therapy":          "Explores unconscious thoughts and past experiences to understand and resolve present emotional difficulties.",
        "Group Therapy":                  "A supportive group environment where shared experiences foster healing and personal growth.",
        "EMDR Therapy":                   "Eye Movement Desensitisation and Reprocessing — helps process and reduce the impact of traumatic memories.",
        "Medication Management":          "Carefully monitored use of prescribed medication to support mental health alongside therapy.",
    }
    return descriptions.get(therapy, "A personalised therapeutic approach tailored to your specific needs and goals.")


def generate_report(analysis: Dict[str, Any]) -> str:
    entities = analysis.get("entities", {})
    parsed   = analysis.get("parsed_fields", {}) or {}

    name     = analysis.get("patient_name", "Unknown")
    age      = analysis.get("patient_age")    or parsed.get("age",    "N/A")
    gender   = analysis.get("patient_gender") or parsed.get("gender", "N/A")
    date     = analysis.get("consultation_date") or parsed.get("date",
               datetime.datetime.now().strftime("%d-%m-%Y"))
    remarks  = analysis.get("remarks") or "No specific remarks recorded."
    disorder = analysis.get("predicted_disorder", "N/A")
    therapy  = analysis.get("recommended_therapy", "N/A")
    sev_score= float(analysis.get("severity_score", 0))
    sev_lbl  = analysis.get("severity_label", "N/A")
    emotions = analysis.get("emotions", [])
    mode     = "AI + ML Models" if analysis.get("mode") == "ml" else "Rule-Based Analysis"

    symptoms   = _dedup(entities.get("SYMPTOM",    []))
    therapies  = _dedup(entities.get("THERAPY",    []))
    meds       = _dedup(entities.get("MEDICATION", []))
    lifestyle  = _dedup(entities.get("LIFESTYLE",  []))

    bar = _severity_bar(sev_score)
    sev_msg     = _severity_message(sev_lbl, disorder)
    therapy_desc= _therapy_description(therapy)

    # Personalised greeting
    first_name = name.split()[0] if name and name != "Unknown" else "Patient"

    report = f"""
╔══════════════════════════════════════════════════════════════╗
║         PSYCHOLOGICAL CONSULTATION ANALYSIS REPORT          ║
╚══════════════════════════════════════════════════════════════╝

  Dear {first_name},

  Thank you for your consultation. This report summarises the
  findings from your session and outlines a personalised plan
  to support your mental health and wellbeing.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PATIENT DETAILS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Name              :  {name}
  Age               :  {age}
  Gender            :  {gender}
  Consultation Date :  {date}
  Session ID        :  {analysis.get("session_id", "N/A")}
  Analysis Mode     :  {mode}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  IDENTIFIED CONCERN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Primary Condition :  {disorder}

  {first_name}, based on the consultation notes, the primary
  concern identified is {disorder}. This has been determined
  through NLP analysis of your reported symptoms and emotional
  indicators.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SYMPTOMS OBSERVED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{_bullet(symptoms)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  EMOTIONAL STATE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Detected Emotions :  {", ".join(e.title() for e in emotions) or "Neutral"}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SEVERITY ASSESSMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Level  :  {sev_lbl}   ({sev_score:.1f} / 3.0)
  [{bar}]
   Low ◄──────────────────────────────────► High

  {sev_msg}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  RECOMMENDED THERAPY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ★  {therapy}

  {therapy_desc}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  LIFESTYLE RECOMMENDATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{_bullet(lifestyle, "Continue healthy lifestyle habits")}

{"" if not meds else chr(10) + "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" + chr(10) + "  MEDICATIONS PRESCRIBED" + chr(10) + "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" + chr(10) + _bullet(meds)}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CLINICIAN REMARKS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  "{remarks}"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  A MESSAGE FOR YOU
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  {first_name}, seeking support is a sign of strength, not
  weakness. Every step you take towards your mental wellbeing
  matters. You are not alone in this journey.

  Remember: Progress, not perfection. Be kind to yourself.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Generated on {datetime.datetime.now().strftime("%d %B %Y at %H:%M")}
  This report was generated using NLP-based psychological
  analysis. Please consult a licensed professional for
  clinical diagnosis and treatment decisions.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""".strip()

    return report



def detect_risk_indicators(text: str) -> Dict[str, Any]:
    """Detect clinical risk indicators requiring immediate professional review."""
    HIGH_RISK = [
        "suicidal", "suicide", "want to die", "end my life", "kill myself",
        "self-harm", "self harm", "cutting myself", "hurt myself",
        "no reason to live", "better off dead", "can't go on",
        "active plan", "plan to end",
    ]
    MODERATE_RISK = [
        "hopeless", "worthless", "burden to everyone", "nobody cares",
        "passive suicidal", "don't want to be here", "wish i was dead",
        "life is meaningless", "can't take it anymore", "giving up",
        "feel empty", "numb", "trapped",
    ]

    tl = text.lower()
    high_matches    = [kw for kw in HIGH_RISK    if kw in tl]
    moderate_matches= [kw for kw in MODERATE_RISK if kw in tl]

    if high_matches:
        return {
            "level":    "HIGH",
            "flag":     True,
            "keywords": high_matches,
            "message":  "URGENT: High-risk indicators detected. Immediate professional review required. Do not leave patient unattended.",
            "action":   "Contact a licensed mental health professional or emergency services immediately.",
        }
    if moderate_matches:
        return {
            "level":    "MODERATE",
            "flag":     True,
            "keywords": moderate_matches,
            "message":  "Risk indicators detected. Professional review recommended.",
            "action":   "Ensure patient has access to crisis support. Schedule urgent follow-up.",
        }
    return {
        "level":    "NONE",
        "flag":     False,
        "keywords": [],
        "message":  None,
        "action":   None,
    }

# ──────────────────────────────────────────────
# Feature 6: Input validation
# ──────────────────────────────────────────────
CONSULTATION_KEYWORDS = [
    # Clinical terms that appear in real consultation notes
    "patient", "presents", "presenting", "symptoms", "symptom",
    "diagnosis", "diagnosed", "disorder", "condition", "history",
    "therapy", "treatment", "prescribed", "medication", "recommend",
    "counselling", "counseling", "session", "consultation", "assessment",
    "anxiety", "depression", "stress", "insomnia", "panic", "ptsd",
    "feeling", "reports", "observed", "noted", "experiencing", "suffer",
    "mood", "sleep", "emotion", "mental", "psychological", "psychiatric",
    "worry", "fear", "sadness", "anger", "nervous", "restless",
    "overthink", "fatigue", "concentration", "withdrawal", "hopeless",
    "cbt", "dbt", "emdr", "mindfulness", "cognitive", "behavioral",
]

def validate_consultation_text(text: str) -> Dict[str, Any]:
    """
    Check if the input text looks like a psychological consultation note.
    Returns validation result with confidence score and reason.
    """
    if not text or len(text.strip()) < 30:
        return {
            "is_valid": False,
            "confidence": 0,
            "reason": "Text is too short to be a consultation note.",
            "warning": "Please provide a detailed consultation note or prescription.",
        }

    tl = text.lower()
    word_count = len(text.split())

    # Count matching clinical keywords
    matched = [kw for kw in CONSULTATION_KEYWORDS if kw in tl]
    match_count = len(matched)
    confidence = min(round((match_count / 8) * 100), 100)

    # Heuristic checks
    has_clinical_structure = any(x in tl for x in [
        "patient", "symptoms", "diagnosis", "therapy", "prescribed",
        "counsell", "assessment", "consultation", "disorder",
    ])
    has_enough_words = word_count >= 20
    looks_like_list  = ("\n" in text and text.count("\n") > 2) or any(x in tl for x in ["1.", "2.", "3."])
    too_short = word_count < 15

    if too_short:
        return {
            "is_valid": False,
            "confidence": confidence,
            "reason": f"Text too short ({word_count} words). Minimum 15 words expected.",
            "warning": "Please provide more detailed consultation notes.",
        }

    if match_count < 2 and not has_clinical_structure:
        return {
            "is_valid": False,
            "confidence": confidence,
            "reason": "Text does not appear to be a medical/psychological consultation note.",
            "warning": (
                "The input does not contain recognisable clinical terms. "
                "Results may be inaccurate. For best results, provide consultation "
                "notes, prescriptions, or psychological assessment text."
            ),
        }

    if match_count < 4:
        return {
            "is_valid": True,
            "confidence": confidence,
            "reason": "Text has minimal clinical content — results may be less accurate.",
            "warning": (
                "Low clinical keyword density detected. "
                "Consider providing more detailed consultation notes."
            ),
        }

    return {
        "is_valid": True,
        "confidence": confidence,
        "reason": f"Valid consultation text detected ({match_count} clinical terms found).",
        "warning": None,
    }


# ──────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────
def analyze(
    text: str,
    patient_name: str = "Patient",
    session_id: Optional[str] = None,
    pdf_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the full NLP pipeline.
    Falls back to rule-based methods if models are not trained yet.
    """
    try:
        if pdf_path:
            text = extract_text_from_pdf(pdf_path)

        if not text or not text.strip():
            raise ValueError("Input text is empty.")

        # ── Feature 6: Validate input looks like a consultation note ──
        validation = validate_consultation_text(text)
        if not validation["is_valid"]:
            # Still run pipeline but flag the warning
            print(f"[pipeline] Input validation warning: {validation['reason']}")

        # ── Parse structured fields (safe with defaults) ──
        try:
            parsed = parse_consultation_fields(text)
        except Exception as pe:
            print(f"[pipeline] parse_consultation_fields error: {pe}")
            parsed = {
                "name": None, "age": None, "gender": None, "date": None,
                "symptoms": [], "therapy": [], "lifestyle": [],
                "disorder": [], "remarks": None,
            }

        # Ensure all list fields are actually lists
        for key in ("symptoms", "therapy", "lifestyle", "disorder"):
            if not isinstance(parsed.get(key), list):
                parsed[key] = []

        # Override patient_name if found in PDF
        if parsed.get("name") and patient_name in ("Patient", "Anonymous Patient", ""):
            patient_name = parsed["name"]

        clean     = preprocess(text)
        emotions  = detect_emotions(clean)
        emotions  = emotions if isinstance(emotions, list) else ["neutral"]
        sev_score = compute_severity(emotions, clean)
        sev_lbl   = severity_label(sev_score)

        # Use parsed disorder if available, else ML/rule-based with confidence
        disorder_result = classify_disorder_with_confidence(clean)
        if parsed.get("disorder"):
            disorder    = parsed["disorder"][0]
            disorder_confidence = 95.0
            disorder_source     = "parsed"
            disorder_all_probs  = {}
        else:
            disorder    = disorder_result["disorder"]
            disorder_confidence = disorder_result["confidence"]
            disorder_source     = disorder_result["source"]
            disorder_all_probs  = disorder_result.get("all_probs", {})
        disorder = disorder or "Anxiety"

        entities = extract_entities(clean)
        if not isinstance(entities, dict):
            entities = {"SYMPTOM": [], "DISORDER": [], "THERAPY": [],
                        "MEDICATION": [], "LIFESTYLE": []}

        # Safely get list fields — ensure no None values
        def safe_list(val):
            return [v for v in (val or []) if v] 

        parsed_symptoms  = safe_list(parsed.get("symptoms"))
        parsed_therapies = safe_list(parsed.get("therapy"))
        parsed_lifestyle = safe_list(parsed.get("lifestyle"))

        # Merge parsed + NER, deduplicate
        symptoms    = list(dict.fromkeys(parsed_symptoms  + safe_list(entities.get("SYMPTOM"))))
        # Clean therapy list — remove anything that doesn't look like a real therapy
        raw_therapies = list(dict.fromkeys(parsed_therapies + safe_list(entities.get("THERAPY"))))
        therapy_whitelist_kw = [
            "therapy", "cbt", "dbt", "emdr", "counseling", "counselling",
            "cognitive", "behavioral", "behavioural", "mindfulness-based",
            "psychodynamic", "exposure", "sleep", "group", "medication",
            "management", "intervention", "act", "trauma",
        ]
        therapies = [
            t for t in raw_therapies
            if any(kw in t.lower() for kw in therapy_whitelist_kw)
        ]
        medications = safe_list(entities.get("MEDICATION"))
        lifestyle   = list(dict.fromkeys(parsed_lifestyle + safe_list(entities.get("LIFESTYLE"))))

        # Use first parsed therapy if available, else recommend
        if therapies:
            therapy = therapies[0]
        else:
            therapy = recommend_therapy(symptoms, disorder, sev_score)
        therapy = therapy or "Cognitive Behavioral Therapy"

        # Update entities with merged data
        entities["SYMPTOM"]   = symptoms
        entities["THERAPY"]   = therapies
        entities["LIFESTYLE"] = lifestyle
        if parsed.get("disorder"):
            entities["DISORDER"] = parsed["disorder"]

        mind_map = build_mind_map(
            patient_name, symptoms, disorder, therapy, medications, lifestyle
        )
        sid = session_id or f"S-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        # Feature 7: Use consultation date from PDF if available
        session_date = None
        raw_date = parsed.get("date")
        if raw_date:
            try:
                for fmt in ["%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%d %B %Y", "%B %d %Y"]:
                    try:
                        session_date = datetime.datetime.strptime(raw_date.strip(), fmt).isoformat()
                        break
                    except ValueError:
                        continue
            except Exception:
                pass
        if not session_date:
            session_date = datetime.datetime.now().isoformat()

        # Clinical scale mapping
        clinical_scale = map_to_clinical_scales(sev_score, disorder)

        result = {
            "patient_name":        patient_name,
            "patient_age":         parsed.get("age"),
            "patient_gender":      parsed.get("gender"),
            "consultation_date":   parsed.get("date"),
            "session_id":          sid,
            "original_text":       text[:800],
            "parsed_fields":       parsed,
            "emotions":            emotions,
            "severity_score":      sev_score,
            "severity_label":      sev_lbl,
            # ── Feature 2: Clinical scale ──
            "clinical_scale":      clinical_scale,
            "predicted_disorder":  disorder,
            # ── Feature 3: Confidence score ──
            "disorder_confidence": disorder_confidence,
            "disorder_source":     disorder_source,
            "disorder_all_probs":  disorder_all_probs,
            "entities":            entities,
            "recommended_therapy": therapy,
            "remarks":             parsed.get("remarks"),
            "mind_map":            mind_map,
            "report":              "",
            "timestamp":           session_date,
            "mode":                "ml" if os.path.isdir(DISORDER_MODEL_PATH) else "rule-based",
            "input_validation":    validation,
            # Feature 10: Risk flags
            "risk":                detect_risk_indicators(text),
            # Feature 9: Therapy explanation
            "therapy_explanation": explain_therapy(therapy, symptoms, disorder),
        }
        result["report"] = generate_report(result)
        return result

    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Pipeline error: {str(e)}")


if __name__ == "__main__":
    sample = (
        "I feel anxious and overwhelmed before exams. I cannot sleep and keep overthinking. "
        "My therapist mentioned CBT might help. I have been prescribed sertraline."
    )
    out = analyze(sample, patient_name="Test Patient")
    print(out["report"])
    print("\nMode:", out["mode"])
    print("Emotions:", out["emotions"])
    print("Disorder:", out["predicted_disorder"])
    print("Therapy:", out["recommended_therapy"])