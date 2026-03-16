"""
NER Model Training using NCBI Disease Corpus + Psychological Synthetic Data
Fixes:
  - Auto-detects NCBI corpus location (Windows, Linux, project folder)
  - Works even when NCBI files are missing (falls back to synthetic data only)
  - No crash when dev/test set is empty
  - ner_f1 always has a valid float value
"""

import os
import re
import json
import random
import warnings
warnings.filterwarnings("ignore")

import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
BASE        = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE)

MODEL_SAVE  = os.path.join(PROJECT_ROOT, "models", "ner_model")
METRICS_OUT = os.path.join(PROJECT_ROOT, "results", "metrics.json")

ITERATIONS = 40
DROP       = 0.35

os.makedirs(MODEL_SAVE, exist_ok=True)
os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)

# ── NCBI file search: checks multiple possible locations ──
def _find_ncbi_file(filename: str) -> str:
    """Search common locations for NCBI corpus files."""
    candidates = [
        # Same folder as this script
        os.path.join(BASE, filename),
        # Project root
        os.path.join(PROJECT_ROOT, filename),
        # datasets subfolder
        os.path.join(PROJECT_ROOT, "datasets", "ncbi_disease", filename),
        os.path.join(PROJECT_ROOT, "datasets", filename),
        # Linux Claude project mount
        os.path.join("/mnt/project", filename),
        # Windows Downloads
        os.path.join(os.path.expanduser("~"), "Downloads", filename),
        os.path.join(os.path.expanduser("~"), "Downloads", "nlp project", filename),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None

NCBI_TRAIN = _find_ncbi_file("NCBI_corpus_training.txt")
NCBI_DEV   = _find_ncbi_file("NCBI_corpus_development.txt")
NCBI_TEST  = _find_ncbi_file("NCBI_corpus_testing.txt")

# NCBI disease category → psychology label
CATEGORY_MAP = {
    "SpecificDisease":  "DISORDER",
    "DiseaseClass":     "DISORDER",
    "Modifier":         "SYMPTOM",
    "CompositeMention": "DISORDER",
}

# ──────────────────────────────────────────────
# Psychological entity vocabulary
# ──────────────────────────────────────────────
PSYCH_ENTITIES = [
    # Therapies
    ("cognitive behavioral therapy", "THERAPY"),
    ("CBT", "THERAPY"),
    ("dialectical behavior therapy", "THERAPY"),
    ("DBT", "THERAPY"),
    ("mindfulness-based therapy", "THERAPY"),
    ("exposure therapy", "THERAPY"),
    ("psychodynamic therapy", "THERAPY"),
    ("sleep therapy", "THERAPY"),
    ("group therapy", "THERAPY"),
    ("EMDR therapy", "THERAPY"),
    ("EMDR", "THERAPY"),
    # Medications
    ("sertraline", "MEDICATION"),
    ("fluoxetine", "MEDICATION"),
    ("escitalopram", "MEDICATION"),
    ("alprazolam", "MEDICATION"),
    ("lorazepam", "MEDICATION"),
    ("clonazepam", "MEDICATION"),
    ("venlafaxine", "MEDICATION"),
    ("bupropion", "MEDICATION"),
    # Lifestyle
    ("exercise regularly", "LIFESTYLE"),
    ("mindfulness", "LIFESTYLE"),
    ("meditation", "LIFESTYLE"),
    ("sleep hygiene", "LIFESTYLE"),
    ("time management", "LIFESTYLE"),
    ("journaling", "LIFESTYLE"),
    ("yoga", "LIFESTYLE"),
    ("breathing exercises", "LIFESTYLE"),
    ("social support", "LIFESTYLE"),
    # Symptoms
    ("overthinking", "SYMPTOM"),
    ("restlessness", "SYMPTOM"),
    ("insomnia", "SYMPTOM"),
    ("fatigue", "SYMPTOM"),
    ("hopelessness", "SYMPTOM"),
    ("worthlessness", "SYMPTOM"),
    ("panic attacks", "SYMPTOM"),
    ("social withdrawal", "SYMPTOM"),
    ("irritability", "SYMPTOM"),
    ("concentration difficulties", "SYMPTOM"),
    ("low self-esteem", "SYMPTOM"),
    ("nervousness", "SYMPTOM"),
    ("sadness", "SYMPTOM"),
    ("mood swings", "SYMPTOM"),
    ("chest tightness", "SYMPTOM"),
    ("nightmares", "SYMPTOM"),
    ("hypervigilance", "SYMPTOM"),
    ("flashbacks", "SYMPTOM"),
    # Disorders
    ("anxiety disorder", "DISORDER"),
    ("major depressive disorder", "DISORDER"),
    ("generalized anxiety disorder", "DISORDER"),
    ("panic disorder", "DISORDER"),
    ("post-traumatic stress disorder", "DISORDER"),
    ("PTSD", "DISORDER"),
    ("bipolar disorder", "DISORDER"),
    ("OCD", "DISORDER"),
    ("obsessive compulsive disorder", "DISORDER"),
    ("depression", "DISORDER"),
    ("insomnia disorder", "DISORDER"),
]

ENTITY_LOOKUP = {e.lower(): l for e, l in PSYCH_ENTITIES}


# ──────────────────────────────────────────────
# STEP 1 – Parse NCBI corpus
# ──────────────────────────────────────────────
def parse_ncbi_file(filepath: str):
    examples = []
    if not filepath or not os.path.isfile(filepath):
        return examples

    with open(filepath, encoding="utf-8", errors="ignore") as f:
        content = f.read()

    docs = [d.strip() for d in re.split(r"\n\n+", content) if d.strip()]
    pattern = r'<category="([^"]+)">([^<]+)</category>'

    for doc in docs:
        lines = doc.split("\n")
        if not lines:
            continue
        parts = lines[0].split("\t")
        if len(parts) < 2:
            continue

        raw_text = " ".join(parts[1:])
        clean    = re.sub(pattern, r"\2", raw_text).strip()
        if not clean:
            continue

        entities = []
        for m in re.finditer(pattern, raw_text):
            ent_type = m.group(1)
            ent_text = m.group(2)
            label    = CATEGORY_MAP.get(ent_type)
            if not label:
                continue
            pre_clean = re.sub(pattern, r"\2", raw_text[:m.start()])
            start     = len(pre_clean)
            end       = start + len(ent_text)
            if start < len(clean) and end <= len(clean):
                entities.append((start, end, label))

        if clean:
            examples.append((clean, {"entities": entities}))

    print(f"  Parsed {len(examples)} examples from {os.path.basename(filepath)}")
    return examples


# ──────────────────────────────────────────────
# STEP 2 – Synthetic psychological examples
# ──────────────────────────────────────────────
def make_psych_examples(n: int = 1200):
    """Generate training sentences from entity vocabulary."""
    templates = [
        "Patient reports {s1} and {s2}. {d} is suspected. {t} is recommended.",
        "The client experiences {s1}, leading to diagnosis of {d}. {t} and {l} advised.",
        "{s1} and {s2} are primary symptoms. Clinician diagnosed {d}. Treatment: {t}.",
        "Assessment reveals {s1}. Diagnosis: {d}. Prescribed {m} and referred to {t}.",
        "Ongoing {s1} has worsened. Current diagnosis: {d}. Recommending {l} and {t}.",
        "{d} manifests as {s1} and {s2}. Initiating {t}.",
        "Patient presents with {s1}. {m} prescribed. {t} to begin next week.",
        "Session notes: {s1}, {s2} observed. {d} confirmed. {l} encouraged.",
        "{t} recommended for managing {s1} associated with {d}.",
        "Follow-up for {d}: {s1} improving. Continue {t} and {l}.",
    ]

    symptoms   = [e for e, l in PSYCH_ENTITIES if l == "SYMPTOM"]
    disorders  = [e for e, l in PSYCH_ENTITIES if l == "DISORDER"]
    therapies  = [e for e, l in PSYCH_ENTITIES if l == "THERAPY"]
    meds       = [e for e, l in PSYCH_ENTITIES if l == "MEDICATION"]
    lifestyles = [e for e, l in PSYCH_ENTITIES if l == "LIFESTYLE"]

    random.seed(42)
    examples = []

    for _ in range(n):
        tpl = random.choice(templates)
        replacements = {
            "s1": random.choice(symptoms),
            "s2": random.choice(symptoms),
            "d":  random.choice(disorders),
            "t":  random.choice(therapies),
            "m":  random.choice(meds),
            "l":  random.choice(lifestyles),
        }
        text = tpl.format(**replacements)
        entities = []
        for token, ent_text in replacements.items():
            label = ENTITY_LOOKUP.get(ent_text.lower())
            if not label:
                continue
            idx = 0
            while True:
                pos = text.lower().find(ent_text.lower(), idx)
                if pos == -1:
                    break
                entities.append((pos, pos + len(ent_text), label))
                idx = pos + 1

        entities = list(set(entities))
        examples.append((text, {"entities": entities}))

    return examples


# ──────────────────────────────────────────────
# STEP 3 – Filter overlapping spans
# ──────────────────────────────────────────────
def filter_overlapping(entities):
    entities = sorted(entities, key=lambda x: (x[0], -(x[1] - x[0])))
    result, last_end = [], -1
    for start, end, label in entities:
        if start >= last_end and end > start:
            result.append((start, end, label))
            last_end = end
    return result


# ──────────────────────────────────────────────
# STEP 4 – Convert to spaCy Examples
# ──────────────────────────────────────────────
def to_examples(data, nlp):
    examples = []
    for text, annot in data:
        annot = {"entities": filter_overlapping(annot.get("entities", []))}
        doc   = nlp.make_doc(text)
        try:
            ex = Example.from_dict(doc, annot)
            examples.append(ex)
        except Exception:
            pass
    return examples


# ──────────────────────────────────────────────
# STEP 5 – Train
# ──────────────────────────────────────────────
def train():
    print("=== NER Model Training ===")

    # Load NCBI data
    ncbi_train = parse_ncbi_file(NCBI_TRAIN)
    ncbi_dev   = parse_ncbi_file(NCBI_DEV)
    ncbi_test  = parse_ncbi_file(NCBI_TEST)

    if not ncbi_train:
        print("  NCBI corpus not found — using synthetic data only.")
        print("  To include NCBI data, place these files in the project folder:")
        print("    NCBI_corpus_training.txt")
        print("    NCBI_corpus_development.txt")
        print("    NCBI_corpus_testing.txt")

    # Synthetic psychological data
    psych_data = make_psych_examples(1200)

    # Combine and split
    all_train = ncbi_train + psych_data
    random.seed(42)
    random.shuffle(all_train)

    # If no dev/test from NCBI, carve out 15% each from all_train
    if not ncbi_dev and not ncbi_test:
        n     = len(all_train)
        n_dev = max(20, int(n * 0.15))
        dev_data  = all_train[:n_dev]
        test_data = all_train[n_dev:n_dev * 2]
        train_data = all_train[n_dev * 2:]
        print(f"  Auto-split → Train: {len(train_data)} | Dev: {len(dev_data)} | Test: {len(test_data)}")
    else:
        train_data = all_train
        dev_data   = ncbi_dev  or []
        test_data  = ncbi_test or []
        print(f"  Train: {len(train_data)} | Dev: {len(dev_data)} | Test: {len(test_data)}")

    # Build spaCy model
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner", last=True)
    for label in ["SYMPTOM", "DISORDER", "THERAPY", "MEDICATION", "LIFESTYLE"]:
        ner.add_label(label)

    # Initialize
    train_ex = to_examples(train_data, nlp)
    dev_ex   = to_examples(dev_data,   nlp)

    if not train_ex:
        raise RuntimeError("No valid training examples generated.")

    optimizer = nlp.initialize(lambda: train_ex[:100])

    best_f1    = 0.0
    best_loss  = float("inf")

    for itn in range(ITERATIONS):
        random.shuffle(train_data)
        losses = {}
        batches = minibatch(
            to_examples(train_data, nlp),
            size=compounding(4.0, 32.0, 1.001)
        )
        for batch in batches:
            nlp.update(batch, drop=DROP, losses=losses)

        ner_loss = losses.get("ner", 0.0)

        # Evaluate on dev set every 5 iterations
        if (itn + 1) % 5 == 0:
            if dev_ex:
                scores  = nlp.evaluate(dev_ex)
                ner_f1  = float(scores.get("ents_f", 0.0) or 0.0)
            else:
                ner_f1 = 0.0

            print(f"  Iter {itn+1:02d}/{ITERATIONS} | NER Loss: {ner_loss:.4f} | Dev F1: {ner_f1:.4f}")

            # Save best model (by F1 if dev set exists, else by loss)
            improved = (dev_ex and ner_f1 > best_f1) or (not dev_ex and ner_loss < best_loss)
            if improved:
                best_f1   = ner_f1
                best_loss = ner_loss
                nlp.to_disk(MODEL_SAVE)
                indicator = f"F1={ner_f1:.4f}" if dev_ex else f"loss={ner_loss:.4f}"
                print(f"  ✓ Model saved ({indicator})")

    # Always save at end if nothing was saved yet
    if best_f1 == 0.0 and best_loss == float("inf"):
        nlp.to_disk(MODEL_SAVE)
        print(f"  ✓ Model saved (final)")

    # Final test evaluation
    nlp2     = spacy.load(MODEL_SAVE)
    test_ex  = to_examples(test_data, nlp2) if test_data else []

    if test_ex:
        scores = nlp2.evaluate(test_ex)
        p = float(scores.get("ents_p", 0.0) or 0.0)
        r = float(scores.get("ents_r", 0.0) or 0.0)
        f = float(scores.get("ents_f", 0.0) or 0.0)
        per_ent = scores.get("ents_per_type", {})
    else:
        print("  No test set — skipping final evaluation.")
        p, r, f = 0.0, 0.0, 0.0
        per_ent = {}

    print("\n=== Test Evaluation ===")
    print(f"  Precision : {p:.4f}")
    print(f"  Recall    : {r:.4f}")
    print(f"  F1        : {f:.4f}")
    if per_ent:
        for ent_type, sc in per_ent.items():
            print(f"  {ent_type:12s} → P:{sc.get('p',0):.2f}  R:{sc.get('r',0):.2f}  F:{sc.get('f',0):.2f}")

    metrics = {
        "model":      "ner_model",
        "precision":  p,
        "recall":     r,
        "f1":         f,
        "per_entity": {k: dict(v) for k, v in per_ent.items()},
    }

    existing = {}
    if os.path.exists(METRICS_OUT):
        with open(METRICS_OUT) as fp:
            try:
                existing = json.load(fp)
            except Exception:
                existing = {}
    existing["ner_model"] = metrics
    with open(METRICS_OUT, "w") as fp:
        json.dump(existing, fp, indent=2)
    print(f"\nMetrics saved to {METRICS_OUT}")
    print(f"Model saved to   {MODEL_SAVE}")


if __name__ == "__main__":
    train()