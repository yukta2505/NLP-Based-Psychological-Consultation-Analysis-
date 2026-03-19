"""
Improved NER Model Training
Uses:
  - NCBI Disease Corpus (optional)
  - High-quality psychological synthetic data
  - Gold clinical cases for better mental-health entity learning

Fixes / Improvements:
  - Better psychological entity coverage
  - Medication + dosage examples
  - Full phrase entities like "loss of concentration"
  - Therapy/lifestyle detection improved
  - More realistic clinical templates
  - Safer overlap filtering
  - Dev/test split works even without NCBI
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
BASE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE)

MODEL_SAVE = os.path.join(PROJECT_ROOT, "models", "ner_model")
METRICS_OUT = os.path.join(PROJECT_ROOT, "results", "metrics.json")

ITERATIONS = 45
DROP = 0.22
SEED = 42

os.makedirs(MODEL_SAVE, exist_ok=True)
os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)

random.seed(SEED)


# ──────────────────────────────────────────────
# FILE SEARCH
# ──────────────────────────────────────────────
def _find_ncbi_file(filename: str) -> str | None:
    candidates = [
        os.path.join(BASE, filename),
        os.path.join(PROJECT_ROOT, filename),
        os.path.join(PROJECT_ROOT, "datasets", "ncbi_disease", filename),
        os.path.join(PROJECT_ROOT, "datasets", filename),
        os.path.join("/mnt/project", filename),
        os.path.join(os.path.expanduser("~"), "Downloads", filename),
        os.path.join(os.path.expanduser("~"), "Downloads", "nlp project", filename),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


NCBI_TRAIN = _find_ncbi_file("NCBI_corpus_training.txt")
NCBI_DEV = _find_ncbi_file("NCBI_corpus_development.txt")
NCBI_TEST = _find_ncbi_file("NCBI_corpus_testing.txt")


# ──────────────────────────────────────────────
# NCBI CATEGORY MAP
# ──────────────────────────────────────────────
CATEGORY_MAP = {
    "SpecificDisease": "DISORDER",
    "DiseaseClass": "DISORDER",
    "CompositeMention": "DISORDER",
    # "Modifier": "SYMPTOM",   # too noisy; avoid forcing this
}


# ──────────────────────────────────────────────
# ENTITY VOCAB
# ──────────────────────────────────────────────
THERAPIES = [
    "cognitive behavioral therapy",
    "CBT",
    "dialectical behavior therapy",
    "DBT",
    "mindfulness-based therapy",
    "exposure therapy",
    "psychodynamic therapy",
    "group therapy",
    "EMDR therapy",
    "EMDR",
    "counseling",
    "individual counseling",
    "supportive counseling",
    "sleep restriction therapy",
    "stimulus control techniques",
    "sleep hygiene education",
    "interoceptive conditioning",
    "cognitive restructuring",
    "breathing exercises",
    "parental guidance session",
]

MEDICATIONS = [
    "sertraline",
    "fluoxetine",
    "escitalopram",
    "alprazolam",
    "lorazepam",
    "clonazepam",
    "venlafaxine",
    "bupropion",
    "melatonin",
]

LIFESTYLES = [
    "mindfulness",
    "mindfulness exercises",
    "meditation",
    "sleep hygiene",
    "time management",
    "time management strategies",
    "journaling",
    "yoga",
    "breathing exercises",
    "social support",
    "regular exercise",
    "regular physical activity",
    "physical activity",
    "exercise",
]

SYMPTOMS = [
    "persistent low mood",
    "low mood",
    "hopelessness",
    "worthlessness",
    "loss of interest",
    "fatigue",
    "poor concentration",
    "loss of concentration",
    "difficulty concentrating",
    "difficulty sleeping",
    "early morning awakening",
    "decreased appetite",
    "weight loss",
    "overthinking",
    "restlessness",
    "nervousness",
    "muscle tension",
    "headaches",
    "racing thoughts",
    "social withdrawal",
    "burnout symptoms",
    "feeling overwhelmed",
    "irritability",
    "neck tension",
    "gastrointestinal discomfort",
    "difficulty initiating sleep",
    "difficulty maintaining sleep",
    "daytime fatigue",
    "panic attacks",
    "chest tightness",
    "heart palpitations",
    "dizziness",
    "intense fear of dying",
    "anticipatory anxiety",
    "crying spells",
    "intrusive negative thoughts",
    "shortness of breath",
    "flashbacks",
    "nightmares",
    "hypervigilance",
    "severe startle response",
    "emotional numbing",
    "detachment",
    "school refusal",
    "social anxiety",
    "fear of embarrassment",
    "negative evaluation",
    "nausea",
    "trembling",
    "mild stress",
    "loss of motivation",
    "inability to perform basic self-care activities",
    "passive suicidal ideation",
    "psychomotor retardation",
    "sleep fragmented",
    "sleep severely disrupted",
]

DISORDERS = [
    "depression",
    "major depressive disorder",
    "anxiety",
    "significant anxiety",
    "anxiety disorder",
    "generalized anxiety disorder",
    "stress",
    "insomnia",
    "insomnia disorder",
    "panic disorder",
    "post-traumatic stress disorder",
    "PTSD",
    "bipolar disorder",
    "OCD",
    "obsessive compulsive disorder",
]

DOSAGES = [
    "10mg",
    "20mg",
    "25mg",
    "50mg",
    "75mg",
    "0.5mg",
]


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def find_all_spans(text: str, phrase: str):
    spans = []
    start = 0
    text_low = text.lower()
    phrase_low = phrase.lower()

    while True:
        idx = text_low.find(phrase_low, start)
        if idx == -1:
            break
        end = idx + len(phrase)
        left_ok = idx == 0 or not text[idx - 1].isalnum()
        right_ok = end == len(text) or not text[end:end + 1].isalnum()
        if left_ok and right_ok:
            spans.append((idx, end))
        start = idx + 1
    return spans


def add_entity_spans(text, entities, phrase, label):
    for s, e in find_all_spans(text, phrase):
        entities.append((s, e, label))


def build_med_with_dose_examples():
    phrases = []
    for med in MEDICATIONS:
        phrases.append(med)
        for dose in DOSAGES:
            phrases.append(f"{med} {dose}")
            phrases.append(f"{med} {dose} daily")
    return phrases


MEDICATION_PHRASES = build_med_with_dose_examples()


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
        clean = re.sub(pattern, r"\2", raw_text).strip()
        if not clean:
            continue

        entities = []
        for m in re.finditer(pattern, raw_text):
            ent_type = m.group(1)
            ent_text = m.group(2)
            label = CATEGORY_MAP.get(ent_type)
            if not label:
                continue

            pre_clean = re.sub(pattern, r"\2", raw_text[:m.start()])
            start = len(pre_clean)
            end = start + len(ent_text)

            if 0 <= start < end <= len(clean):
                entities.append((start, end, label))

        examples.append((clean, {"entities": entities}))

    print(f"  Parsed {len(examples)} examples from {os.path.basename(filepath)}")
    return examples


# ──────────────────────────────────────────────
# STEP 2 – Gold clinical cases
# ──────────────────────────────────────────────
def make_gold_psych_examples():
    raw_cases = [
        {
            "text": """Patient is a 28-year-old female presenting with persistent low mood for the past 3 months. Reports feeling hopeless and worthless most of the day. Loss of interest in previously enjoyed activities including music and socializing. Experiencing significant fatigue, poor concentration, and recurrent thoughts of being a burden. Sleep is disrupted with early morning awakening. Appetite decreased, weight loss of 4kg noted. Prescribed fluoxetine 20mg. CBT sessions recommended twice weekly.""",
            "entities": [
                ("persistent low mood", "SYMPTOM"),
                ("hopeless", "SYMPTOM"),
                ("worthless", "SYMPTOM"),
                ("loss of interest", "SYMPTOM"),
                ("fatigue", "SYMPTOM"),
                ("poor concentration", "SYMPTOM"),
                ("early morning awakening", "SYMPTOM"),
                ("weight loss", "SYMPTOM"),
                ("fluoxetine 20mg", "MEDICATION"),
                ("CBT", "THERAPY"),
            ],
        },
        {
            "text": """Patient is a 22-year-old male student experiencing excessive worry and nervousness particularly before examinations. Reports persistent overthinking, restlessness, and difficulty concentrating in class. Muscle tension and headaches noted. Sleep onset delayed due to racing thoughts. Patient avoids group discussions fearing judgment. Prescribed escitalopram 10mg. Cognitive behavioral therapy initiated. Mindfulness and breathing exercises advised.""",
            "entities": [
                ("nervousness", "SYMPTOM"),
                ("overthinking", "SYMPTOM"),
                ("restlessness", "SYMPTOM"),
                ("difficulty concentrating", "SYMPTOM"),
                ("muscle tension", "SYMPTOM"),
                ("headaches", "SYMPTOM"),
                ("racing thoughts", "SYMPTOM"),
                ("escitalopram 10mg", "MEDICATION"),
                ("Cognitive behavioral therapy", "THERAPY"),
                ("Mindfulness", "LIFESTYLE"),
                ("breathing exercises", "LIFESTYLE"),
            ],
        },
        {
            "text": """35-year-old male professional presenting with burnout symptoms following 18 months of excessive workload. Reports feeling overwhelmed, irritable, and unable to switch off after work hours. Physical symptoms include frequent headaches, neck tension, and gastrointestinal discomfort. Difficulty delegating tasks and persistent sense of pressure. No formal psychiatric diagnosis. Counseling initiated. Time management strategies, regular exercise, and journaling recommended.""",
            "entities": [
                ("burnout symptoms", "SYMPTOM"),
                ("feeling overwhelmed", "SYMPTOM"),
                ("irritable", "SYMPTOM"),
                ("headaches", "SYMPTOM"),
                ("neck tension", "SYMPTOM"),
                ("gastrointestinal discomfort", "SYMPTOM"),
                ("Counseling", "THERAPY"),
                ("Time management strategies", "LIFESTYLE"),
                ("regular exercise", "LIFESTYLE"),
                ("journaling", "LIFESTYLE"),
            ],
        },
        {
            "text": """Patient is a 45-year-old female with a 6-month history of difficulty initiating and maintaining sleep. Reports lying awake for 2 to 3 hours before sleep onset. Waking up 3 to 4 times per night. Feels completely unrested in the morning despite spending 8 hours in bed. Daytime fatigue is severely impacting work performance. Has tried melatonin without improvement. Sleep hygiene education provided. Sleep restriction therapy and stimulus control techniques initiated.""",
            "entities": [
                ("difficulty initiating and maintaining sleep", "SYMPTOM"),
                ("Daytime fatigue", "SYMPTOM"),
                ("melatonin", "MEDICATION"),
                ("Sleep hygiene education", "THERAPY"),
                ("Sleep restriction therapy", "THERAPY"),
                ("stimulus control techniques", "THERAPY"),
            ],
        },
        {
            "text": """Patient is a 31-year-old female with recurrent unexpected panic attacks occurring 3 to 4 times per week. During episodes experiences chest tightness, heart palpitations, dizziness, and intense fear of dying. Episodes last approximately 10 minutes. Patient now avoids public transport, shopping malls, and crowded places. Persistent anticipatory anxiety about future attacks. Prescribed alprazolam 0.5mg for acute episodes. Exposure therapy and interoceptive conditioning recommended.""",
            "entities": [
                ("panic attacks", "SYMPTOM"),
                ("chest tightness", "SYMPTOM"),
                ("heart palpitations", "SYMPTOM"),
                ("dizziness", "SYMPTOM"),
                ("intense fear of dying", "SYMPTOM"),
                ("anticipatory anxiety", "SYMPTOM"),
                ("alprazolam 0.5mg", "MEDICATION"),
                ("Exposure therapy", "THERAPY"),
                ("interoceptive conditioning", "THERAPY"),
            ],
        },
        {
            "text": """Patient presents with a 4-month history of low mood combined with excessive anxiety. Reports crying spells, social withdrawal, and persistent worry about health and finances. Experiences intrusive negative thoughts and physical symptoms including chest tightness and shortness of breath. Concentration severely impaired. Sleep fragmented. Prescribed sertraline 50mg daily. DBT group therapy recommended alongside individual counseling.""",
            "entities": [
                ("low mood", "SYMPTOM"),
                ("excessive anxiety", "DISORDER"),
                ("crying spells", "SYMPTOM"),
                ("social withdrawal", "SYMPTOM"),
                ("intrusive negative thoughts", "SYMPTOM"),
                ("chest tightness", "SYMPTOM"),
                ("shortness of breath", "SYMPTOM"),
                ("Sleep fragmented", "SYMPTOM"),
                ("sertraline 50mg daily", "MEDICATION"),
                ("DBT", "THERAPY"),
                ("group therapy", "THERAPY"),
                ("individual counseling", "THERAPY"),
            ],
        },
        {
            "text": """26-year-old male veteran presenting with intrusive flashbacks and nightmares related to combat experiences. Reports hypervigilance in public spaces and severe startle response. Avoidance of news and loud noises. Emotional numbing and detachment from family observed. Sleep severely disrupted. Difficulty trusting others. Diagnosed with post-traumatic stress disorder. EMDR therapy initiated. Venlafaxine prescribed. Social support and mindfulness techniques encouraged.""",
            "entities": [
                ("flashbacks", "SYMPTOM"),
                ("nightmares", "SYMPTOM"),
                ("hypervigilance", "SYMPTOM"),
                ("severe startle response", "SYMPTOM"),
                ("Emotional numbing", "SYMPTOM"),
                ("detachment", "SYMPTOM"),
                ("Sleep severely disrupted", "SYMPTOM"),
                ("post-traumatic stress disorder", "DISORDER"),
                ("EMDR therapy", "THERAPY"),
                ("Venlafaxine", "MEDICATION"),
                ("Social support", "LIFESTYLE"),
                ("mindfulness techniques", "LIFESTYLE"),
            ],
        },
        {
            "text": """16-year-old female student referred by school counselor for persistent school refusal and social anxiety. Patient reports intense fear of embarrassment and negative evaluation by peers. Avoids oral presentations and group activities. Physical symptoms include nausea and trembling before school. Academic performance declining. Parents report significant behavioral changes over the past 5 months. CBT with focus on cognitive restructuring recommended. Parental guidance session scheduled.""",
            "entities": [
                ("school refusal", "SYMPTOM"),
                ("social anxiety", "SYMPTOM"),
                ("fear of embarrassment", "SYMPTOM"),
                ("negative evaluation", "SYMPTOM"),
                ("nausea", "SYMPTOM"),
                ("trembling", "SYMPTOM"),
                ("CBT", "THERAPY"),
                ("cognitive restructuring", "THERAPY"),
                ("Parental guidance session", "THERAPY"),
            ],
        },
        {
            "text": """Patient is a 40-year-old male presenting for a routine psychological wellness check. Reports generally stable mood with occasional periods of mild stress related to work responsibilities. Sleep is adequate at 7 hours per night. Appetite normal. Maintains regular exercise routine and social connections. No significant psychiatric symptoms identified. Encouraged to continue current lifestyle practices. Follow-up in 6 months.""",
            "entities": [
                ("mild stress", "DISORDER"),
                ("regular exercise", "LIFESTYLE"),
            ],
        },
        {
            "text": """Patient is a 34-year-old female with a 6-month history of severe depressive episode. Reports complete loss of motivation and inability to perform basic self-care activities. Persistent feelings of worthlessness and hopelessness. Passive suicidal ideation reported without active plan. Significant weight loss and psychomotor retardation observed. Previously trialed fluoxetine with poor response. Switched to venlafaxine 75mg. Urgent referral to psychiatry made. Crisis safety plan established.""",
            "entities": [
                ("severe depressive episode", "DISORDER"),
                ("loss of motivation", "SYMPTOM"),
                ("inability to perform basic self-care activities", "SYMPTOM"),
                ("worthlessness", "SYMPTOM"),
                ("hopelessness", "SYMPTOM"),
                ("Passive suicidal ideation", "SYMPTOM"),
                ("weight loss", "SYMPTOM"),
                ("psychomotor retardation", "SYMPTOM"),
                ("fluoxetine", "MEDICATION"),
                ("venlafaxine 75mg", "MEDICATION"),
            ],
        },
    ]

    results = []
    for case in raw_cases:
        text = " ".join(case["text"].split())
        entities = []
        for phrase, label in case["entities"]:
            phrase_clean = " ".join(phrase.split())
            for s, e in find_all_spans(text, phrase_clean):
                entities.append((s, e, label))
        results.append((text, {"entities": filter_overlapping(entities)}))
    return results


# ──────────────────────────────────────────────
# STEP 3 – Synthetic psychological examples
# ──────────────────────────────────────────────
def make_psych_examples(n: int = 1800):
    templates = [
        "Patient reports {s1} and {s2}. {d} is suspected. {t} is recommended.",
        "The client experiences {s1}, {s2}, and {s3}. Diagnosis of {d} was made. {t} and {l} advised.",
        "{s1} and {s2} are primary symptoms. Clinician diagnosed {d}. Treatment includes {t}.",
        "Assessment reveals {s1}. Diagnosis: {d}. Prescribed {m}.",
        "Ongoing {s1} has worsened. Current diagnosis: {d}. Recommending {l} and {t}.",
        "{d} manifests as {s1} and {s2}. Initiating {t}.",
        "Patient presents with {s1}. {m} prescribed. {t} to begin next week.",
        "Session notes: {s1}, {s2} observed. {d} confirmed. {l} encouraged.",
        "{t} recommended for managing {s1} associated with {d}.",
        "Follow-up for {d}: {s1} improving. Continue {t} and {l}.",
        "Patient has {s1}, {s2}, and {s3}. Started on {m}.",
        "Clinician noted {s1}. {l} was recommended along with {t}.",
        "The patient describes {s1} with episodes of {s2}. {m} was prescribed for {d}.",
        "Persistent {s1} and {s2} are affecting daily functioning. {t} initiated and {l} encouraged.",
        "{m} was started after evaluation for {d}. Main complaints include {s1}.",
    ]

    examples = []

    for _ in range(n):
        tpl = random.choice(templates)
        replacements = {
            "s1": random.choice(SYMPTOMS),
            "s2": random.choice(SYMPTOMS),
            "s3": random.choice(SYMPTOMS),
            "d": random.choice(DISORDERS),
            "t": random.choice(THERAPIES),
            "m": random.choice(MEDICATION_PHRASES),
            "l": random.choice(LIFESTYLES),
        }

        text = tpl.format(**replacements)
        entities = []

        add_entity_spans(text, entities, replacements["s1"], "SYMPTOM")
        add_entity_spans(text, entities, replacements["s2"], "SYMPTOM")
        add_entity_spans(text, entities, replacements["s3"], "SYMPTOM")
        add_entity_spans(text, entities, replacements["d"], "DISORDER")
        add_entity_spans(text, entities, replacements["t"], "THERAPY")
        add_entity_spans(text, entities, replacements["m"], "MEDICATION")
        add_entity_spans(text, entities, replacements["l"], "LIFESTYLE")

        examples.append((text, {"entities": filter_overlapping(entities)}))

    return examples


# ──────────────────────────────────────────────
# STEP 4 – Overlap filter
# ──────────────────────────────────────────────
def filter_overlapping(entities):
    entities = sorted(entities, key=lambda x: (x[0], -(x[1] - x[0])))
    result = []

    for start, end, label in entities:
        if start >= end:
            continue

        overlaps = False
        for rs, re, _ in result:
            if not (end <= rs or start >= re):
                overlaps = True
                break

        if not overlaps:
            result.append((start, end, label))

    return sorted(result, key=lambda x: x[0])


# ──────────────────────────────────────────────
# STEP 5 – Convert to spaCy examples
# ──────────────────────────────────────────────
def to_examples(data, nlp):
    examples = []
    for text, annot in data:
        ents = filter_overlapping(annot.get("entities", []))
        doc = nlp.make_doc(text)
        try:
            ex = Example.from_dict(doc, {"entities": ents})
            examples.append(ex)
        except Exception:
            continue
    return examples


# ──────────────────────────────────────────────
# STEP 6 – Train
# ──────────────────────────────────────────────
def train():
    print("=== Improved NER Model Training ===")

    ncbi_train = parse_ncbi_file(NCBI_TRAIN)
    ncbi_dev = parse_ncbi_file(NCBI_DEV)
    ncbi_test = parse_ncbi_file(NCBI_TEST)

    if not ncbi_train:
        print("  NCBI corpus not found — using psych data only.")

    gold_psych = make_gold_psych_examples()
    synthetic_psych = make_psych_examples(1800)

    all_train = ncbi_train + gold_psych + synthetic_psych
    random.shuffle(all_train)

    if not ncbi_dev and not ncbi_test:
        n = len(all_train)
        n_dev = max(25, int(n * 0.12))
        n_test = max(25, int(n * 0.12))

        dev_data = all_train[:n_dev]
        test_data = all_train[n_dev:n_dev + n_test]
        train_data = all_train[n_dev + n_test:]
        print(f"  Auto-split → Train: {len(train_data)} | Dev: {len(dev_data)} | Test: {len(test_data)}")
    else:
        train_data = all_train
        dev_data = ncbi_dev or gold_psych[:3]
        test_data = ncbi_test or gold_psych[3:]
        print(f"  Train: {len(train_data)} | Dev: {len(dev_data)} | Test: {len(test_data)}")

    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner", last=True)

    for label in ["SYMPTOM", "DISORDER", "THERAPY", "MEDICATION", "LIFESTYLE"]:
        ner.add_label(label)

    train_ex = to_examples(train_data, nlp)
    dev_ex = to_examples(dev_data, nlp)

    if not train_ex:
        raise RuntimeError("No valid training examples generated.")

    optimizer = nlp.initialize(lambda: train_ex[: min(200, len(train_ex))])

    best_f1 = -1.0
    best_loss = float("inf")
    saved_any = False

    for itn in range(ITERATIONS):
        random.shuffle(train_ex)
        losses = {}

        batches = minibatch(train_ex, size=compounding(8.0, 32.0, 1.001))
        for batch in batches:
            nlp.update(batch, drop=DROP, losses=losses)

        ner_loss = float(losses.get("ner", 0.0))

        if (itn + 1) % 5 == 0:
            if dev_ex:
                scores = nlp.evaluate(dev_ex)
                ner_f1 = float(scores.get("ents_f", 0.0) or 0.0)
            else:
                ner_f1 = 0.0

            print(f"  Iter {itn + 1:02d}/{ITERATIONS} | Loss: {ner_loss:.4f} | Dev F1: {ner_f1:.4f}")

            improved = ner_f1 > best_f1 if dev_ex else ner_loss < best_loss
            if improved:
                best_f1 = ner_f1
                best_loss = ner_loss
                nlp.to_disk(MODEL_SAVE)
                saved_any = True
                print("  ✓ Saved best model")

    if not saved_any:
        nlp.to_disk(MODEL_SAVE)
        print("  ✓ Saved final model")

    nlp2 = spacy.load(MODEL_SAVE)
    test_ex = to_examples(test_data, nlp2) if test_data else []

    if test_ex:
        scores = nlp2.evaluate(test_ex)
        p = float(scores.get("ents_p", 0.0) or 0.0)
        r = float(scores.get("ents_r", 0.0) or 0.0)
        f = float(scores.get("ents_f", 0.0) or 0.0)
        per_ent = scores.get("ents_per_type", {})
    else:
        p, r, f = 0.0, 0.0, 0.0
        per_ent = {}

    print("\n=== Final Test Evaluation ===")
    print(f"  Precision : {p:.4f}")
    print(f"  Recall    : {r:.4f}")
    print(f"  F1        : {f:.4f}")
    if per_ent:
        for ent_type, sc in per_ent.items():
            print(f"  {ent_type:12s} → P:{sc.get('p', 0):.2f} R:{sc.get('r', 0):.2f} F:{sc.get('f', 0):.2f}")

    metrics = {
        "model": "ner_model",
        "precision": p,
        "recall": r,
        "f1": f,
        "per_entity": {k: dict(v) for k, v in per_ent.items()},
    }

    existing = {}
    if os.path.exists(METRICS_OUT):
        with open(METRICS_OUT, "r", encoding="utf-8") as fp:
            try:
                existing = json.load(fp)
            except Exception:
                existing = {}

    existing["ner_model"] = metrics

    with open(METRICS_OUT, "w", encoding="utf-8") as fp:
        json.dump(existing, fp, indent=2)

    print(f"\nMetrics saved to {METRICS_OUT}")
    print(f"Model saved to   {MODEL_SAVE}")


if __name__ == "__main__":
    train()