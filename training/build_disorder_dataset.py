"""
build_disorder_dataset.py
─────────────────────────────────────────────────────────────────
Downloads REAL mental health datasets from HuggingFace and builds
a balanced disorder classification training set.

Replaces synthetic phrases with real patient-written text.

Sources:
  Depression  → Reddit Depression dataset (already have locally)
  Anxiety     → Real anxiety posts from HuggingFace
  Stress      → Dreaddit stress dataset (Turcan & McKeown, 2019)
  Insomnia    → Sleep/insomnia Reddit posts
  Panic       → Subset of anxiety dataset with panic keywords

Usage:
    python training/build_disorder_dataset.py

Output:
    datasets/mental_health/disorder_dataset_real.csv
"""

import os
import re
import json
import random
import pandas as pd

BASE         = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE)
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "datasets", "mental_health")
OUTPUT_CSV   = os.path.join(OUTPUT_DIR, "disorder_dataset_real.csv")
META_FILE    = os.path.join(OUTPUT_DIR, "disorder_dataset_metadata.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DISORDER_LABELS = ["Depression", "Anxiety", "Stress", "Insomnia", "Panic Disorder"]
TARGET_PER_CLASS = 1500   # balanced target per class
random.seed(42)

print("=" * 60)
print("  Building Real Disorder Classification Dataset")
print("=" * 60)

records = []
source_counts = {}


# ─────────────────────────────────────────────────────────────────
# 1. DEPRESSION — Reddit Depression Dataset (local)
# ─────────────────────────────────────────────────────────────────
def load_depression():
    candidates = [
        os.path.join(OUTPUT_DIR, "depression_dataset_reddit_cleaned.csv"),
        os.path.join(PROJECT_ROOT, "depression_dataset_reddit_cleaned.csv"),
        "/mnt/project/depression_dataset_reddit_cleaned.csv",
    ]
    for path in candidates:
        if os.path.exists(path):
            df  = pd.read_csv(path)
            dep = df[df["is_depression"] == 1]["clean_text"].dropna().tolist()
            dep = [t for t in dep if len(str(t).split()) >= 10]
            dep = random.sample(dep, min(TARGET_PER_CLASS, len(dep)))
            print(f"\n[Depression] Loaded {len(dep)} real Reddit posts from {path}")
            return dep
    print("\n[Depression] Local file not found — trying HuggingFace...")
    return load_hf_depression()


def load_hf_depression():
    try:
        from datasets import load_dataset
        ds = load_dataset("mrm8488/depression-detect", split="train")
        texts = [r["text"] for r in ds
                 if r.get("label") == 1 and len(str(r.get("text","")).split()) >= 10]
        texts = random.sample(texts, min(TARGET_PER_CLASS, len(texts)))
        print(f"  Downloaded {len(texts)} depression posts from HuggingFace")
        return texts
    except Exception as e:
        print(f"  HuggingFace depression failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────────
# 2. ANXIETY — HuggingFace mental health dataset
# ─────────────────────────────────────────────────────────────────
def load_anxiety():
    print("\n[Anxiety] Downloading from HuggingFace...")
    try:
        from datasets import load_dataset

        # Primary: mental health Reddit dataset
        ds = load_dataset("vibhorag101/reddit-mental-health-dataset",
                          split="train", trust_remote_code=True)
        anxiety_texts = []
        for row in ds:
            label = str(row.get("label", "")).lower()
            text  = str(row.get("post", "") or row.get("text", ""))
            if "anxi" in label and len(text.split()) >= 10:
                anxiety_texts.append(text)

        if len(anxiety_texts) >= 200:
            anxiety_texts = random.sample(anxiety_texts,
                                          min(TARGET_PER_CLASS, len(anxiety_texts)))
            print(f"  Downloaded {len(anxiety_texts)} anxiety posts")
            return anxiety_texts
    except Exception as e:
        print(f"  Primary source failed: {e}")

    try:
        from datasets import load_dataset
        ds = load_dataset("Amod/mental_health_counseling_conversations",
                          split="train")
        anxiety_kw = ["anxious", "anxiety", "worry", "nervous", "panic",
                      "fear", "overthink", "restless", "dread"]
        texts = []
        for row in ds:
            ctx = str(row.get("Context", ""))
            if any(kw in ctx.lower() for kw in anxiety_kw) and len(ctx.split()) >= 10:
                texts.append(ctx)
        texts = random.sample(texts, min(TARGET_PER_CLASS, len(texts)))
        print(f"  Extracted {len(texts)} anxiety contexts from counseling data")
        return texts
    except Exception as e:
        print(f"  Secondary source failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────────
# 3. STRESS — Dreaddit Dataset (Turcan & McKeown, ACL 2019)
#    Citation: Turcan, R., & McKeown, K. (2019). Dreaddit: A Reddit
#    Dataset for Stress Analysis in Social Media. EMNLP 2019.
# ─────────────────────────────────────────────────────────────────
def load_stress():
    print("\n[Stress] Downloading Dreaddit dataset...")
    try:
        from datasets import load_dataset
        ds = load_dataset("dreaddit", split="train", trust_remote_code=True)
        # label 1 = stressed
        texts = [str(r.get("text","")) for r in ds
                 if r.get("label") == 1 and len(str(r.get("text","")).split()) >= 10]
        texts = random.sample(texts, min(TARGET_PER_CLASS, len(texts)))
        print(f"  Downloaded {len(texts)} stress posts from Dreaddit")
        return texts
    except Exception as e:
        print(f"  Dreaddit failed: {e}")

    # Fallback: stress from mental health counseling conversations
    try:
        from datasets import load_dataset
        ds = load_dataset("Amod/mental_health_counseling_conversations",
                          split="train")
        stress_kw = ["stress", "overwhelm", "burnout", "pressure",
                     "workload", "exhausted", "overwork", "deadline"]
        texts = []
        for row in ds:
            ctx = str(row.get("Context", ""))
            if any(kw in ctx.lower() for kw in stress_kw) and len(ctx.split()) >= 10:
                texts.append(ctx)
        texts = random.sample(texts, min(TARGET_PER_CLASS, len(texts)))
        print(f"  Extracted {len(texts)} stress contexts")
        return texts
    except Exception as e:
        print(f"  Stress fallback failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────────
# 4. INSOMNIA — SleepReddit / mental health sleep posts
# ─────────────────────────────────────────────────────────────────
def load_insomnia():
    print("\n[Insomnia] Downloading sleep disorder posts...")
    try:
        from datasets import load_dataset
        ds = load_dataset("Amod/mental_health_counseling_conversations",
                          split="train")
        sleep_kw = ["insomnia", "can't sleep", "cannot sleep", "sleep problem",
                    "sleep disorder", "awake", "lying awake", "sleep onset",
                    "waking up", "sleepless", "sleep quality", "fatigue",
                    "unrested", "sleep schedule", "melatonin", "tired"]
        texts = []
        for row in ds:
            ctx = str(row.get("Context", ""))
            if any(kw in ctx.lower() for kw in sleep_kw) and len(ctx.split()) >= 10:
                texts.append(ctx)
        print(f"  Found {len(texts)} insomnia contexts from counseling data")
        if texts:
            texts = random.sample(texts, min(TARGET_PER_CLASS, len(texts)))
            return texts
    except Exception as e:
        print(f"  Insomnia source 1 failed: {e}")

    try:
        from datasets import load_dataset
        ds = load_dataset("vibhorag101/reddit-mental-health-dataset",
                          split="train", trust_remote_code=True)
        texts = []
        for row in ds:
            text = str(row.get("post", "") or row.get("text", ""))
            if any(kw in text.lower() for kw in ["insomnia","sleep","awake","tired"]) \
               and len(text.split()) >= 10:
                texts.append(text)
        texts = random.sample(texts, min(TARGET_PER_CLASS, len(texts)))
        print(f"  Extracted {len(texts)} sleep-related posts")
        return texts
    except Exception as e:
        print(f"  Insomnia fallback failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────────
# 5. PANIC DISORDER — Subset of anxiety data with panic keywords
# ─────────────────────────────────────────────────────────────────
def load_panic(anxiety_texts):
    print("\n[Panic Disorder] Filtering panic-specific posts...")
    panic_kw = ["panic attack", "panic attacks", "heart racing", "heart pounding",
                "chest tight", "fear of dying", "can't breathe", "shortness of breath",
                "derealization", "depersonalization", "fear of losing control",
                "sudden fear", "unexpected fear", "avoidance", "agoraphobia"]

    # Filter from anxiety texts
    panic_texts = [t for t in anxiety_texts
                   if any(kw in t.lower() for kw in panic_kw)]

    # Also try counseling conversations
    try:
        from datasets import load_dataset
        ds = load_dataset("Amod/mental_health_counseling_conversations",
                          split="train")
        for row in ds:
            ctx = str(row.get("Context", ""))
            if any(kw in ctx.lower() for kw in panic_kw) and len(ctx.split()) >= 10:
                panic_texts.append(ctx)
    except Exception:
        pass

    panic_texts = list(set(panic_texts))
    panic_texts = random.sample(panic_texts, min(TARGET_PER_CLASS, len(panic_texts)))
    print(f"  Found {len(panic_texts)} panic disorder posts")
    return panic_texts


# ─────────────────────────────────────────────────────────────────
# BUILD DATASET
# ─────────────────────────────────────────────────────────────────
def build():
    all_records = []

    # Load each class
    dep_texts    = load_depression()
    anx_texts    = load_anxiety()
    stress_texts = load_stress()
    ins_texts    = load_insomnia()
    panic_texts  = load_panic(anx_texts)

    class_data = [
        ("Depression",    dep_texts),
        ("Anxiety",       anx_texts),
        ("Stress",        stress_texts),
        ("Insomnia",      ins_texts),
        ("Panic Disorder",panic_texts),
    ]

    for label, texts in class_data:
        count = 0
        for t in texts:
            clean = str(t).strip()
            if len(clean.split()) >= 8:
                all_records.append({
                    "text":   clean[:500],
                    "label":  label,
                    "source": "real_dataset",
                })
                count += 1
        source_counts[label] = count
        print(f"  {label:<20}: {count} records")

    df = pd.DataFrame(all_records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n{'='*60}")
    print(f"  Total records: {len(df)}")
    print(f"\n  Class distribution:")
    print(df["label"].value_counts().to_string())

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Saved to: {OUTPUT_CSV}")

    # Save metadata
    meta = {
        "total_records": len(df),
        "records_per_class": source_counts,
        "sources": {
            "Depression":    "Reddit Depression Dataset (Kaggle) / HuggingFace mrm8488/depression-detect",
            "Anxiety":       "vibhorag101/reddit-mental-health-dataset / Amod counseling conversations",
            "Stress":        "Dreaddit — Turcan & McKeown (2019), EMNLP",
            "Insomnia":      "Amod/mental_health_counseling_conversations (sleep subset)",
            "Panic Disorder":"Anxiety dataset filtered by panic keywords",
        },
        "citations": [
            "Turcan, R., & McKeown, K. (2019). Dreaddit: A Reddit Dataset for Stress Analysis. EMNLP 2019.",
            "Demszky et al. (2020). GoEmotions: A Dataset of Fine-Grained Emotions. ACL 2020.",
            "Shen et al. Reddit Depression Dataset.",
            "Amod/mental_health_counseling_conversations. HuggingFace.",
        ],
        "note": "All data sourced from publicly available real patient-written text. No synthetic phrases used.",
    }
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata: {META_FILE}")
    print(f"{'='*60}")

    return df


if __name__ == "__main__":
    df = build()
    print("\nNext step:")
    print("  python training/train_disorder_model.py")
    print("\nThe disorder model will automatically use this real dataset.")