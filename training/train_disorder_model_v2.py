"""
train_disorder_model_colab_v2.py
─────────────────────────────────────────────────────────────────
Complete disorder classifier training for Google Colab GPU.

Uses REAL datasets only:
  - Depression  : Reddit Depression Dataset
  - Anxiety     : Reddit Mental Health Dataset
  - Stress      : Dreaddit (Turcan & McKeown, EMNLP 2019)
  - Insomnia    : Filtered from counseling conversations
  - Panic       : Filtered panic-keyword posts

Run in Colab:
  1. Runtime → Change runtime type → T4 GPU
  2. Upload: depression_dataset_reddit_cleaned.csv
  3. Run all cells

Expected time: ~15-20 min on T4 GPU
Expected accuracy: ~78-85%
"""

# ─────────────────────────────────────────────────────────────────
# CELL 1 — Install dependencies
# ─────────────────────────────────────────────────────────────────
# !pip install transformers torch datasets scikit-learn pandas numpy -q

import os, json, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
MAX_LEN        = 128
BATCH_SIZE     = 32
EPOCHS         = 3
LR             = 2e-5
TARGET_SAMPLES = 1500    # per class target
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_SAVE  = "/content/models/disorder_model"
METRICS_OUT = "/content/results/metrics.json"

os.makedirs(MODEL_SAVE,                   exist_ok=True)
os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)

DISORDER_LABELS = ["Depression", "Anxiety", "Stress", "Insomnia", "Panic Disorder"]
LABEL2ID = {l: i for i, l in enumerate(DISORDER_LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

print(f"=== Disorder Classifier Training (Real Data) ===")
print(f"Device : {DEVICE}")
print(f"Classes: {DISORDER_LABELS}")
print(f"Epochs : {EPOCHS} | Batch: {BATCH_SIZE}")

# ─────────────────────────────────────────────────────────────────
# STEP 1 — Load real datasets
# ─────────────────────────────────────────────────────────────────
print("\n[1/5] Loading real datasets...")

from datasets import load_dataset

random.seed(42)
records = []


# ── Depression ── Reddit Depression (upload file to Colab)
dep_path = "/content/depression_dataset_reddit_cleaned.csv"
if os.path.exists(dep_path):
    df_dep    = pd.read_csv(dep_path)
    dep_texts = df_dep[df_dep["is_depression"] == 1]["clean_text"].dropna().tolist()
    dep_texts = [t for t in dep_texts if len(str(t).split()) >= 8]
    dep_texts = random.sample(dep_texts, min(TARGET_SAMPLES, len(dep_texts)))
    for t in dep_texts:
        records.append({"text": str(t)[:500], "label": "Depression", "source": "reddit_depression"})
    print(f"  Depression  : {len(dep_texts)} real Reddit posts")
else:
    print("  Depression  : File not found! Upload depression_dataset_reddit_cleaned.csv")
    print("  Downloading from HuggingFace fallback...")
    try:
        ds = load_dataset("mrm8488/depression-detect", split="train")
        dep_texts = [r["text"] for r in ds if r.get("label") == 1
                     and len(str(r.get("text","")).split()) >= 8]
        dep_texts = random.sample(dep_texts, min(TARGET_SAMPLES, len(dep_texts)))
        for t in dep_texts:
            records.append({"text": str(t)[:500], "label": "Depression", "source": "hf_depression"})
        print(f"  Depression  : {len(dep_texts)} posts from HuggingFace")
    except Exception as e:
        print(f"  Depression fallback failed: {e}")


# ── Anxiety ── Reddit Mental Health Dataset
print("  Downloading Anxiety data...")
try:
    ds_mh = load_dataset("vibhorag101/reddit-mental-health-dataset",
                          split="train", trust_remote_code=True)
    anx_texts = []
    for row in ds_mh:
        label = str(row.get("label", "")).lower()
        text  = str(row.get("post", "") or row.get("text", ""))
        if "anxi" in label and len(text.split()) >= 8:
            anx_texts.append(text)
    anx_texts = random.sample(anx_texts, min(TARGET_SAMPLES, len(anx_texts)))
    for t in anx_texts:
        records.append({"text": str(t)[:500], "label": "Anxiety", "source": "reddit_mental_health"})
    print(f"  Anxiety     : {len(anx_texts)} real posts")
except Exception as e:
    print(f"  Anxiety primary failed: {e}, trying counseling data...")
    try:
        ds_c = load_dataset("Amod/mental_health_counseling_conversations", split="train")
        anx_kw = ["anxious","anxiety","worry","nervous","panic","fear","overthink","restless"]
        anx_texts = [str(r.get("Context","")) for r in ds_c
                     if any(k in str(r.get("Context","")).lower() for k in anx_kw)
                     and len(str(r.get("Context","")).split()) >= 8]
        anx_texts = random.sample(anx_texts, min(TARGET_SAMPLES, len(anx_texts)))
        for t in anx_texts:
            records.append({"text": str(t)[:500], "label": "Anxiety", "source": "counseling_anxiety"})
        print(f"  Anxiety     : {len(anx_texts)} counseling contexts")
    except Exception as e2:
        print(f"  Anxiety all sources failed: {e2}")
        anx_texts = []


# ── Stress ── Dreaddit (Turcan & McKeown, EMNLP 2019)
print("  Downloading Stress data (Dreaddit)...")
try:
    ds_stress = load_dataset("dreaddit", split="train", trust_remote_code=True)
    stress_texts = [str(r.get("text","")) for r in ds_stress
                    if r.get("label") == 1 and len(str(r.get("text","")).split()) >= 8]
    stress_texts = random.sample(stress_texts, min(TARGET_SAMPLES, len(stress_texts)))
    for t in stress_texts:
        records.append({"text": str(t)[:500], "label": "Stress", "source": "dreaddit"})
    print(f"  Stress      : {len(stress_texts)} Dreaddit posts (Turcan & McKeown 2019)")
except Exception as e:
    print(f"  Dreaddit failed: {e}, using counseling fallback...")
    try:
        ds_c = load_dataset("Amod/mental_health_counseling_conversations", split="train")
        s_kw = ["stress","overwhelm","burnout","pressure","workload","exhausted","deadline"]
        stress_texts = [str(r.get("Context","")) for r in ds_c
                        if any(k in str(r.get("Context","")).lower() for k in s_kw)
                        and len(str(r.get("Context","")).split()) >= 8]
        stress_texts = random.sample(stress_texts, min(TARGET_SAMPLES, len(stress_texts)))
        for t in stress_texts:
            records.append({"text": str(t)[:500], "label": "Stress", "source": "counseling_stress"})
        print(f"  Stress      : {len(stress_texts)} contexts")
    except Exception as e2:
        print(f"  Stress all failed: {e2}")
        stress_texts = []


# ── Insomnia ── Counseling + mental health filtered posts
print("  Downloading Insomnia data...")
try:
    ds_c = load_dataset("Amod/mental_health_counseling_conversations", split="train")
    ins_kw = ["insomnia","can't sleep","cannot sleep","sleep problem","awake",
              "lying awake","sleep onset","waking up","sleepless","unrested",
              "sleep quality","tired all the time","sleep schedule","melatonin"]
    ins_texts = [str(r.get("Context","")) for r in ds_c
                 if any(k in str(r.get("Context","")).lower() for k in ins_kw)
                 and len(str(r.get("Context","")).split()) >= 8]

    # Also from Reddit mental health if available
    try:
        for row in ds_mh:
            text = str(row.get("post","") or row.get("text",""))
            if any(k in text.lower() for k in ["insomnia","sleep","awake"]) \
               and len(text.split()) >= 8:
                ins_texts.append(text)
    except Exception:
        pass

    ins_texts = list(set(ins_texts))
    ins_texts = random.sample(ins_texts, min(TARGET_SAMPLES, len(ins_texts)))
    for t in ins_texts:
        records.append({"text": str(t)[:500], "label": "Insomnia", "source": "counseling_insomnia"})
    print(f"  Insomnia    : {len(ins_texts)} sleep-related posts")
except Exception as e:
    print(f"  Insomnia failed: {e}")
    ins_texts = []


# ── Panic Disorder ── From anxiety data filtered by panic keywords
print("  Extracting Panic Disorder data...")
panic_kw = ["panic attack","panic attacks","heart racing","heart pounding",
            "chest tight","fear of dying","can't breathe","shortness of breath",
            "derealization","depersonalization","fear of losing control",
            "sudden fear","unexpected fear","agoraphobia","palpitation"]

# From anxiety texts already loaded
panic_texts = [t for t in anx_texts if any(k in t.lower() for k in panic_kw)]

# Also from counseling
try:
    for r in ds_c:
        ctx = str(r.get("Context",""))
        if any(k in ctx.lower() for k in panic_kw) and len(ctx.split()) >= 8:
            panic_texts.append(ctx)
except Exception:
    pass

panic_texts = list(set(panic_texts))
panic_texts = random.sample(panic_texts, min(TARGET_SAMPLES, len(panic_texts)))
for t in panic_texts:
    records.append({"text": str(t)[:500], "label": "Panic Disorder", "source": "panic_filtered"})
print(f"  Panic       : {len(panic_texts)} panic-specific posts")


# ── Summary ──
df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"\n  Total real records: {len(df)}")
print("\n  Class distribution:")
print(df["label"].value_counts().to_string())
print("\n  Source breakdown:")
print(df["source"].value_counts().to_string())

# Save for reference
df.to_csv("/content/disorder_dataset_real.csv", index=False)
print("\n  Dataset saved to /content/disorder_dataset_real.csv")


# ─────────────────────────────────────────────────────────────────
# STEP 2 — Prepare features
# ─────────────────────────────────────────────────────────────────
print("\n[2/5] Preparing data splits...")

X = df["text"].values
y = df["label"].map(LABEL2ID).values

X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)

print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# Class weights
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(DISORDER_LABELS)),
    y=y_train,
)
weight_tensor = torch.tensor(weights, dtype=torch.float).to(DEVICE)
print(f"  Class weights: { {l: round(float(w),2) for l,w in zip(DISORDER_LABELS,weights)} }")


# ─────────────────────────────────────────────────────────────────
# STEP 3 — Model + Dataset
# ─────────────────────────────────────────────────────────────────
print("\n[3/5] Loading DistilBERT...")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(DISORDER_LABELS),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    ignore_mismatched_sizes=True,
).to(DEVICE)

loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)
print(f"  Model ready on {DEVICE}")


class DisorderDS(Dataset):
    def __init__(self, texts, labels):
        self.enc = tokenizer(
            list(texts), truncation=True,
            padding="max_length", max_length=MAX_LEN,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, i):
        return {
            "input_ids":      self.enc["input_ids"][i],
            "attention_mask": self.enc["attention_mask"][i],
            "labels":         self.labels[i],
        }

train_loader = DataLoader(DisorderDS(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(DisorderDS(X_val,   y_val),   batch_size=BATCH_SIZE)
test_loader  = DataLoader(DisorderDS(X_test,  y_test),  batch_size=BATCH_SIZE)


# ─────────────────────────────────────────────────────────────────
# STEP 4 — Train
# ─────────────────────────────────────────────────────────────────
print(f"\n[4/5] Training ({len(train_loader)} steps/epoch)...")
optimizer   = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler   = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=max(1, total_steps // 10),
    num_training_steps=total_steps,
)

def run_eval(loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                input_ids      = batch["input_ids"].to(DEVICE),
                attention_mask = batch["attention_mask"].to(DEVICE),
            ).logits
            all_preds.extend(logits.argmax(-1).cpu().numpy())
            all_labels.extend(batch["labels"].numpy())
    return np.array(all_preds), np.array(all_labels)

best_val_acc = 0.0
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        logits = model(
            input_ids      = batch["input_ids"].to(DEVICE),
            attention_mask = batch["attention_mask"].to(DEVICE),
        ).logits
        loss = loss_fn(logits, batch["labels"].to(DEVICE))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        if (i + 1) % 30 == 0:
            print(f"  Ep{epoch} step {i+1}/{len(train_loader)} loss={loss.item():.4f}")

    val_preds, val_labels = run_eval(val_loader)
    val_acc = accuracy_score(val_labels, val_preds)
    val_f1  = f1_score(val_labels, val_preds, average="macro", zero_division=0)
    print(f"\n  ── Epoch {epoch}/{EPOCHS} | loss={total_loss/len(train_loader):.4f} "
          f"| val_acc={val_acc:.4f} | val_F1={val_f1:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model.save_pretrained(MODEL_SAVE)
        tokenizer.save_pretrained(MODEL_SAVE)
        with open(os.path.join(MODEL_SAVE, "label_map.json"), "w") as f:
            json.dump({
                "id2label":   {str(i): l for i, l in ID2LABEL.items()},
                "label2id":   LABEL2ID,
                "num_labels": len(DISORDER_LABELS),
                "trained_on": "real_datasets_only",
            }, f, indent=2)
        print(f"  ✓ Saved (acc={val_acc:.4f})")


# ─────────────────────────────────────────────────────────────────
# STEP 5 — Evaluate
# ─────────────────────────────────────────────────────────────────
print("\n[5/5] Final test evaluation...")
test_preds, test_labels = run_eval(test_loader)

print("\n=== Classification Report ===")
present_labels = sorted(set(test_labels) | set(test_preds))
present_names  = [DISORDER_LABELS[i] for i in present_labels if i < len(DISORDER_LABELS)]
print(classification_report(test_labels, test_preds,
                             labels=present_labels, target_names=present_names))

print("\n=== Confusion Matrix ===")
cm = confusion_matrix(test_labels, test_preds, labels=present_labels)
cm_df = pd.DataFrame(cm, index=present_names, columns=present_names)
print(cm_df.to_string())

metrics = {
    "model":      "disorder_model",
    "labels":     DISORDER_LABELS,
    "trained_on": "real_datasets_only",
    "sources":    ["reddit_depression", "reddit_mental_health", "dreaddit", "counseling_conversations"],
    "accuracy":   float(accuracy_score(test_labels, test_preds)),
    "precision":  float(precision_score(test_labels, test_preds, average="macro", zero_division=0)),
    "recall":     float(recall_score(test_labels,    test_preds, average="macro", zero_division=0)),
    "f1_macro":   float(f1_score(test_labels,        test_preds, average="macro", zero_division=0)),
    "confusion_matrix": cm.tolist(),
}

print(f"\n=== Summary ===")
print(f"  Accuracy  : {metrics['accuracy']:.4f}")
print(f"  Precision : {metrics['precision']:.4f}")
print(f"  Recall    : {metrics['recall']:.4f}")
print(f"  F1 Macro  : {metrics['f1_macro']:.4f}")
print(f"\n  Note: Training on REAL data — accuracy may be lower than")
print(f"  synthetic baseline but is genuinely more valid.")

existing = {}
if os.path.exists(METRICS_OUT):
    try: existing = json.load(open(METRICS_OUT))
    except: pass
existing["disorder_model"] = metrics
json.dump(existing, open(METRICS_OUT, "w"), indent=2)

print(f"\n  Model   → {MODEL_SAVE}")
print(f"  Metrics → {METRICS_OUT}")

print("""
Download model:
  import shutil
  from google.colab import files
  shutil.make_archive('disorder_model', 'zip', '/content/models/disorder_model')
  files.download('disorder_model.zip')
  files.download('/content/results/metrics.json')
""")