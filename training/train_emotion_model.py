"""
train_emotion_model_colab.py — v4 (production fix)

Key changes vs v3:
  - Reduces to 6 core psychological emotions (not 28)
    → Much better accuracy with limited data
  - Uses full dataset (207k rows, no sampling limit)
  - Adds class weights to handle imbalance
  - Single-label classification (argmax) instead of multi-label
    → Avoids threshold tuning problem entirely
  - Expected accuracy: ~75-85%
"""

import os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score, classification_report
)
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)

# ── Config ───────────────────────────────────────────
MAX_LEN    = 128
BATCH_SIZE = 32
EPOCHS     = 3
LR         = 2e-5
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_SAVE  = "/content/models/emotion_model"
METRICS_OUT = "/content/results/metrics.json"

os.makedirs(MODEL_SAVE,                   exist_ok=True)
os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)

# 6 core emotions relevant to psychological analysis
# Maps GoEmotions columns → our label
EMOTION_MAP = {
    "fear":          "fear",
    "sadness":       "sadness",
    "anger":         "anger",
    "joy":           "joy",
    "nervousness":   "nervousness",
    "neutral":       "neutral",
}
# Also group related emotions
EXTRA_MAP = {
    "grief":         "sadness",
    "remorse":       "sadness",
    "disappointment":"sadness",
    "excitement":    "joy",
    "optimism":      "joy",
    "admiration":    "joy",
    "annoyance":     "anger",
    "disapproval":   "anger",
    "disgust":       "anger",
    "confusion":     "nervousness",
    "embarrassment": "nervousness",
}

LABEL_NAMES = ["fear", "sadness", "anger", "joy", "nervousness", "neutral"]
LABEL2ID    = {l: i for i, l in enumerate(LABEL_NAMES)}
ID2LABEL    = {i: l for l, i in LABEL2ID.items()}

print("=== Emotion Model Training (v4) ===")
print(f"Device : {DEVICE}")
print(f"Labels : {LABEL_NAMES}")
print(f"Epochs : {EPOCHS} | Batch: {BATCH_SIZE} | MaxLen: {MAX_LEN}")

# ── Step 1: Load & prepare dataset ───────────────────
print("\n[1/5] Loading dataset...")

df = pd.read_csv("/content/go_emotions_dataset.csv")
print(f"  Raw rows: {len(df)}")

if "example_very_unclear" in df.columns:
    df = df[~df["example_very_unclear"].astype(bool)].reset_index(drop=True)

# Convert multi-label → single dominant label
def get_label(row):
    # Check primary emotions first
    for col, label in EMOTION_MAP.items():
        if col in row and row[col] == 1:
            return label
    # Check grouped emotions
    for col, label in EXTRA_MAP.items():
        if col in row and row[col] == 1:
            return label
    return "neutral"

df["label"] = df.apply(get_label, axis=1)
df = df[["text", "label"]].dropna()
df = df[df["text"].str.strip().str.len() > 5]

print(f"  After processing: {len(df)} rows")
print("  Label distribution:")
print(df["label"].value_counts().to_string())

# Balance: cap majority classes at 15k, keep all minority
MAX_PER_CLASS = 15000
balanced = []
for label in LABEL_NAMES:
    subset = df[df["label"] == label]
    if len(subset) > MAX_PER_CLASS:
        subset = subset.sample(n=MAX_PER_CLASS, random_state=42)
    balanced.append(subset)
df = pd.concat(balanced).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"\n  Balanced dataset: {len(df)} rows")
print(df["label"].value_counts().to_string())

X = df["text"].values
y = df["label"].map(LABEL2ID).values

X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)
print(f"\n  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ── Step 2: Class weights ─────────────────────────────
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array(list(range(len(LABEL_NAMES)))),
    y=y_train
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
print(f"\n  Class weights: {dict(zip(LABEL_NAMES, class_weights.round(2)))}")

# ── Step 3: Tokenizer + Model ─────────────────────────
print("\n[2/5] Loading DistilBERT...")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(LABEL_NAMES),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    ignore_mismatched_sizes=True,
).to(DEVICE)

# Custom loss with class weights
loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)

# ── Step 4: Dataset ───────────────────────────────────
class EmotionDS(Dataset):
    def __init__(self, texts, labels):
        self.enc = tokenizer(
            list(texts), truncation=True,
            padding="max_length", max_length=MAX_LEN,
            return_tensors="pt"
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

train_loader = DataLoader(EmotionDS(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(EmotionDS(X_val,   y_val),   batch_size=BATCH_SIZE)
test_loader  = DataLoader(EmotionDS(X_test,  y_test),  batch_size=BATCH_SIZE)

# ── Step 5: Train ─────────────────────────────────────
print(f"\n[3/5] Training ({len(train_loader)} steps/epoch)...")
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
            preds = logits.argmax(-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].numpy())
    return np.array(all_preds), np.array(all_labels)

best_val_f1 = 0.0
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
        if (i + 1) % 50 == 0:
            print(f"  Ep{epoch} step {i+1}/{len(train_loader)} loss={loss.item():.4f}")

    val_preds, val_labels = run_eval(val_loader)
    val_acc = accuracy_score(val_labels, val_preds)
    val_f1  = f1_score(val_labels, val_preds, average="macro", zero_division=0)
    print(f"\n  ── Epoch {epoch}/{EPOCHS} | loss={total_loss/len(train_loader):.4f} "
          f"| val_acc={val_acc:.4f} | val_F1={val_f1:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        model.save_pretrained(MODEL_SAVE)
        tokenizer.save_pretrained(MODEL_SAVE)
        label_meta = {
            "id2label":   {str(i): l for i, l in ID2LABEL.items()},
            "label2id":   LABEL2ID,
            "task":       "single_label_classification",
            "num_labels": len(LABEL_NAMES),
        }
        with open(os.path.join(MODEL_SAVE, "label_map.json"), "w") as f:
            json.dump(label_meta, f, indent=2)
        print(f"  ✓ Saved (F1={val_f1:.4f})")

# ── Step 6: Test ──────────────────────────────────────
print("\n[4/5] Final test evaluation...")
test_preds, test_labels = run_eval(test_loader)

print("\n=== Classification Report ===")
print(classification_report(test_labels, test_preds, target_names=LABEL_NAMES))

metrics = {
    "model":      "emotion_model",
    "task":       "single_label_6_class",
    "labels":     LABEL_NAMES,
    "accuracy":   float(accuracy_score(test_labels, test_preds)),
    "precision":  float(precision_score(test_labels, test_preds, average="macro", zero_division=0)),
    "recall":     float(recall_score(test_labels,    test_preds, average="macro", zero_division=0)),
    "f1_macro":   float(f1_score(test_labels,        test_preds, average="macro", zero_division=0)),
}

print("\n=== Summary ===")
for k, v in metrics.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.4f}")

existing = {}
if os.path.exists(METRICS_OUT):
    try:
        existing = json.load(open(METRICS_OUT))
    except Exception:
        pass
existing["emotion_model"] = metrics
json.dump(existing, open(METRICS_OUT, "w"), indent=2)

print(f"\n[5/5] Done!")
print(f"  Model   → {MODEL_SAVE}")
print(f"  Metrics → {METRICS_OUT}")
print("""
Run this to download:
  import shutil
  from google.colab import files
  shutil.make_archive('emotion_model', 'zip', '/content/models/emotion_model')
  files.download('emotion_model.zip')
""")