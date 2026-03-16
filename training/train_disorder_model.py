"""
train_disorder_model_colab.py — v2 (production)

Trains a DistilBERT classifier for mental health disorder detection.
5 classes: Depression, Anxiety, Stress, Insomnia, Panic Disorder

Uses:
  - Real Reddit depression data (depression_dataset_reddit_cleaned.csv)
  - Synthetic data for other 4 classes
  - Class weights for imbalance
  - 3 epochs on GPU (~10-15 min on Colab T4)

Expected accuracy: ~88-93%
"""

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

# ── Config ───────────────────────────────────────────
MAX_LEN    = 128
BATCH_SIZE = 32
EPOCHS     = 3
LR         = 2e-5
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_SAVE  = "/content/models/disorder_model"
METRICS_OUT = "/content/results/metrics.json"

os.makedirs(MODEL_SAVE,                   exist_ok=True)
os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)

DISORDER_LABELS = ["Depression", "Anxiety", "Stress", "Insomnia", "Panic Disorder"]
LABEL2ID = {l: i for i, l in enumerate(DISORDER_LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

print("=== Disorder Classifier Training ===")
print(f"Device : {DEVICE}")
print(f"Classes: {DISORDER_LABELS}")
print(f"Epochs : {EPOCHS} | Batch: {BATCH_SIZE} | MaxLen: {MAX_LEN}")

# ── Step 1: Build dataset ─────────────────────────────
print("\n[1/5] Building dataset...")

SEED_PHRASES = {
    "Anxiety": [
        "I feel anxious and restless all the time and cannot calm down.",
        "My heart races and I feel intense fear before important events.",
        "I constantly worry about everything even small things.",
        "I cannot stop overthinking and my mind races at night.",
        "I feel nervous in social situations and want to escape.",
        "I am always afraid something bad is going to happen.",
        "My anxiety is so bad I cannot function normally at work.",
        "I feel a constant sense of dread and unease throughout the day.",
        "I worry about my health, relationships, and future constantly.",
        "Anxiety makes it hard for me to concentrate on anything.",
        "I feel tense and on edge even when nothing is wrong.",
        "I experience excessive fear that is hard to control.",
    ],
    "Stress": [
        "I am completely overwhelmed with work and responsibilities.",
        "I feel burned out and exhausted from everything in my life.",
        "I cannot relax even on weekends because work follows me.",
        "Too many demands are placed on me and I cannot cope.",
        "I snap at people around me when I am under pressure.",
        "I feel constant tension in my neck shoulders and back.",
        "Work deadlines and pressure make me feel physically sick.",
        "I feel like everything is too much and I cannot handle it.",
        "I am running on empty and have nothing left to give.",
        "The stress of daily life is affecting my physical health.",
        "I feel trapped and overwhelmed by my circumstances.",
        "Financial pressure and work stress are destroying my health.",
    ],
    "Insomnia": [
        "I cannot fall asleep no matter how tired I am at night.",
        "I wake up multiple times during the night and cannot sleep.",
        "I lie awake for hours before finally falling asleep.",
        "I feel completely exhausted but my mind will not let me sleep.",
        "My sleep schedule is completely disrupted and irregular.",
        "I dread going to bed because I know I will not sleep.",
        "I sleep only two or three hours a night and feel terrible.",
        "Even after sleeping I wake up feeling completely unrested.",
        "I have not had a good night sleep in weeks or months.",
        "Racing thoughts at night prevent me from falling asleep.",
        "I depend on medication just to get a few hours of sleep.",
        "Lack of sleep is affecting every area of my life badly.",
    ],
    "Panic Disorder": [
        "I experience sudden intense episodes of extreme fear and panic.",
        "My chest tightens and I feel like I am dying during attacks.",
        "I avoid places and situations where panic attacks have happened.",
        "I live in constant fear of having another panic attack.",
        "Sweating shaking and dizziness come on suddenly without warning.",
        "I feel completely detached from reality during a panic episode.",
        "My heart pounds uncontrollably and I cannot breathe properly.",
        "I went to the emergency room thinking I was having a heart attack.",
        "Panic attacks come out of nowhere and leave me exhausted.",
        "I cannot go out alone because I fear having a panic attack.",
        "The anticipatory anxiety about panic attacks controls my life.",
        "I have changed my entire lifestyle to avoid triggering panic.",
    ],
}

VARIATIONS = [
    "",
    " This has been going on for several months now.",
    " It is significantly affecting my daily life and relationships.",
    " I have tried many things but nothing seems to help.",
    " My doctor suggested I seek professional mental health support.",
    " I struggle to maintain my job and relationships because of this.",
    " I feel like I will never get better and it scares me.",
    " My family is very worried about my mental health.",
    " I cannot remember the last time I felt normal and okay.",
    " This condition is ruining my quality of life completely.",
]

records = []
random.seed(42)

# ── Real depression data ──
dep_csv = "/content/depression_dataset_reddit_cleaned.csv"
if os.path.exists(dep_csv):
    df_dep    = pd.read_csv(dep_csv)
    dep_texts = df_dep[df_dep["is_depression"] == 1]["clean_text"].dropna().tolist()
    dep_texts = random.sample(dep_texts, min(3000, len(dep_texts)))
    for t in dep_texts:
        records.append({"text": str(t)[:400], "label": "Depression"})
    print(f"  Real depression samples : {len(dep_texts)}")
else:
    # Synthetic depression fallback
    dep_seeds = [
        "I feel completely hopeless and worthless every single day.",
        "Nothing brings me joy anymore and I feel completely empty inside.",
        "I have been crying every day for weeks and cannot stop.",
        "I cannot get out of bed and face the world anymore.",
        "I feel like a burden to everyone around me all the time.",
        "Life feels completely meaningless and I am utterly exhausted.",
        "I have lost all interest in activities I used to enjoy.",
        "I feel totally isolated and nobody understands my deep pain.",
        "I cannot concentrate on anything and my memory is failing.",
        "I feel numb and disconnected from everyone and everything.",
        "Dark thoughts consume me and I cannot escape them.",
        "I have no motivation to do anything even basic self-care.",
    ]
    for _ in range(2000):
        phrase = random.choice(dep_seeds)
        var    = random.choice(VARIATIONS)
        records.append({"text": phrase + var, "label": "Depression"})
    print("  Using synthetic depression data (upload CSV for better results)")

# ── Synthetic data for other classes ──
for disorder, seeds in SEED_PHRASES.items():
    count = 0
    for _ in range(800):
        phrase = random.choice(seeds)
        var    = random.choice(VARIATIONS)
        records.append({"text": phrase + var, "label": disorder})
        count += 1
    print(f"  {disorder:20s}: {count} synthetic samples")

df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"\n  Total dataset: {len(df)} samples")
print("\n  Class distribution:")
print(df["label"].value_counts().to_string())

X = df["text"].values
y = df["label"].map(LABEL2ID).values

X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)

print(f"\n  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ── Class weights ──
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(DISORDER_LABELS)),
    y=y_train,
)
weight_tensor = torch.tensor(weights, dtype=torch.float).to(DEVICE)
print(f"\n  Class weights: { {l: round(float(w),2) for l,w in zip(DISORDER_LABELS, weights)} }")

# ── Step 2: Tokenizer + Model ─────────────────────────
print("\n[2/5] Loading DistilBERT...")
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

# ── Step 3: Dataset ───────────────────────────────────
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

# ── Step 4: Train ─────────────────────────────────────
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
                "id2label": {str(i): l for i, l in ID2LABEL.items()},
                "label2id": LABEL2ID,
                "num_labels": len(DISORDER_LABELS),
            }, f, indent=2)
        print(f"  ✓ Saved (acc={val_acc:.4f})")

# ── Step 5: Test evaluation ───────────────────────────
print("\n[4/5] Final test evaluation...")
test_preds, test_labels = run_eval(test_loader)

print("\n=== Classification Report ===")
print(classification_report(test_labels, test_preds, target_names=DISORDER_LABELS))

print("=== Confusion Matrix ===")
cm = confusion_matrix(test_labels, test_preds)
print(pd.DataFrame(cm, index=DISORDER_LABELS, columns=DISORDER_LABELS).to_string())

metrics = {
    "model":      "disorder_model",
    "labels":     DISORDER_LABELS,
    "accuracy":   float(accuracy_score(test_labels, test_preds)),
    "precision":  float(precision_score(test_labels, test_preds, average="macro", zero_division=0)),
    "recall":     float(recall_score(test_labels,    test_preds, average="macro", zero_division=0)),
    "f1_macro":   float(f1_score(test_labels,        test_preds, average="macro", zero_division=0)),
    "confusion_matrix": cm.tolist(),
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
existing["disorder_model"] = metrics
json.dump(existing, open(METRICS_OUT, "w"), indent=2)

print(f"\n[5/5] Done!")
print(f"  Model   → {MODEL_SAVE}")
print(f"  Metrics → {METRICS_OUT}")
print("""
Download model:
  import shutil
  from google.colab import files
  shutil.make_archive('disorder_model', 'zip', '/content/models/disorder_model')
  files.download('disorder_model.zip')
""")