# ============================================================
# ⚡ COMPLETE — fixes all root causes
# ============================================================

#!pip install transformers datasets seaborn wordcloud -q

import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from torch.optim import AdamW
from torch.amp import autocast, GradScaler

# ============================================================
# ⚙️ CONFIG
# ============================================================
MAX_LEN    = 128
BATCH_SIZE = 32
EPOCHS     = 8
LR         = 2e-5
SEED       = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

DATA_PATH  = "/kaggle/input/datasets/yuktabaid/go-emotions-dataset/go_emotions_dataset.csv"
SAVE_PATH  = "/kaggle/working/emotion_model_best"
SAVE_PATH2 = "/kaggle/working/emotion_model_2"
os.makedirs(SAVE_PATH,  exist_ok=True)
os.makedirs(SAVE_PATH2, exist_ok=True)

LABELS   = ["fear", "sadness", "anger", "joy", "nervousness", "neutral"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

# ============================================================
# ✅ FIX 1 — RIGHT MODEL
# cardiffnlp has 4 classes → mismatched head → garbage init
# roberta-base has NO emotion bias → trains cleanly from scratch
# ============================================================
MODEL_NAME = "roberta-base"

# ============================================================
# 📥 LOAD DATA
# ============================================================
df = pd.read_csv(DATA_PATH)

def map_label(row):
    for emotion in ["fear", "sadness", "anger", "joy", "nervousness"]:
        if row.get(emotion, 0) == 1:
            return emotion
    return "neutral"

df["label"] = df.apply(map_label, axis=1)
df = df[["text", "label"]].dropna()
df = df[df["text"].str.strip().str.len() > 5]

print("Raw distribution:\n", df["label"].value_counts())

# ============================================================
# ✅ FIX 2 — SMARTER RESAMPLING
# Problem: nervousness=437 raw → even after oversampling
# it's too few. We oversample it much more aggressively.
# ============================================================
TARGET = {
    "neutral":     6000,
    "joy":         5000,
    "anger":       5000,
    "sadness":     5000,
    "fear":        4000,
    "nervousness": 3000,   # was 1000-1500 → massive jump
}

balanced = []
for label, target_n in TARGET.items():
    subset = df[df["label"] == label]
    n = len(subset)
    if n >= target_n:
        balanced.append(subset.sample(target_n, random_state=SEED))
    else:
        # oversample with replacement
        balanced.append(subset.sample(target_n, replace=True, random_state=SEED))

df_bal = pd.concat(balanced).sample(frac=1, random_state=SEED).reset_index(drop=True)
print("\nBalanced distribution:\n", df_bal["label"].value_counts())
print(f"\nTotal samples: {len(df_bal)}")

# ============================================================
# ✅ FIX 3 — STRATIFIED SPLIT with enough nervousness in val/test
# ============================================================
from sklearn.model_selection import train_test_split

X, y = df_bal["text"].values, df_bal["label"].map(LABEL2ID).values

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED
)

print(f"\nTrain: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
print("Val nervousness count:", (y_val == LABEL2ID["nervousness"]).sum())

# ============================================================
# ⚖️ CLASS WEIGHTS
# ============================================================
weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)
print("\nClass weights:", {l: f"{w:.3f}" for l, w in zip(LABELS, weights.cpu())})

# ============================================================
# ✅ FIX 4 — FOCAL LOSS (replaces label smoothing)
# Focuses training on hard/minority examples
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(
            logits, targets, weight=self.weight, reduction="none"
        )
        pt     = torch.exp(-ce)
        focal  = (1 - pt) ** self.gamma * ce
        return focal.mean()

loss_fn = FocalLoss(gamma=2.0, weight=weights)

# ============================================================
# 🤖 MODEL — roberta-base (clean 6-class head)
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS)
).to(DEVICE)

# ============================================================
# 📦 DATASET
# ============================================================
class EmotionDataset(Dataset):
    def __init__(self, texts, labels):
        self.enc = tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.enc["input_ids"][idx],
            "attention_mask": self.enc["attention_mask"][idx],
            "labels":         self.labels[idx]
        }

train_loader = DataLoader(EmotionDataset(X_train, y_train),
                          batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True)
val_loader   = DataLoader(EmotionDataset(X_val, y_val),
                          batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)
test_loader  = DataLoader(EmotionDataset(X_test, y_test),
                          batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)

# ============================================================
# ✅ FIX 5 — OPTIMIZER + OneCycleLR (works with param groups)
# ============================================================
optimizer_grouped_parameters = [
    {"params": [p for n, p in model.named_parameters()
                if "classifier" not in n], "lr": LR},
    {"params": [p for n, p in model.named_parameters()
                if "classifier" in n],     "lr": LR * 10},
]
optimizer = AdamW(optimizer_grouped_parameters, weight_decay=0.01)

total_steps = len(train_loader) * EPOCHS

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr        = [LR, LR * 10],   # one per param group
    total_steps   = total_steps,
    pct_start     = 0.1,             # 10% warmup
    anneal_strategy = "cos",
    div_factor    = 25,
    final_div_factor = 1e4,
)

scaler = GradScaler("cuda")

# ============================================================
# 📊 EVALUATE
# ============================================================
def evaluate(loader, mdl=None):
    if mdl is None: mdl = model
    mdl.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            out = mdl(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE)
            )
            all_preds.extend(torch.argmax(out.logits, 1).cpu().numpy())
            all_labels.extend(batch["labels"].numpy())
    return np.array(all_preds), np.array(all_labels)

# ============================================================
# 🔥 TRAINING
# ============================================================
best_val_f1 = -1.0   # ✅ track macro F1, not just accuracy
                      # accuracy lies when classes are imbalanced

for epoch in range(EPOCHS):
    model.train()

    # Freeze backbone epoch 0, unfreeze from epoch 1
    if epoch == 0:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        print("🔒 Backbone frozen for epoch 1")
    elif epoch == 1:
        for param in model.parameters():
            param.requires_grad = True
        print("🔓 Backbone unfrozen")

    total_loss = 0
    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()

        with autocast("cuda"):
            out  = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE)
            )
            loss = loss_fn(out.logits, batch["labels"].to(DEVICE))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()          # per-batch for OneCycleLR

        total_loss += loss.item()

        if (step + 1) % 100 == 0:
            print(f"  Step {step+1}/{len(train_loader)} | "
                  f"Loss: {loss.item():.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")

    val_preds, val_labels = evaluate(val_loader)

    # ✅ Use macro F1 as the save criterion — sensitive to nervousness/neutral
    from sklearn.metrics import f1_score
    val_f1  = f1_score(val_labels, val_preds, average="macro")
    val_acc = (val_preds == val_labels).mean()

    print(f"\n{'='*55}")
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} "
          f"| Val Acc: {val_acc:.4f} | Val Macro-F1: {val_f1:.4f}")
    print(classification_report(val_labels, val_preds,
                                 target_names=LABELS, zero_division=0))

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        model.save_pretrained(SAVE_PATH)
        tokenizer.save_pretrained(SAVE_PATH)
        print(f"✅ Best model saved! (macro-F1: {val_f1:.4f})")

# ============================================================
# 🔥 ENSEMBLE — Model 2 (different seed + LR)
# ============================================================
print("\n\n=== Training Model 2 ===")
torch.manual_seed(123)

model2 = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(LABELS)
).to(DEVICE)

opt2 = AdamW(model2.parameters(), lr=3e-5, weight_decay=0.01)
sch2 = torch.optim.lr_scheduler.OneCycleLR(
    opt2,
    max_lr       = 3e-5,
    total_steps  = len(train_loader) * 4,
    pct_start    = 0.1,
    anneal_strategy = "cos",
)
sc2 = GradScaler("cuda")
best_f1_2 = -1.0

for epoch in range(4):
    model2.train()
    for batch in train_loader:
        opt2.zero_grad()
        with autocast("cuda"):
            out  = model2(batch["input_ids"].to(DEVICE),
                          attention_mask=batch["attention_mask"].to(DEVICE))
            loss = loss_fn(out.logits, batch["labels"].to(DEVICE))
        sc2.scale(loss).backward()
        sc2.unscale_(opt2)
        torch.nn.utils.clip_grad_norm_(model2.parameters(), 1.0)
        sc2.step(opt2)
        sc2.update()
        sch2.step()

    p2, l2 = evaluate(val_loader, model2)
    f2 = f1_score(l2, p2, average="macro")
    a2 = (p2 == l2).mean()
    print(f"Model2 Epoch {epoch+1} | Val Acc: {a2:.4f} | Macro-F1: {f2:.4f}")
    if f2 > best_f1_2:
        best_f1_2 = f2
        model2.save_pretrained(SAVE_PATH2)
        print("✅ Model2 saved!")

# ============================================================
# 🎯 ENSEMBLE INFERENCE
# ============================================================
print("\n\n=== Ensemble Evaluation ===")

model_a = AutoModelForSequenceClassification.from_pretrained(
    SAVE_PATH).to(DEVICE)
model_b = AutoModelForSequenceClassification.from_pretrained(
    SAVE_PATH2).to(DEVICE)
model_a.eval(); model_b.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        inp = {
            "input_ids":      batch["input_ids"].to(DEVICE),
            "attention_mask": batch["attention_mask"].to(DEVICE)
        }
        pa = torch.softmax(model_a(**inp).logits, dim=1)
        pb = torch.softmax(model_b(**inp).logits, dim=1)
        avg = (pa + pb) / 2
        all_preds.extend(torch.argmax(avg, 1).cpu().numpy())
        all_labels.extend(batch["labels"].numpy())

ens_preds  = np.array(all_preds)
ens_labels = np.array(all_labels)

print("\n🔥 Ensemble Test Report:")
print(classification_report(ens_labels, ens_preds,
                              target_names=LABELS, zero_division=0))
print(f"Ensemble Macro-F1 : {f1_score(ens_labels, ens_preds, average='macro'):.4f}")
print(f"Ensemble Accuracy : {(ens_preds == ens_labels).mean():.4f}")

# ============================================================
# 📊 PLOTS
# ============================================================
cm = confusion_matrix(ens_labels, ens_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=LABELS, yticklabels=LABELS, cmap="Blues")
plt.title("Ensemble Confusion Matrix")
plt.ylabel("True"); plt.xlabel("Predicted")
plt.tight_layout(); plt.show()

per_class_acc = cm.diagonal() / cm.sum(axis=1)
plt.figure(figsize=(8, 4))
bars = plt.bar(LABELS, per_class_acc,
               color=["#e74c3c" if a < 0.6 else "#2ecc71" for a in per_class_acc])
plt.axhline(0.6, color="gray", linestyle="--", linewidth=1)
plt.title("Per-Class Accuracy")
plt.ylim(0, 1)
for bar, acc in zip(bars, per_class_acc):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.02, f"{acc:.2f}", ha="center")
plt.tight_layout(); plt.show()

print("\n✅ Done!")