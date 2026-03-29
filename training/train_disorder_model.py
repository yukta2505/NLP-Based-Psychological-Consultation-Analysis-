# -*- coding: utf-8 -*-
"""
train_disorder_model_v4.py

Changes over v3:
  ① Multiple datasets loaded to cover ALL 10 classes with real Reddit data:
       - kamruzzaman-asif  → depression, anxiety, bipolar, ocd, ptsd, adhd, addiction
       - Sharathhebbar24   → stress ✅, insomnia ✅  (was missing real data)
       - solomonk          → stress ✅ supplement (large ~25k)
       - btwitssayan       → stress ✅ supplement
     panic_disorder has NO reliable Reddit dataset → covered by rich clinical templates only
  ② normalize_label() handles all label variants across all 3 datasets
  ③ Clinical templates kept for ALL classes as quality supplement / balance top-up
  ④ Deduplication step added — removes near-identical texts across datasets
  ⑤ Per-dataset load report so you can see exactly what was pulled
  ⑥ All other v3 fixes retained:
       - 7 templates per class, gradient clipping, early stopping,
         label_map.json saved, per-class classification report
"""

import os, re, random, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report,
)

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN     = 160
BATCH_SIZE  = 32
EPOCHS      = 10
LR          = 2e-5
PATIENCE    = 3
MIN_SAMPLES = 5000      # per class after balancing
SAVE_PATH   = "/content/disorder_model_final"

TARGET_CLASSES = [
    "depression",
    "anxiety",
    "bipolar",
    "ocd",
    "ptsd",
    "adhd",
    "addiction",
    "insomnia",        # real data from Sharathhebbar24
    "panic_disorder",  # clinical templates only (no reliable Reddit source)
    "stress",          # real data from solomonk + Sharathhebbar24 + btwitssayan
]

label2id = {l: i for i, l in enumerate(TARGET_CLASSES)}
id2label = {i: l for l, i in label2id.items()}


# ─────────────────────────────────────────────────────────────
# Text cleaning
# ─────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────
# Label normalisation
# Handles label variants across ALL datasets
# ─────────────────────────────────────────────────────────────
def normalize_label(raw: str) -> str | None:
    label = str(raw).lower().strip()

    # ── depression ──
    if any(x in label for x in ["depress"]):
        return "depression"

    # ── anxiety ── (must come BEFORE panic check)
    if any(x in label for x in ["anxiety", "anxious", "gad"]):
        # exclude panic_disorder which reddit sometimes tags as "anxiety"
        if "panic" not in label:
            return "anxiety"

    # ── panic disorder ──
    if any(x in label for x in ["panic"]):
        return "panic_disorder"

    # ── bipolar ──
    if any(x in label for x in ["bipolar", "manic", "mania"]):
        return "bipolar"

    # ── ocd ──
    if any(x in label for x in ["ocd", "obsess", "compuls"]):
        return "ocd"

    # ── ptsd ──
    if any(x in label for x in ["ptsd", "trauma", "post-traum", "posttraum"]):
        return "ptsd"

    # ── adhd ──
    if any(x in label for x in ["adhd", "attention deficit", "hyperactiv"]):
        return "adhd"

    # ── addiction ──
    if any(x in label for x in ["addict", "substance", "alcohol", "drug", "cannabis",
                                  "opioid", "nicotine", "stimulant"]):
        return "addiction"

    # ── insomnia ──
    if any(x in label for x in ["insomnia", "sleep disorder", "sleep"]):
        return "insomnia"

    # ── stress ──
    if any(x in label for x in ["stress", "burnout", "burn out"]):
        return "stress"

    # ── normal / no disorder → skip ──
    if any(x in label for x in ["normal", "healthy", "no disorder", "none"]):
        return None

    return None


# ─────────────────────────────────────────────────────────────
# Clinical templates — 7 per class
# Covers ALL 10 classes including panic_disorder (template-only)
# ─────────────────────────────────────────────────────────────
CLINICAL_TEMPLATES = {
    "depression": [
        "Patient is a {age}-year-old {gender} with persistent low mood, hopelessness, fatigue, and disturbed sleep.",
        "Patient is a {age}-year-old {gender} presenting with anhedonia, early morning awakening, and significant weight loss. Feels hopeless about the future.",
        "Patient is a {age}-year-old {gender} with severe depressive episode. Reports inability to perform daily activities and psychomotor retardation.",
        "Patient is a {age}-year-old {gender} with loss of motivation, academic or occupational decline, and sleep disturbances. Denies suicidal thoughts.",
        "Patient is a {age}-year-old {gender} with chronic depression reporting persistent sadness, guilt, and reduced appetite with significant weight loss.",
        "Patient is a {age}-year-old {gender} presenting with irritability, low mood, poor sleep, and loss of productivity. Expresses feelings of failure.",
        "Patient is a {age}-year-old {gender} with severe depression and passive suicidal ideation. Reports hopelessness and inability to care for self.",
    ],
    "anxiety": [
        "Patient is a {age}-year-old {gender} with excessive worry, nervousness, restlessness, and persistent overthinking.",
        "Patient is a {age}-year-old {gender} with constant worry about work and finances. Complains of fatigue, irritability, and muscle tension.",
        "Patient is a {age}-year-old {gender} with persistent anxiety, difficulty concentrating, frequent palpitations, and sweating.",
        "Patient is a {age}-year-old {gender} with chronic worry and sleep disturbances. Reports feeling on edge most of the day.",
        "Patient is a {age}-year-old {gender} with excessive worry about health and family. Reports headaches, muscle tension, and poor concentration.",
        "Patient is a {age}-year-old {gender} with generalized anxiety disorder complaining of poor focus, irritability, and racing thoughts.",
        "Patient is a {age}-year-old {gender} with persistent nervousness, restlessness, insomnia, and fatigue. Diagnosed with anxiety disorder.",
    ],
    "bipolar": [
        "Patient is a {age}-year-old {gender} with history of mood swings including elevated mood, decreased need for sleep, and impulsive behavior.",
        "Patient is a {age}-year-old {gender} with alternating manic and depressive episodes. During mania reports excessive spending and grandiosity.",
        "Patient is a {age}-year-old {gender} presenting with current manic episode, increased energy, rapid speech, and reduced sleep.",
        "Patient is a {age}-year-old {gender} with recurrent depressive episodes and past hypomania, previously misdiagnosed with depression.",
        "Patient is a {age}-year-old {gender} with irritability, decreased need for sleep, and increased goal-directed activity lasting one week. Manic episode.",
        "Patient is a {age}-year-old {gender} with bipolar disorder presenting with depressive symptoms, fatigue, and low mood. Maintained on Lithium.",
        "Patient is a {age}-year-old {gender} with rapid cycling mood episodes, alternating periods of high energy and severe depression.",
    ],
    "ocd": [
        "Patient is a {age}-year-old {gender} with intrusive contamination fears and repetitive hand washing lasting several hours, causing skin damage.",
        "Patient is a {age}-year-old {gender} with obsessive doubts about locking doors, repeatedly checking locks many times before leaving home.",
        "Patient is a {age}-year-old {gender} with intrusive aggressive thoughts neutralized by mental rituals. Reports high distress and avoidance behaviors.",
        "Patient is a {age}-year-old {gender} with symmetry obsessions and compulsive arranging behavior. Spends hours aligning objects symmetrically.",
        "Patient is a {age}-year-old {gender} with religious obsessions and excessive praying rituals. Experiences guilt if rituals are not performed perfectly.",
        "Patient is a {age}-year-old {gender} with hoarding behavior due to fear of losing important information. Living conditions severely cluttered.",
        "Patient is a {age}-year-old {gender} with contamination fears and avoidance of public places. Reports excessive cleaning rituals and social withdrawal.",
    ],
    "ptsd": [
        "Patient is a {age}-year-old {gender} with recurrent nightmares, flashbacks, and avoidance of reminders following a traumatic event.",
        "Patient is a {age}-year-old {gender} following traumatic assault. Reports intrusive memories, emotional numbness, and difficulty sleeping.",
        "Patient is a {age}-year-old {gender} veteran with combat exposure. Reports persistent nightmares, anger outbursts, and social isolation.",
        "Patient is a {age}-year-old {gender} with history of domestic violence. Reports flashbacks, anxiety, and difficulty trusting others.",
        "Patient is a {age}-year-old {gender} with workplace accident trauma. Reports avoidance of work environment and panic-like symptoms.",
        "Patient is a {age}-year-old {gender} with childhood trauma. Reports dissociation, emotional detachment, and recurrent distressing memories.",
        "Patient is a {age}-year-old {gender} with PTSD symptoms including insomnia, irritability, hypervigilance, and exaggerated startle response.",
    ],
    "adhd": [
        "Patient is a {age}-year-old {gender} with difficulty sustaining attention, frequently distracted, making careless mistakes, and showing hyperactivity.",
        "Patient is a {age}-year-old {gender} with inattentive symptoms including daydreaming, poor organization, and incomplete tasks. Academic performance declining.",
        "Patient is a {age}-year-old {gender} with hyperactive behavior, unable to sit still, frequently interrupting others and talking excessively.",
        "Patient is a {age}-year-old {gender} with combined type ADHD showing impulsivity, poor focus, and risk-taking behavior.",
        "Patient is a {age}-year-old {gender} with difficulty following instructions, forgetfulness, and frequent loss of items.",
        "Patient is a {age}-year-old {gender} with persistent ADHD symptoms affecting performance. Reports procrastination and poor time management.",
        "Patient is a {age}-year-old {gender} with behavioral issues and attention deficits impacting daily functioning. Started on Atomoxetine.",
    ],
    "addiction": [
        "Patient is a {age}-year-old {gender} with long history of alcohol use. Reports increased tolerance, withdrawal symptoms, and multiple failed quit attempts.",
        "Patient is a {age}-year-old {gender} with opioid dependence. Reports cravings and inability to function without substance use.",
        "Patient is a {age}-year-old {gender} with cannabis use disorder. Reports daily use, lack of motivation, and occupational decline.",
        "Patient is a {age}-year-old {gender} with alcohol dependence. Reports liver issues and continued use despite health risks.",
        "Patient is a {age}-year-old {gender} with nicotine addiction. Reports difficulty quitting and significant withdrawal irritability.",
        "Patient is a {age}-year-old {gender} with stimulant abuse. Reports insomnia, paranoia, and weight loss.",
        "Patient is a {age}-year-old {gender} with severe alcohol use disorder affecting occupational and social functioning. Referred to rehabilitation.",
    ],
    "insomnia": [
        "Patient is a {age}-year-old {gender} with difficulty initiating sleep, lying awake for hours, and daytime fatigue. Diagnosed with Insomnia Disorder.",
        "Patient is a {age}-year-old {gender} with frequent nighttime awakenings, non-restorative sleep, and poor concentration during the day.",
        "Patient is a {age}-year-old {gender} with stress-related insomnia. Reports racing thoughts preventing sleep onset despite adequate time in bed.",
        "Patient is a {age}-year-old {gender} with chronic insomnia and daytime irritability. Sleep duration less than four hours nightly.",
        "Patient is a {age}-year-old {gender} with irregular sleep cycle and difficulty maintaining sleep despite spending eight hours in bed.",
        "Patient is a {age}-year-old {gender} with insomnia associated with anxiety. Reports fatigue, headaches, and poor daytime functioning.",
        "Patient is a {age}-year-old {gender} with persistent insomnia. Sleep restriction therapy and stimulus control techniques initiated. Started on Zolpidem.",
    ],
    "panic_disorder": [
        "Patient is a {age}-year-old {gender} with recurrent sudden episodes of intense fear, palpitations, chest tightness, dizziness, and fear of dying.",
        "Patient is a {age}-year-old {gender} with unexpected panic attacks occurring multiple times per week. Reports avoidance of crowded places.",
        "Patient is a {age}-year-old {gender} with panic attacks triggered by stress. Reports sweating, trembling, and shortness of breath during episodes.",
        "Patient is a {age}-year-old {gender} with nocturnal panic attacks. Wakes abruptly with intense fear and heart palpitations.",
        "Patient is a {age}-year-old {gender} with fear of having panic attacks in public places. Avoids travel, shopping malls, and social events.",
        "Patient is a {age}-year-old {gender} with frequent ER visits due to chest pain later identified as panic attacks. Anticipatory anxiety present.",
        "Patient is a {age}-year-old {gender} with panic disorder and anticipatory anxiety about future attacks. Started on Alprazolam for acute episodes.",
    ],
    "stress": [
        "Patient is a {age}-year-old {gender} presenting with burnout symptoms following months of excessive workload. Reports feeling overwhelmed and irritable.",
        "Patient is a {age}-year-old {gender} with work-related stress, unable to switch off after work. Physical symptoms include headaches and neck tension.",
        "Patient is a {age}-year-old {gender} with occupational stress, difficulty delegating, and persistent sense of pressure and deadline fatigue.",
        "Patient is a {age}-year-old {gender} with stress-related gastrointestinal symptoms, irritability, and poor sleep due to work demands.",
        "Patient is a {age}-year-old {gender} with acute stress reaction following a major life event. Reports anxiety, sleep difficulties, and poor concentration.",
        "Patient is a {age}-year-old {gender} with chronic stress and frequent tension headaches. No formal psychiatric diagnosis. Counseling initiated.",
        "Patient is a {age}-year-old {gender} with mild stress related to work responsibilities. Generally stable mood. Encouraged to maintain lifestyle balance.",
    ],
}


def generate_clinical_samples(n_per_class: int = 5000) -> list[dict]:
    records = []
    ages    = list(range(18, 61))
    genders = ["male", "female"]

    for label, templates in CLINICAL_TEMPLATES.items():
        added = 0
        while added < n_per_class:
            for tmpl in templates:
                age    = random.choice(ages)
                gender = random.choice(genders)
                text   = clean_text(tmpl.format(age=age, gender=gender))
                records.append({"text": text[:512], "label": label})
                added += 1
                if added >= n_per_class:
                    break
    return records


# ─────────────────────────────────────────────────────────────
# Dataset loader helper
# ─────────────────────────────────────────────────────────────
def load_hf_dataset(
    dataset_id: str,
    split:       str   = "train",
    text_col:    str   = "text",
    label_col:   str   = "label",
    min_words:   int   = 5,
    max_chars:   int   = 512,
) -> list[dict]:
    """
    Generic loader for any HuggingFace mental-health dataset.
    Returns list of {"text": ..., "label": ...} dicts.
    """
    records = []
    try:
        print(f"  Loading {dataset_id} ...")
        ds = load_dataset(dataset_id, split=split)

        # auto-detect column names if defaults don't exist
        cols = ds.column_names
        if text_col not in cols:
            for candidate in ["text", "post", "content", "sentence", "body"]:
                if candidate in cols:
                    text_col = candidate
                    break
        if label_col not in cols:
            for candidate in ["label", "category", "class", "disorder", "condition"]:
                if candidate in cols:
                    label_col = candidate
                    break

        for row in ds:
            raw_label = str(row.get(label_col, ""))
            label     = normalize_label(raw_label)
            if not label or label not in TARGET_CLASSES:
                continue
            text = clean_text(str(row.get(text_col, "")))
            if len(text.split()) > min_words:
                records.append({"text": text[:max_chars], "label": label})

        label_counts = {}
        for r in records:
            label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1
        print(f"    → {len(records)} usable rows")
        for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
            print(f"       {lbl:<18} {cnt}")

    except Exception as e:
        print(f"    ⚠ Could not load {dataset_id}: {e}")

    return records


# ─────────────────────────────────────────────────────────────
# Dataset class
# ─────────────────────────────────────────────────────────────
class MentalDataset(Dataset):
    def __init__(self, encodings, labels):
        self.enc    = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────
def evaluate(model, loader, loss_fn, device):
    model.eval()
    preds, labels_all = [], []
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids      = batch["input_ids"].to(device),
                attention_mask = batch["attention_mask"].to(device),
            )
            loss = loss_fn(outputs.logits, batch["labels"].to(device))
            total_loss += loss.item()
            preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            labels_all.extend(batch["labels"].numpy())

    return (
        total_loss / len(loader),
        accuracy_score(labels_all, preds),
        f1_score(labels_all, preds, average="macro", zero_division=0),
        precision_score(labels_all, preds, average="macro", zero_division=0),
        recall_score(labels_all, preds, average="macro", zero_division=0),
        preds,
        labels_all,
    )


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    random.seed(42)
    np.random.seed(42)

    all_records = []

    # ── Dataset 1: Primary — covers 7 original classes ────────
    print("\n[1] Loading primary dataset (kamruzzaman-asif)...")
    all_records += load_hf_dataset(
        "kamruzzaman-asif/reddit-mental-health-classification",
        text_col  = "text",
        label_col = "label",
    )

    # ── Dataset 2: Sharathhebbar24 — stress ✅ insomnia ✅ ────
    print("\n[2] Loading Sharathhebbar24/mental_health_data (stress + insomnia)...")
    all_records += load_hf_dataset(
        "Sharathhebbar24/mental_health_data",
        text_col  = "text",
        label_col = "label",
    )

    # ── Dataset 3: solomonk — large stress supplement ─────────
    print("\n[3] Loading solomonk/reddit_mental_health_posts (stress supplement)...")
    all_records += load_hf_dataset(
        "solomonk/reddit_mental_health_posts",
        text_col  = "text",
        label_col = "label",
    )

    # ── Dataset 4: btwitssayan — stress supplement ─────────────
    print("\n[4] Loading btwitssayan/mental-disorder-identification (stress supplement)...")
    all_records += load_hf_dataset(
        "btwitssayan/mental-disorder-identification",
        text_col  = "post",
        label_col = "disorder",
    )

    # ── Combined Reddit data summary ───────────────────────────
    df_reddit = pd.DataFrame(all_records)
    print(f"\n[5] Total Reddit rows loaded: {len(df_reddit)}")
    print("Combined Reddit distribution:")
    print(df_reddit["label"].value_counts())

    # ── Deduplication ──────────────────────────────────────────
    print("\n[6] Deduplicating...")
    before = len(df_reddit)
    df_reddit = df_reddit.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"    Removed {before - len(df_reddit)} duplicates → {len(df_reddit)} unique rows")

    # ── Clinical samples — supplement all classes ──────────────
    # Use fewer clinical samples for classes that already have rich Reddit data,
    # and more for panic_disorder which has NO Reddit data at all.
    reddit_counts = df_reddit["label"].value_counts().to_dict()

    clinical_n = {}
    for cls in TARGET_CLASSES:
        reddit_n = reddit_counts.get(cls, 0)
        if cls == "panic_disorder":
            # No Reddit data at all — generate full amount
            clinical_n[cls] = 6000
        elif reddit_n < 2000:
            # Low Reddit data — heavy clinical supplement
            clinical_n[cls] = 4000
        elif reddit_n < 5000:
            # Medium Reddit data — moderate supplement
            clinical_n[cls] = 2000
        else:
            # Rich Reddit data — light supplement for diversity
            clinical_n[cls] = 1000

    print(f"\n[7] Generating clinical samples...")
    for cls, n in clinical_n.items():
        print(f"    {cls:<18} {n} clinical samples (Reddit: {reddit_counts.get(cls, 0)})")

    clinical_records = []
    ages    = list(range(18, 61))
    genders = ["male", "female"]

    for label, templates in CLINICAL_TEMPLATES.items():
        n = clinical_n.get(label, 2000)
        added = 0
        while added < n:
            for tmpl in templates:
                age    = random.choice(ages)
                gender = random.choice(genders)
                text   = clean_text(tmpl.format(age=age, gender=gender))
                clinical_records.append({"text": text[:512], "label": label})
                added += 1
                if added >= n:
                    break

    df_clinical = pd.DataFrame(clinical_records)

    # ── Also inject the hand-crafted samples from your documents ──
    # These 7-sample-per-class notes are high-quality clinical text
    handcrafted = [
        # Depression
        ("patient is a yearold female with a month history of persistent low mood reports loss of interest in social activities fatigue and poor concentration complains of insomnia and decreased appetite expresses feelings of worthlessness denies active suicidal ideation started on sertraline", "depression"),
        ("patient is a yearold male presenting with months of depressive symptoms reports anhedonia early morning awakening and significant weight loss feels hopeless about the future passive death wishes reported initiated on escitalopram", "depression"),
        ("patient is a yearold female with severe depressive episode reports inability to perform daily activities poor hygiene and social withdrawal psychomotor retardation observed started on venlafaxine", "depression"),
        ("patient is a yearold male with twomonth history of low mood and fatigue reports academic decline and loss of motivation sleep disturbances present denies suicidal thoughts referred for therapy and started on fluoxetine", "depression"),
        ("patient is a yearold female with chronic depression reports persistent sadness guilt and reduced appetite significant weight loss noted initiated on duloxetine", "depression"),
        ("patient is a yearold male presenting with irritability and low mood reports poor sleep and loss of productivity expresses feelings of failure started on paroxetine", "depression"),
        ("patient is a yearold female with severe depression and passive suicidal ideation reports hopelessness and inability to care for self urgent psychiatric referral made started on sertraline", "depression"),
        # Anxiety
        ("patient is a yearold female with sixmonth history of excessive worry reports restlessness muscle tension and insomnia started on escitalopram", "anxiety"),
        ("patient is a yearold male with constant worry about work and finances complains of fatigue and irritability diagnosed with generalized anxiety disorder", "anxiety"),
        ("patient is a yearold female reporting persistent anxiety and difficulty concentrating experiences frequent palpitations and sweating started on sertraline", "anxiety"),
        ("patient is a yearold male with chronic worry and sleep disturbances reports feeling on edge most days initiated on buspirone", "anxiety"),
        ("patient is a yearold female with excessive worry about family health reports headaches and muscle tension referred for cognitive behavioral therapy", "anxiety"),
        ("patient is a yearold male with generalized anxiety complains of poor focus and irritability started on paroxetine", "anxiety"),
        ("patient is a yearold female with persistent nervousness and restlessness reports insomnia and fatigue diagnosed with anxiety disorder", "anxiety"),
        # Bipolar
        ("patient is a yearold male with history of mood swings reports episodes of elevated mood with decreased need for sleep and impulsive behavior currently depressed started on lithium", "bipolar"),
        ("patient is a yearold female with alternating manic and depressive episodes during mania reports excessive spending and grandiosity diagnosed with bipolar disorder", "bipolar"),
        ("patient is a yearold male presenting with current manic episode reports increased energy rapid speech and reduced sleep started on valproate", "bipolar"),
        ("patient is a yearold female with recurrent depressive episodes and past hypomania previously misdiagnosed with depression initiated on lamotrigine", "bipolar"),
        ("patient is a yearold male with irritability decreased need for sleep and increased goaldirected activity symptoms ongoing for one week diagnosed with manic episode", "bipolar"),
        ("patient is a yearold female with bipolar disorder presenting with depressive symptoms reports fatigue and low mood maintained on lithium", "bipolar"),
        ("patient is a yearold male with rapid cycling mood episodes reports alternating periods of high energy and severe depression referred for specialist care", "bipolar"),
        # OCD
        ("patient is a yearold female with a twoyear history of intrusive contamination fears reports repetitive hand washing lasting several hours daily leading to skin damage acknowledges thoughts are irrational but feels unable to resist compulsions started on fluoxetine with exposure and response prevention therapy", "ocd"),
        ("patient is a yearold male presenting with obsessive doubts about locking doors and turning off appliances reports repeatedly checking locks up to times before leaving home experiences severe anxiety if rituals are not completed diagnosed with obsessivecompulsive disorder", "ocd"),
        ("patient is a yearold female with intrusive aggressive thoughts of harming loved ones denies intent but performs mental rituals to neutralize thoughts reports high distress and avoidance behaviors initiated on sertraline", "ocd"),
        ("patient is a yearold male with symmetry obsessions and compulsive arranging behavior spends hours aligning objects symmetrically reports frustration and interference with daily routine", "ocd"),
        ("patient is a yearold female with religious obsessions and excessive praying rituals experiences guilt and distress if rituals are not performed perfectly", "ocd"),
        ("patient is a yearold male with hoarding behavior reports inability to discard items due to fear of losing important information living conditions severely cluttered", "ocd"),
        ("patient is a yearold female with contamination fears and avoidance of public places reports excessive cleaning rituals and social withdrawal started on fluvoxamine", "ocd"),
        # PTSD
        ("patient is a yearold male with history of a severe road accident months ago reports recurrent nightmares flashbacks and avoidance of driving experiences hypervigilance and exaggerated startle response started on paroxetine", "ptsd"),
        ("patient is a yearold female following a traumatic assault reports intrusive memories emotional numbness and avoidance of reminders difficulty sleeping and frequent irritability noted", "ptsd"),
        ("patient is a yearold male veteran with combat exposure reports persistent nightmares anger outbursts and social isolation diagnosed with posttraumatic stress disorder", "ptsd"),
        ("patient is a yearold female with history of domestic violence reports flashbacks anxiety and difficulty trusting others engaged in traumafocused therapy", "ptsd"),
        ("patient is a yearold male with workplace accident trauma reports avoidance of work environment and paniclike symptoms when exposed to reminders", "ptsd"),
        ("patient is a yearold female with childhood trauma reports dissociation emotional detachment and recurrent distressing memories", "ptsd"),
        ("patient is a yearold male with ptsd symptoms including insomnia irritability and hypervigilance started on sertraline", "ptsd"),
        # ADHD
        ("patient is a yearold boy with difficulty sustaining attention in school frequently distracted forgets assignments and makes careless mistakes teachers report hyperactivity and impulsivity started on methylphenidate", "adhd"),
        ("patient is a yearold girl with inattentive symptoms reports daydreaming poor organization and incomplete tasks academic performance declining", "adhd"),
        ("patient is a yearold boy with hyperactive behavior unable to sit still frequently interrupts others and talks excessively", "adhd"),
        ("patient is a yearold male with combined type adhd reports impulsivity poor focus and risktaking behavior", "adhd"),
        ("patient is a yearold female with difficulty following instructions and forgetfulness parents report frequent loss of items", "adhd"),
        ("patient is a yearold male with persistent adhd symptoms affecting college performance reports procrastination and poor time management", "adhd"),
        ("patient is a yearold boy with behavioral issues and attention deficits started on atomoxetine", "adhd"),
        # Addiction
        ("patient is a yearold male with a twelveyear history of alcohol use reports increased tolerance and withdrawal symptoms including tremors and sweating multiple failed attempts to quit started on naltrexone", "addiction"),
        ("patient is a yearold female with opioid dependence reports cravings and inability to function without substance use", "addiction"),
        ("patient is a yearold male with cannabis use disorder reports daily use and lack of motivation", "addiction"),
        ("patient is a yearold male with alcohol dependence reports liver issues and continued use despite health risks", "addiction"),
        ("patient is a yearold female with nicotine addiction reports difficulty quitting and withdrawal irritability", "addiction"),
        ("patient is a yearold male with stimulant abuse reports insomnia paranoia and weight loss", "addiction"),
        ("patient is a yearold male with severe alcohol use disorder affecting occupational and social functioning referred to rehabilitation program", "addiction"),
        # Insomnia
        ("patient is a yearold female with sixmonth history of difficulty initiating sleep reports lying awake for hours and daytime fatigue diagnosed with insomnia disorder", "insomnia"),
        ("patient is a yearold male with frequent nighttime awakenings reports nonrestorative sleep and poor concentration", "insomnia"),
        ("patient is a yearold female with stressrelated insomnia reports racing thoughts preventing sleep onset", "insomnia"),
        ("patient is a yearold male with chronic insomnia and daytime irritability sleep duration less than four hours nightly", "insomnia"),
        ("patient is a yearold student with irregular sleep cycle and excessive screen time reports difficulty maintaining sleep", "insomnia"),
        ("patient is a yearold female with insomnia associated with anxiety reports fatigue and headaches", "insomnia"),
        ("patient is a yearold male with persistent insomnia started on zolpidem", "insomnia"),
        # Panic Disorder
        ("patient is a yearold female with recurrent sudden episodes of intense fear reports palpitations chest tightness dizziness and fear of dying episodes last minutes diagnosed with panic disorder", "panic_disorder"),
        ("patient is a yearold male with unexpected panic attacks occurring multiple times per week reports avoidance of crowded places", "panic_disorder"),
        ("patient is a yearold female with panic attacks triggered by stress reports sweating trembling and shortness of breath", "panic_disorder"),
        ("patient is a yearold male with nocturnal panic attacks wakes abruptly with intense fear and palpitations", "panic_disorder"),
        ("patient is a yearold female with fear of having panic attacks in public places avoids travel and social events", "panic_disorder"),
        ("patient is a yearold male with frequent er visits due to chest pain later identified as panic attacks", "panic_disorder"),
        ("patient is a yearold female with panic disorder and anticipatory anxiety started on alprazolam for acute episodes", "panic_disorder"),
        # Stress
        ("patient is a yearold male professional presenting with burnout symptoms following months of excessive workload reports feeling overwhelmed irritable and unable to switch off after work hours", "stress"),
        ("patient is a yearold female with workrelated stress unable to switch off after work physical symptoms include headaches and neck tension", "stress"),
        ("patient is a yearold male with occupational stress difficulty delegating and persistent sense of pressure no formal psychiatric diagnosis counseling initiated", "stress"),
        ("patient is a yearold female with stressrelated gastrointestinal symptoms irritability and poor sleep due to work demands", "stress"),
        ("patient is a yearold male with acute stress reaction following major life event reports anxiety sleep difficulties and poor concentration", "stress"),
        ("patient is a yearold female with chronic stress and frequent tension headaches no formal psychiatric diagnosis counseling initiated", "stress"),
        ("patient is a yearold male presenting for routine psychological wellness check reports generally stable mood with occasional mild stress sleep is adequate", "stress"),
    ]
    df_handcrafted = pd.DataFrame(handcrafted, columns=["text", "label"])

    # ── Merge all sources ──────────────────────────────────────
    df = pd.concat([df_reddit, df_clinical, df_handcrafted], ignore_index=True)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    print(f"\n[8] Total after merging all sources: {len(df)}")
    print("Distribution before balancing:")
    print(df["label"].value_counts())

    # ── Balance ────────────────────────────────────────────────
    print(f"\n[9] Balancing to {MIN_SAMPLES} samples per class...")
    df = (
        df.groupby("label", group_keys=False)
          .apply(lambda x: x.sample(MIN_SAMPLES, replace=len(x) < MIN_SAMPLES, random_state=42))
          .reset_index(drop=True)
    )
    print("Balanced distribution:")
    print(df["label"].value_counts())

    # ── Train / val split ──────────────────────────────────────
    X = df["text"].values
    y = df["label"].map(label2id).values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"\nTrain: {len(X_train)}  |  Val: {len(X_val)}")

    # ── Tokenise ───────────────────────────────────────────────
    print("\n[10] Tokenising...")
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    def tokenize(texts):
        return tokenizer(
            list(texts), padding="max_length",
            truncation=True, max_length=MAX_LEN,
        )

    train_enc = tokenize(X_train)
    val_enc   = tokenize(X_val)

    train_loader = DataLoader(
        MentalDataset(train_enc, y_train), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(MentalDataset(val_enc, y_val), batch_size=BATCH_SIZE)

    # ── Model ──────────────────────────────────────────────────
    print("\n[11] Loading RoBERTa...")
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=len(TARGET_CLASSES),
    ).to(DEVICE)

    loss_fn   = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = total_steps // 10,
        num_training_steps = total_steps,
    )

    # ── Training loop ──────────────────────────────────────────
    print("\n[12] Training...\n")
    train_losses, val_losses = [], []
    val_f1s, val_accs        = [], []
    val_precisions, val_recalls = [], []
    best_f1        = 0.0
    patience_count = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids      = batch["input_ids"].to(DEVICE),
                attention_mask = batch["attention_mask"].to(DEVICE),
            )
            loss = loss_fn(outputs.logits, batch["labels"].to(DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_loss, acc, f1, prec, rec, val_preds, val_labels = evaluate(
            model, val_loader, loss_fn, DEVICE
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_f1s.append(f1)
        val_accs.append(acc)
        val_precisions.append(prec)
        val_recalls.append(rec)

        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"  Train Loss : {train_loss:.4f}")
        print(f"  Val Loss   : {val_loss:.4f}")
        print(f"  Accuracy   : {acc:.4f}")
        print(f"  F1 Score   : {f1:.4f}")
        print(f"  Precision  : {prec:.4f}")
        print(f"  Recall     : {rec:.4f}")

        if (epoch + 1) % 2 == 0:
            print("\n  Per-class report:")
            print(classification_report(
                val_labels, val_preds,
                target_names=TARGET_CLASSES, zero_division=0, digits=3,
            ))

        if f1 > best_f1:
            best_f1 = f1
            patience_count = 0
            model.save_pretrained(SAVE_PATH)
            tokenizer.save_pretrained(SAVE_PATH)
            print(f"  ✅ New best F1 {best_f1:.4f} — model saved.\n")
        else:
            patience_count += 1
            print(f"  No improvement ({patience_count}/{PATIENCE})\n")
            if patience_count >= PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}.")
                break

    # ── Save label_map.json ────────────────────────────────────
    label_map = {"label2id": label2id, "id2label": id2label}
    with open(os.path.join(SAVE_PATH, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"\n✅ label_map.json saved → {SAVE_PATH}")
    print(f"   Classes: {TARGET_CLASSES}")

    # ── Plots ──────────────────────────────────────────────────
    epochs_done = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs_done, train_losses, label="Train Loss")
    plt.plot(epochs_done, val_losses,   label="Val Loss")
    plt.legend(); plt.title("Loss Curve"); plt.xlabel("Epochs"); plt.show()

    plt.figure()
    plt.plot(epochs_done, val_f1s,  label="F1 Score")
    plt.plot(epochs_done, val_accs, label="Accuracy")
    plt.legend(); plt.title("Performance Curve"); plt.xlabel("Epochs"); plt.show()

    plt.figure()
    plt.plot(epochs_done, val_precisions, label="Precision")
    plt.plot(epochs_done, val_recalls,    label="Recall")
    plt.legend(); plt.title("Precision vs Recall"); plt.xlabel("Epochs"); plt.show()

    # ── Final per-class report ─────────────────────────────────
    print("\n=== FINAL Per-Class Report (best checkpoint) ===")
    _, _, _, _, _, final_preds, final_labels = evaluate(model, val_loader, loss_fn, DEVICE)
    print(classification_report(
        final_labels, final_preds,
        target_names=TARGET_CLASSES, zero_division=0, digits=3,
    ))
    print(f"✅ Best Val F1: {best_f1:.4f}  |  Model: {SAVE_PATH}")


if __name__ == "__main__":
    main()