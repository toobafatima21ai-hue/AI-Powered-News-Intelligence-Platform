# train.py

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from sklearn.metrics import (
    accuracy_score,
    f1_score
)

import pandas as pd
import numpy as np

# =====================================================
# LOAD DATASET
# =====================================================

print("Loading dataset...")

train_df = pd.read_csv(
    "train.csv",
    header=None
)

test_df = pd.read_csv(
    "test.csv",
    header=None
)

# =====================================================
# REMOVE HEADER ROW
# =====================================================

train_df = train_df.iloc[1:]
test_df = test_df.iloc[1:]

# =====================================================
# RENAME COLUMNS
# =====================================================

train_df.columns = [
    "label",
    "title",
    "description"
]

test_df.columns = [
    "label",
    "title",
    "description"
]

# =====================================================
# USE SMALLER SUBSET FOR FASTER TRAINING
# =====================================================

train_df = train_df.sample(
    n=10000,
    random_state=42
)

test_df = test_df.sample(
    n=2000,
    random_state=42
)

# =====================================================
# COMBINE TITLE + DESCRIPTION
# =====================================================

train_df["text"] = (
    train_df["title"].astype(str)
    + " "
    + train_df["description"].astype(str)
)

test_df["text"] = (
    test_df["title"].astype(str)
    + " "
    + test_df["description"].astype(str)
)

# =====================================================
# LABELS START FROM 0
# =====================================================

train_df["label"] = (
    train_df["label"].astype(int) - 1
)

test_df["label"] = (
    test_df["label"].astype(int) - 1
)

# =====================================================
# CONVERT TO DATASET
# =====================================================

train_dataset = Dataset.from_pandas(
    train_df[["text", "label"]]
)

test_dataset = Dataset.from_pandas(
    test_df[["text", "label"]]
)

# =====================================================
# TOKENIZER
# =====================================================

model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

# =====================================================
# TOKENIZATION
# =====================================================

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

print("Tokenizing dataset...")

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True
)

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True
)

# =====================================================
# LOAD MODEL
# =====================================================

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=4
)

# =====================================================
# METRICS
# =====================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    f1 = f1_score(
        labels,
        predictions,
        average="weighted"
    )

    return {
        "accuracy": accuracy,
        "f1": f1
    }

# =====================================================
# TRAINING ARGUMENTS
# =====================================================

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_steps=50,
    report_to="none"
)
# =====================================================
# TRAINER
# =====================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# =====================================================
# TRAIN MODEL
# =====================================================

print("Training started...")

trainer.train()

# =====================================================
# EVALUATE
# =====================================================

results = trainer.evaluate()

print("\n==============================")
print("Evaluation Results")
print("==============================")

print(results)

# =====================================================
# SAVE MODEL
# =====================================================

trainer.save_model(
    "bert_news_classifier"
)

tokenizer.save_pretrained(
    "bert_news_classifier"
)

print("\nModel saved successfully!")
print("Folder created: bert_news_classifier")