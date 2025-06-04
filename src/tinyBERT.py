import shutil
import time

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import EvalPrediction
from utils import LABEL_MAP

# Constants
MODEL_NAME = "huawei-noah/TinyBERT_General_4L_312D"
DATASETS = [
    ("amazon", "project/src/data/amazon.csv"),
    ("tweets", "project/src/data/tweets.csv"),
    ("youtube", "project/src/data/youtube.csv"),
]

# Function to preprocess data
def preprocess_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)  # noqa: PD901
    df = df.dropna()  # noqa: PD901
    df = df[df["sentiment"].isin(LABEL_MAP.keys())]  # noqa: PD901
    df["label"] = df["sentiment"].map(LABEL_MAP)
    return df

# Metrics function
def compute_metrics(pred: EvalPrediction) -> dict:
    true_labels = pred.label_ids
    predicted_labels = np.argmax(pred.predictions, axis=1)
    precision, recall, f1_score, _ = precision_recall_fscore_support(true_labels, predicted_labels, average="weighted")
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1_score,
    }

# Main loop through datasets
for name, path in DATASETS:
    print(f"\n=== Training on {name} dataset ===")

    df = preprocess_data(path)  # noqa: PD901

    # Splitting data
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)
    val_texts, test_texts, val_labels, test_labels = train_test_split(temp_texts, temp_labels, test_size=0.5, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Tokenize data
    def tokenize_function(texts: pd.Series, tokenizer: PreTrainedTokenizerBase = tokenizer) -> dict:
        return tokenizer(list(texts), truncation=True, padding=True, max_length=128)

    train_encodings = tokenize_function(train_texts)
    val_encodings = tokenize_function(val_texts)
    test_encodings = tokenize_function(test_texts)

    # Convert to HuggingFace Datasets
    train_dataset = Dataset.from_dict({**train_encodings, "label": list(train_labels)})
    val_dataset = Dataset.from_dict({**val_encodings, "label": list(val_labels)})
    test_dataset = Dataset.from_dict({**test_encodings, "label": list(test_labels)})

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    training_args = TrainingArguments(
        output_dir=f"./project/src/tmp/results_{name}",
        do_eval=True,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=False,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time

    # Evaluate
    predictions = trainer.predict(test_dataset)
    metrics = compute_metrics(predictions)

    print(f"Training time: {training_time:.2f} seconds")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")

    # Optionally print classification report
    preds = np.argmax(predictions.predictions, axis=1)
    print("\nClassification Report:")
    print(classification_report(test_labels, preds, target_names=LABEL_MAP.keys()))

    # Delete tmp files
    shutil.rmtree("./project/src/tmp", ignore_errors=True)
