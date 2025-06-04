from __future__ import annotations

import csv
import random
import re
import string
from pathlib import Path

import numpy as np

SEED = 42

LABEL_MAP = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

def preprocess_text(text: str) -> str:
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def tokenize(text: str) -> list[str]:
    return text.split()

def load_dataset(path: str) -> list[tuple[str, int]]:
    data = []
    with Path.open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:  # noqa: PLR2004
                continue
            label, text = preprocess_text(row[0]), preprocess_text(row[1])
            if label not in LABEL_MAP:
                continue
            data.append((text, LABEL_MAP[label]))
    return data

def split_data(data: list) -> tuple[list, list, list]:
    random.shuffle(data)
    total = len(data)

    train_end = int(total * 0.7)
    val_end = train_end + int(total * 0.2)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data

def wrap_text(text: str, width: int) -> list[str]:
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line) + len(word) + 1 <= width:
            current_line += (" " if current_line else "") + word
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines

def load_glove(filename: str) -> dict[str, np.ndarray]:
    words = {}
    with Path(filename).open(encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            words[parts[0]] = np.array(parts[1:], dtype="float32")
    return words

def print_reports(reports: dict) -> None:
    report_lines = {}
    param_lines = {}
    wrap_width = 60

    for name, content in reports.items():
        if isinstance(content, dict):
            report_lines[name] = content["classification"].splitlines()
            params_str = ", ".join(f"{k}={v}" for k, v in content["params"].items())
            wrapped_params = wrap_text(f"Best Params: {params_str}", wrap_width)
            param_lines[name] = wrapped_params
        else:
            report_lines[name] = content.splitlines()
            param_lines[name] = ["Best Params: N/A"]

    max_lines = max(len(lines) for lines in report_lines.values())
    for lines in report_lines.values():
        lines += [""] * (max_lines - len(lines))  # noqa: PLW2901

    for name in report_lines:
        report_lines[name] += [""] + param_lines[name]

    max_total_lines = max(len(lines) for lines in report_lines.values())
    for lines in report_lines.values():
        lines += [""] * (max_total_lines - len(lines))  # noqa: PLW2901

    dataset_names = list(report_lines.keys())
    header = " | ".join(name.center(wrap_width) for name in dataset_names)
    divider = " | ".join("=" * wrap_width for _ in dataset_names)
    print(header)
    print(divider)

    for i in range(max_total_lines):
        row = " | ".join(report_lines[name][i].center(wrap_width) for name in dataset_names)
        print(row)
