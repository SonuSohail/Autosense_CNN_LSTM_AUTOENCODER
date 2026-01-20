import argparse
import json
import sys
import os
from datetime import datetime

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score


def evaluate(selected_model="default"):
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    gt_path = os.path.join(data_dir, "ground_truth.csv")
    before_path = os.path.join(data_dir, "before_predictions.csv")
    after_path_specific = os.path.join(data_dir, f"after_predictions_{selected_model}.csv")
    after_path_generic = os.path.join(data_dir, "after_predictions.csv")

    # Choose model-specific after file if exists, else fallback
    after_path = after_path_specific if os.path.exists(after_path_specific) else after_path_generic

    if not all([os.path.exists(p) for p in [gt_path, before_path, after_path]]):
        raise FileNotFoundError("Prediction or ground truth files missing")

    gt = pd.read_csv(gt_path)
    before = pd.read_csv(before_path)
    after = pd.read_csv(after_path)

    if gt.empty or before.empty or after.empty:
        raise ValueError("Prediction or ground truth files are empty")

    gt = gt.sort_values("image").reset_index(drop=True)
    before = before.sort_values("image").reset_index(drop=True)
    after = after.sort_values("image").reset_index(drop=True)

    y_true = gt["label"]
    y_before = before["predicted"]
    y_after = after["predicted"]

    min_len = min(len(y_true), len(y_before), len(y_after))
    y_true = y_true[:min_len]
    y_before = y_before[:min_len]
    y_after = y_after[:min_len]

    before_acc = accuracy_score(y_true, y_before)
    after_acc = accuracy_score(y_true, y_after)

    metrics = {
        "model": selected_model,
        "timestamp": datetime.now().isoformat(),
        "before": {
            "accuracy": round(before_acc, 3),
            "precision": round(precision_score(y_true, y_before, average="macro", zero_division=0), 3),
            "recall": round(recall_score(y_true, y_before, average="macro", zero_division=0), 3)
        },
        "after": {
            "accuracy": round(after_acc, 3),
            "precision": round(precision_score(y_true, y_after, average="macro", zero_division=0), 3),
            "recall": round(recall_score(y_true, y_after, average="macro", zero_division=0), 3)
        },
        "delta": {
            "accuracy": round(after_acc - before_acc, 3),
            "precision": round(
                precision_score(y_true, y_after, average="macro", zero_division=0) -
                precision_score(y_true, y_before, average="macro", zero_division=0), 3),
            "recall": round(
                recall_score(y_true, y_after, average="macro", zero_division=0) -
                recall_score(y_true, y_before, average="macro", zero_division=0), 3),
        }
    }

    # Write model-specific and latest cache
    with open(os.path.join(data_dir, f"metrics_{selected_model}.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(data_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", type=str, default="default")
        args = parser.parse_args()

        metrics = evaluate(selected_model=args.model)
        print(json.dumps(metrics))   # 🔥 ONLY output
    except Exception as e:
        # Print JSON error so backend never breaks
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
