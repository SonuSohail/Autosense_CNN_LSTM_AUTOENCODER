import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
import json
import sys

def evaluate():
    gt = pd.read_csv("../data/ground_truth.csv")
    before = pd.read_csv("../data/before_predictions.csv")
    after = pd.read_csv("../data/after_predictions.csv")

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

    metrics = {
        "before": {
            "accuracy": round(accuracy_score(y_true, y_before), 3),
            "precision": round(precision_score(y_true, y_before, average="macro", zero_division=0), 3),
            "recall": round(recall_score(y_true, y_before, average="macro", zero_division=0), 3)
        },
        "after": {
            "accuracy": round(accuracy_score(y_true, y_after), 3),
            "precision": round(precision_score(y_true, y_after, average="macro", zero_division=0), 3),
            "recall": round(recall_score(y_true, y_after, average="macro", zero_division=0), 3)
        }
    }

    return metrics

if __name__ == "__main__":
    try:
        metrics = evaluate()
        print(json.dumps(metrics))   # 🔥 ONLY output
    except Exception as e:
        # Print JSON error so backend never breaks
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
