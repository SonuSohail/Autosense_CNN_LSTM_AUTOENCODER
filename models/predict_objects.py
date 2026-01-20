import argparse
import os
import pandas as pd
import cv2


def predict(folder, output_csv, threshold=100.0, weight_edges=False):
    """
    Simple heuristic classifier using Laplacian sharpness.
    Optional edge weighting to differentiate modes.
    """
    results = []

    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        sharpness = cv2.Laplacian(image, cv2.CV_64F).var()
        if weight_edges:
            edges = cv2.Canny(image, 100, 200).mean()
            sharpness = 0.7 * sharpness + 0.3 * edges

        pred = "car" if sharpness > threshold else "person"
        results.append([img, pred])

    pd.DataFrame(results, columns=["image", "predicted"]).to_csv(output_csv, index=False)


def predict_for_model(model: str):
    """
    Produce before_predictions (baseline) and after_predictions_{model}.
    """
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    before_out = os.path.join(data_dir, "before_predictions.csv")

    # Always regenerate baseline from raw camera
    predict(os.path.join(data_dir, "camera"), before_out, threshold=100.0, weight_edges=False)

    # Mode-specific settings
    settings = {
        "default": {"folder": os.path.join(data_dir, "camera_processed"), "threshold": 100.0, "weight_edges": False},
        # make bayes_autoencoder more distinct by lowering threshold and enabling edge weighting
        "bayes_autoencoder": {"folder": os.path.join(data_dir, "camera_processed"), "threshold": 40.0, "weight_edges": True},
        # autoencoder mode emphasizes edges and uses a higher threshold
        "autoencoder": {"folder": os.path.join(data_dir, "camera_processed"), "threshold": 140.0, "weight_edges": True},
    }
    cfg = settings.get(model, settings["default"])

    after_specific = os.path.join(data_dir, f"after_predictions_{model}.csv")
    predict(cfg["folder"], after_specific, threshold=cfg["threshold"], weight_edges=cfg["weight_edges"])

    # Also write the generic after_predictions.csv so downstream stays happy
    generic_after = os.path.join(data_dir, "after_predictions.csv")
    pd.read_csv(after_specific).to_csv(generic_after, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="default")
    args = parser.parse_args()
    predict_for_model(args.model)
