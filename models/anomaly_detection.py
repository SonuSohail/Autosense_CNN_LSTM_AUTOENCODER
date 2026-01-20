# anomaly_detection.py
# Autoencoder + rule-based anomaly detection for fused sensor data

import argparse
import json
import os
import pandas as pd
import numpy as np

# Autoencoder is optional (fallback to rules if unavailable)
try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
except Exception:  # ImportError or binary/runtime issues
    MLPRegressor = None
    StandardScaler = None


def autoencoder_scores(features: pd.DataFrame):
    """
    Train a lightweight autoencoder (MLPRegressor) to reconstruct inputs.
    Returns reconstruction error per row.
    """
    if StandardScaler is None or MLPRegressor is None:
        raise RuntimeError("Autoencoder dependencies not available (scikit-learn)")

    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    model = MLPRegressor(
        hidden_layer_sizes=(8, 4, 8),
        max_iter=400,
        random_state=42,
        learning_rate_init=0.001,
        early_stopping=True,
        n_iter_no_change=15,
    )
    model.fit(X, X)
    reconstructed = model.predict(X)
    errors = np.mean((X - reconstructed) ** 2, axis=1)
    return errors


def detect_anomalies(fused_file, use_autoencoder=True):
    """
    Detect anomalies in fused sensor data.
    Combines rule-based checks with autoencoder reconstruction error.
    """
    if not os.path.exists(fused_file):
        raise FileNotFoundError("Fused output missing for anomaly detection")

    df = pd.read_csv(fused_file)
    if df.empty:
        raise ValueError("Fused output is empty")

    anomalies = []

    # Rule-based checks
    for i, row in df.iterrows():
        if i == 0:
            continue  # skip first row

        speed_jump = abs(row["speed_smooth"] - df.loc[i-1, "speed_smooth"])
        accel_jump = abs(row["acceleration_smooth"] - df.loc[i-1, "acceleration_smooth"])
        lidar_distance = row["distance_m"]
        radar_velocity = row["object_velocity"]

        if speed_jump > 5:
            anomalies.append({"time": float(row["time"]), "type": "Speed Jump"})

        if accel_jump > 0.1:
            anomalies.append({"time": float(row["time"]), "type": "Acceleration Spike"})

        if lidar_distance < 5 or lidar_distance > 20:
            anomalies.append({"time": float(row["time"]), "type": "LiDAR Outlier"})

        if radar_velocity < 10 or radar_velocity > 100:
            anomalies.append({"time": float(row["time"]), "type": "Radar Outlier"})

    # Autoencoder-based anomaly score
    ae_stats = {}
    if use_autoencoder and (StandardScaler is not None) and (MLPRegressor is not None):
        feature_cols = [
            col for col in ["speed_smooth", "acceleration_smooth", "distance_m", "object_velocity", "fused_velocity", "confidence"]
            if col in df.columns
        ]
        # Require at least 30 rows to avoid sklearn validation_fraction errors
        if feature_cols and len(df) >= 30:
            try:
                errors = autoencoder_scores(df[feature_cols].fillna(0))
                df["ae_score"] = errors
                threshold = float(np.percentile(errors, 95))
                for i, row in df.iterrows():
                    if row["ae_score"] > threshold:
                        anomalies.append({
                            "time": float(row["time"]),
                            "type": "Autoencoder Outlier",
                            "score": float(row["ae_score"])
                        })
                ae_stats = {
                    "threshold": threshold,
                    "max_score": float(np.max(errors)),
                    "mean_score": float(np.mean(errors)),
                }
            except Exception as e:
                ae_stats = {"disabled": True, "reason": f"autoencoder failed: {e}"}
        elif feature_cols:
            ae_stats = {"disabled": True, "reason": "too few rows for autoencoder (need >=30)"}
    elif use_autoencoder:
        ae_stats = {"disabled": True, "reason": "scikit-learn unavailable for autoencoder on this Python env"}

    return anomalies, ae_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_autoencoder", type=str, default="true")
    args = parser.parse_args()

    use_auto = args.use_autoencoder.lower() != "false"
    fused_file = "../data/fused_final_output.csv"
    anomalies, ae_stats = detect_anomalies(fused_file, use_autoencoder=use_auto)

    payload = {
        "anomalies": anomalies,
        "count": len(anomalies),
        "autoencoder": ae_stats,
    }

    # Save anomalies for backend/UI
    os.makedirs("../data", exist_ok=True)
    with open("../data/anomalies.json", "w") as f:
        json.dump(payload, f, indent=2)

    print("Anomaly Detection Completed.\n")

    if anomalies:
        print(f"Detected {len(anomalies)} anomalies.")
    else:
        print("No anomalies detected.")


if __name__ == "__main__":
    main()
