# fusion_logic.py
# Simplified multi-sensor fusion logic (Bayesian-inspired)

import pandas as pd
import os
import argparse
import json

def fuse_sensor_data(aligned_data, lidar_file, radar_file, model="default", use_autoencoder=False):
    """
    Bayesian-inspired sensor fusion with confidence estimation.
    """

    if not os.path.exists(lidar_file):
        raise FileNotFoundError("LiDAR data file not found")

    if not os.path.exists(radar_file):
        raise FileNotFoundError("Radar data file not found")

    lidar = pd.read_csv(lidar_file)
    radar = pd.read_csv(radar_file)

    if lidar.empty or radar.empty:
        raise ValueError("LiDAR or Radar data is empty")
    aligned_data["time"] = aligned_data["time"].astype(int)
    lidar["time"] = lidar["time"].astype(int)
    radar["time"] = radar["time"].astype(int)

    # Merge all sensor data by time
    fused = pd.merge(aligned_data, lidar, on="time", how="inner")
    fused = pd.merge(fused, radar, on="time", how="inner")

    if fused.empty:
        raise ValueError("Fusion failed: no matching timestamps across sensors")

    # Bayesian-style weighting using inverse variance of each source
    speed_var = fused["speed_smooth"].var() or 1.0
    radar_var = fused["object_velocity"].var() or 1.0
    lidar_var = fused["distance_m"].var() or 1.0

    # Base inverse-variance weights
    weight_speed = 1.0 / (speed_var + 1e-6)
    weight_radar = 1.0 / (radar_var + 1e-6)
    weight_lidar = 1.0 / (lidar_var + 1e-6)

    # Model-specific modifiers (allows different fusion behavior per selected model)
    modifiers = {
        "default": {"speed": 1.0, "radar": 1.0, "lidar": 1.0, "conf_scale": 1.0},
        "bayes_autoencoder": {"speed": 1.0, "radar": 1.4, "lidar": 0.9, "conf_scale": 1.2},
        "autoencoder": {"speed": 0.9, "radar": 1.1, "lidar": 1.0, "conf_scale": 1.3},
    }
    mod = modifiers.get(str(model), modifiers["default"])

    weight_speed *= mod["speed"]
    weight_radar *= mod["radar"]
    weight_lidar *= mod["lidar"]

    # Normalize weights for speed+radar fusion baseline, include lidar when appropriate
    weight_sum = weight_speed + weight_radar
    fused["fused_velocity"] = (
        (weight_speed * fused["speed_smooth"].fillna(0) +
         weight_radar * fused["object_velocity"].fillna(0)) / weight_sum
    )

    # Confidence combines normalized weights and spatial consistency
    conf_raw = (
        0.4 * fused["fused_velocity"].abs() +
        0.3 * fused["distance_m"].rolling(window=2, min_periods=1).mean().fillna(0) +
        0.3 * fused["acceleration_smooth"].abs().fillna(0) * 10
    )
    # Apply optional autoencoder scaling to confidence
    conf_min, conf_max = conf_raw.min(), conf_raw.max()
    conf_norm = (conf_raw - conf_min) / (conf_max - conf_min + 1e-6)
    if use_autoencoder:
        conf_norm = conf_norm * (0.9 + 0.1 * mod["conf_scale"])  # small boost when autoencoder used
    fused["confidence"] = conf_norm.clip(0.0, 1.0)

    # Store weights for history/visuals
    fused["w_speed"] = round(weight_speed / (weight_sum), 3)
    fused["w_radar"] = round(weight_radar / (weight_sum), 3)
    fused["w_lidar"] = round(weight_lidar / (weight_speed + weight_radar + weight_lidar), 3)

    print("Multi-sensor fusion completed successfully.")
    return fused


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="default")
    parser.add_argument("--use_autoencoder", type=str, default="false")
    args = parser.parse_args()

    model = args.model
    use_autoencoder = str(args.use_autoencoder).lower() == "true"

    aligned_path = "../data/fused_lstm_output.csv"
    lidar_path = "../data/lidar/lidar_data.csv"
    radar_path = "../data/radar/radar_data.csv"

    if not os.path.exists(aligned_path):
        raise FileNotFoundError("Aligned LSTM output not found")

    aligned_data = pd.read_csv(aligned_path)

    if aligned_data.empty:
        raise ValueError("Aligned data is empty")

    fused_output = fuse_sensor_data(aligned_data, lidar_path, radar_path, model=model, use_autoencoder=use_autoencoder)

    # Add risk level based on confidence and fused velocity
    fused_output["risk_level"] = fused_output["confidence"].apply(
        lambda x: "LOW" if x > 0.75 else ("MEDIUM" if x > 0.4 else "HIGH")
    )

    # Save fused output
    os.makedirs("../data", exist_ok=True)
    fused_output.to_csv("../data/fused_final_output.csv", index=False)

    # Also persist a small metadata JSON for frontend use
    try:
        meta = {
            "model": model,
            "use_autoencoder": use_autoencoder,
            "rows": int(len(fused_output)),
        }
        with open("../data/fusion_meta.json", "w") as f:
            json.dump(meta, f)
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()

        # ✅ Write status ONLY if fusion succeeds
        with open("../data/status_fusion.txt", "w") as f:
            f.write("Completed")

    except Exception as e:
        print("Fusion module failed:", e)
        # ❌ Do NOT write status file on failure
        raise
