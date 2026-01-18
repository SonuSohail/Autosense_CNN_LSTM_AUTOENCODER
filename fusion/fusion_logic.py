# fusion_logic.py
# Simplified multi-sensor fusion logic (Bayesian-inspired)

import pandas as pd
import os

def fuse_sensor_data(aligned_data, lidar_file, radar_file):
    """
    Simulates multi-sensor fusion with confidence estimation.
    """

    if not os.path.exists(lidar_file):
        raise FileNotFoundError("LiDAR data file not found")

    if not os.path.exists(radar_file):
        raise FileNotFoundError("Radar data file not found")

    lidar = pd.read_csv(lidar_file)
    radar = pd.read_csv(radar_file)

    if lidar.empty or radar.empty:
        raise ValueError("LiDAR or Radar data is empty")

    # Merge all sensor data by time
    fused = pd.merge(aligned_data, lidar, on="time", how="inner")
    fused = pd.merge(fused, radar, on="time", how="inner")

    if fused.empty:
        raise ValueError("Fusion failed: no matching timestamps across sensors")

    # Confidence estimation (rule-based)
    fused["confidence"] = (
        0.3 * fused["speed_smooth"].fillna(0) +
        0.3 * fused["acceleration_smooth"].fillna(0) * 100 +
        0.2 * fused["distance_m"] +
        0.2 * fused["object_velocity"]
    )

    print("Multi-sensor fusion completed successfully.")
    return fused


def main():
    aligned_path = "../data/fused_lstm_output.csv"
    lidar_path = "../data/lidar/lidar_data.csv"
    radar_path = "../data/radar/radar_data.csv"

    if not os.path.exists(aligned_path):
        raise FileNotFoundError("Aligned LSTM output not found")

    aligned_data = pd.read_csv(aligned_path)

    if aligned_data.empty:
        raise ValueError("Aligned data is empty")

    fused_output = fuse_sensor_data(aligned_data, lidar_path, radar_path)

    # Save fused output
    os.makedirs("../data", exist_ok=True)
    fused_output.to_csv("../data/fused_final_output.csv", index=False)


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
