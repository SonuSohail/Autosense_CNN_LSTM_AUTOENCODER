# anomaly_detection.py
# Rule-based anomaly detection for fused sensor data

import pandas as pd
import json
import os

def detect_anomalies(fused_file):
    """
    Detects anomalies in fused sensor data.
    Flags:
    - Sudden jumps in speed
    - Acceleration spikes
    - Unusual LiDAR distances
    - Unusual radar velocities
    """

    df = pd.read_csv(fused_file)
    anomalies = []

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

    return anomalies


if __name__ == "__main__":
    fused_file = "../data/fused_final_output.csv"
    anomalies = detect_anomalies(fused_file)

    # Save anomalies for backend/UI
    os.makedirs("../data", exist_ok=True)
    with open("../data/anomalies.json", "w") as f:
        json.dump(anomalies, f)

    print("Anomaly Detection Completed.\n")

    if anomalies:
        print("Detected anomalies:")
        for a in anomalies:
            print(f"Time {a['time']}: {a['type']}")
    else:
        print("No anomalies detected.")
