# lstm_model.py
# Simulated LSTM-based temporal alignment for GPS and IMU data

import pandas as pd
import os

def align_sensor_data(gps_file, imu_file):
    """
    Simulates LSTM temporal alignment by
    synchronizing sensor data using time stamps
    and smoothing values.
    """

    if not os.path.exists(gps_file):
        raise FileNotFoundError("GPS data file not found")

    if not os.path.exists(imu_file):
        raise FileNotFoundError("IMU data file not found")

    # Load GPS and IMU data
    gps = pd.read_csv(gps_file)
    imu = pd.read_csv(imu_file)

    if gps.empty or imu.empty:
        raise ValueError("GPS or IMU data is empty")

    # Merge based on time (temporal alignment)
    fused = pd.merge(gps, imu, on="time", how="inner")

    if fused.empty:
        raise ValueError("Temporal alignment failed: no matching timestamps")

    # Simple smoothing (simulating LSTM output)
    fused["acceleration_smooth"] = fused["acceleration"].rolling(window=2).mean()
    fused["speed_smooth"] = fused["speed_kmph"].rolling(window=2).mean()

    print("Temporal alignment completed successfully.")
    return fused


def main():
    gps_path = "../data/gps/gps_data.csv"
    imu_path = "../data/imu/imu_data.csv"

    aligned_data = align_sensor_data(gps_path, imu_path)

    # Save output
    os.makedirs("../data", exist_ok=True)
    aligned_data.to_csv("../data/fused_lstm_output.csv", index=False)


if __name__ == "__main__":
    try:
        main()

        # ✅ Write status ONLY if temporal alignment succeeds
        with open("../data/status_temporal.txt", "w") as f:
            f.write("Completed")

    except Exception as e:
        print("Temporal alignment module failed:", e)
        # ❌ Do NOT write status file on failure
        raise
