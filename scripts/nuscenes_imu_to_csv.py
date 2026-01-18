import json
import pandas as pd
import os
import numpy as np

os.makedirs("data/imu", exist_ok=True)

# Load ego pose (used to derive IMU)
with open(r"C:\Users\harsh\Downloads\v1.0-mini\v1.0-mini\ego_pose.json") as f:
    poses = json.load(f)

rows = []

# Use pose differences to approximate IMU
for i in range(1, 16):
    prev = np.array(poses[i-1]["translation"])
    curr = np.array(poses[i]["translation"])

    # Approximate acceleration and angular change
    acceleration = np.linalg.norm(curr - prev)
    gyroscope = abs(curr[2] - prev[2])

    rows.append([i, acceleration, gyroscope])

df = pd.DataFrame(rows, columns=["time", "acceleration", "gyroscope"])
df.to_csv("data/imu/imu_data.csv", index=False)

print("✅ IMU CSV derived from ego_pose created successfully")
