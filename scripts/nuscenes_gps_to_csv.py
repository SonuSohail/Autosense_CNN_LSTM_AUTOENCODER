import json
import pandas as pd
import os

os.makedirs("data/gps", exist_ok=True)

with open(r"C:\Users\harsh\Downloads\v1.0-mini\v1.0-mini\ego_pose.json") as f:
    poses = json.load(f)

rows = []
for i, p in enumerate(poses[:15]):
    rows.append([
        i,
        p["translation"][0],
        p["translation"][1],
        abs(p["translation"][2])
    ])

df = pd.DataFrame(rows, columns=["time", "latitude", "longitude", "speed_kmph"])
df.to_csv("data/gps/gps_data.csv", index=False)

print("✅ GPS CSV created successfully")
