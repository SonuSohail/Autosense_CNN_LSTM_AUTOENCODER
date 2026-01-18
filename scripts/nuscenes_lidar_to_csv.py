import numpy as np
import os
import pandas as pd

lidar_dir = r"C:\Users\harsh\Downloads\v1.0-mini\samples\LIDAR_TOP"
rows = []

files = sorted(os.listdir(lidar_dir))[:15]

for i, file in enumerate(files):
    file_path = os.path.join(lidar_dir, file)
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 5)
    avg_distance = np.mean(np.linalg.norm(points[:, :3], axis=1))
    rows.append([i, avg_distance])

df = pd.DataFrame(rows, columns=["time", "distance_m"])
os.makedirs("data/lidar", exist_ok=True)
df.to_csv("data/lidar/lidar_data.csv", index=False)


print("✅ LiDAR CSV created successfully")
