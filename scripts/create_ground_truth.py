import os
import pandas as pd

camera_dir = "data/camera"
images = sorted(os.listdir(camera_dir))

rows = []
for img in images:
    # Simple dominant class assumption
    rows.append([img, "car"])

df = pd.DataFrame(rows, columns=["image", "label"])
df.to_csv("data/ground_truth.csv", index=False)

print(f"✅ ground_truth.csv created with {len(df)} entries")
