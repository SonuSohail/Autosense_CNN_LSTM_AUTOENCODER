import os
import pandas as pd
import cv2

def predict(folder, output_csv):
    results = []

    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # Measure image sharpness (variance of Laplacian)
        sharpness = cv2.Laplacian(image, cv2.CV_64F).var()

        # Prediction rule
        if sharpness > 100:
            pred = "car"
        else:
            pred = "person"
        results.append([img, pred])

    pd.DataFrame(results, columns=["image", "predicted"]).to_csv(output_csv, index=False)

if __name__ == "__main__":
    predict("../data/camera", "../data/before_predictions.csv")
    predict("../data/camera_processed", "../data/after_predictions.csv")
