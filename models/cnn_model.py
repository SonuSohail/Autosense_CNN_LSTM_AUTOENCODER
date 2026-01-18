# cnn_model.py
# Simulated CNN-based camera data processing

import cv2
import os

def process_camera_images(input_folder, output_folder):
    """
    This function simulates CNN-based feature extraction
    by applying basic image preprocessing operations.
    """

    if not os.path.exists(input_folder):
        raise FileNotFoundError("Camera input folder not found")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    processed_count = 0

    for image_name in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_name)

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            continue

        # Convert to grayscale (simulating feature extraction)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur (simulating noise reduction)
        denoised = cv2.GaussianBlur(gray, (5, 5), 0)

        # Save processed image
        output_path = os.path.join(output_folder, image_name)
        cv2.imwrite(output_path, denoised)

        processed_count += 1

    if processed_count == 0:
        raise RuntimeError("No camera images were processed")

    print(f"Camera data processed successfully ({processed_count} images).")


def main():
    input_path = "../data/camera"
    output_path = "../data/camera_processed"
    process_camera_images(input_path, output_path)


if __name__ == "__main__":
    try:
        main()

        # ✅ Write status ONLY if processing succeeds
        os.makedirs("../data", exist_ok=True)
        with open("../data/status_camera.txt", "w") as f:
            f.write("Processed")

    except Exception as e:
        print("Camera module failed:", e)
        # ❌ Do NOT write status file on failure
        raise
