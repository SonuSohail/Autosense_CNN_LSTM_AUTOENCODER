from flask import Flask, jsonify
from flask_cors import CORS
import subprocess
import pandas as pd
import json
import os

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "AutoSense Backend Running"

@app.route("/run_pipeline")
def run_pipeline():
    try:
        subprocess.run(["python", "../models/cnn_model.py"], check=True)
        subprocess.run(["python", "../models/lstm_model.py"], check=True)
        subprocess.run(["python", "../fusion/fusion_logic.py"], check=True)
        subprocess.run(["python", "../models/anomaly_detection.py"], check=True)

        fused = pd.read_csv("../data/fused_final_output.csv")
        last = fused.iloc[-1]

        def read_status(file, default="Not Executed"):
            return open(file).read().strip() if os.path.exists(file) else default

        camera_status = read_status("../data/status_camera.txt")
        temporal_status = read_status("../data/status_temporal.txt")
        fusion_status = read_status("../data/status_fusion.txt")

        # Read anomaly result
        anomaly_msg = "No anomalies detected"
        if os.path.exists("../data/anomalies.json"):
            with open("../data/anomalies.json") as f:
                anomalies = json.load(f)
                if anomalies:
                    anomaly_msg = f"{len(anomalies)} anomalies detected"

        return jsonify({
            "status": "success",
            "camera": camera_status,
            "temporal": temporal_status,
            "fusion": fusion_status,
            "confidence": round(float(last["confidence"]), 2),
            "speed": round(float(last["speed_smooth"]), 2),
            "distance": round(float(last["distance_m"]), 2),
            "anomaly": anomaly_msg
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route("/metrics")
def metrics():
    subprocess.run(["python", "../models/predict_objects.py"], check=True)

    output = subprocess.run(
        ["python", "../models/evaluate_metrics.py"],
        capture_output=True,
        text=True
    )

    if not output.stdout.strip():
        return jsonify({"error": "No metrics output"}), 500

    return jsonify(json.loads(output.stdout))


if __name__ == "__main__":
    app.run(debug=True)
