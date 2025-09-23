from flask import Flask, request, jsonify, send_file
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# --------------------------
# Config
# --------------------------
BASE_URL = "https://chatbot2-ktaa.onrender.com"
UPLOAD_FOLDER = "uploaded_models"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

GLOBAL_MODEL_PATH = os.path.join(UPLOAD_FOLDER, "global_model_weights.h5")
NUM_CLIENTS = 3  # Adjust based on your project
client_models = {}  # store client file paths in memory
ROUND_COUNTER = 0

# --------------------------
# Upload model endpoint
# --------------------------
@app.route("/upload_model", methods=["POST"])
def upload_model():
    global client_models

    client_id = request.form.get("client_id")
    file = request.files.get("file")

    if not client_id or not file:
        return jsonify({"error": "Missing file or client_id"}), 400

    save_path = os.path.join(UPLOAD_FOLDER, f"{client_id}_model.h5")
    file.save(save_path)
    client_models[client_id] = save_path

    print(f"✅ Model received from {client_id} and saved at {save_path}")

    # Check if all clients uploaded
    if len(client_models) == NUM_CLIENTS:
        return jsonify({"message": f"All {NUM_CLIENTS} clients uploaded. Ready to aggregate!"})

    return jsonify({"message": f"Model from {client_id} uploaded. Waiting for more clients..."})

# --------------------------
# Aggregate models (FedAvg)
# --------------------------
@app.route("/aggregate", methods=["GET"])
def aggregate_models():
    global ROUND_COUNTER

    if len(client_models) < NUM_CLIENTS:
        return jsonify({"error": f"Waiting for {NUM_CLIENTS - len(client_models)} more clients."}), 400

    models = []
    for client_id, path in client_models.items():
        try:
            model = load_model(path)
            models.append(model.get_weights())
            print(f"📥 Loaded weights from {client_id}")
        except Exception as e:
            print(f"❌ Error loading {path}: {e}")

    # FedAvg aggregation
    avg_weights = []
    for weights_tuple in zip(*models):
        avg_weights.append(np.mean(weights_tuple, axis=0))

    # Rebuild global model architecture (must match clients)
    global_model = tf.keras.Sequential([
        tf.keras.layers.Embedding(100, 16, input_length=9),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(5, activation="softmax")
    ])
    global_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    global_model.set_weights(avg_weights)
    global_model.save(GLOBAL_MODEL_PATH)

    ROUND_COUNTER += 1
    client_models.clear()  # reset for next round
    print(f"🌍 Global model aggregated and saved at {GLOBAL_MODEL_PATH} (Round {ROUND_COUNTER})")

    return jsonify({"message": f"Global model aggregated successfully (Round {ROUND_COUNTER})"})

# --------------------------
# Download global model
# --------------------------
@app.route("/download_global", methods=["GET"])
def download_global():
    if not os.path.exists(GLOBAL_MODEL_PATH):
        return jsonify({"error": "Global model not available yet."}), 404
    return send_file(GLOBAL_MODEL_PATH, as_attachment=True)

# --------------------------
# Server info
# --------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Federated Learning Server is running 🚀",
        "upload_endpoint": f"{BASE_URL}/upload_model",
        "aggregate_endpoint": f"{BASE_URL}/aggregate",
        "download_endpoint": f"{BASE_URL}/download_global"
    })

# --------------------------
# Run server
# --------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
