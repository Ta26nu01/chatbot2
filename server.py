from flask import Flask, request, jsonify, send_file
import os
from datetime import datetime

app = Flask(__name__)

# Directory to store uploaded models
MODEL_DIR = "server_models"
os.makedirs(MODEL_DIR, exist_ok=True)

LATEST_MODEL_PATH = os.path.join(MODEL_DIR, "latest_model.h5")

# -------------------------------
# 1️⃣ Upload trained model (from client)
# -------------------------------
@app.route("/upload_model", methods=["POST"])
def upload_model():
    if "model_file" not in request.files:
        return jsonify({"error": "No model file provided"}), 400

    model_file = request.files["model_file"]
    client_id = request.form.get("client_id", "unknown_client")

    # Save with timestamp to avoid overwrites
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(MODEL_DIR, f"{client_id}_{timestamp}.h5")
    model_file.save(save_path)

    # Also update latest model reference
    model_file.save(LATEST_MODEL_PATH)

    return jsonify({
        "message": f"✅ Model from {client_id} uploaded successfully!",
        "saved_as": save_path
    }), 200

# -------------------------------
# 2️⃣ Download latest model (for chatbot interface)
# -------------------------------
@app.route("/download_model", methods=["GET"])
def download_model():
    if not os.path.exists(LATEST_MODEL_PATH):
        return jsonify({"error": "No model available yet"}), 404
    return send_file(LATEST_MODEL_PATH, as_attachment=True)

# -------------------------------
# Run Flask server
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
