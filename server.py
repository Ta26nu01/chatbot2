from flask import Flask, request, jsonify, send_file
import os
import glob

app = Flask(__name__)

# Folder to save uploaded models
UPLOAD_FOLDER = "server_models"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Path to store aggregated global model
GLOBAL_MODEL_PATH = os.path.join(UPLOAD_FOLDER, "global_model_weights.h5")

@app.route("/upload_model", methods=["POST"])
def upload_model():
    """
    Endpoint for clients to upload their trained models.
    """
    if 'model_file' not in request.files or 'client_id' not in request.form:
        return jsonify({"error": "Missing file or client_id"}), 400

    model_file = request.files['model_file']
    client_id = request.form['client_id']

    # Save the uploaded model
    save_path = os.path.join(UPLOAD_FOLDER, f"{client_id}_global_model_weights.h5")
    model_file.save(save_path)

    return jsonify({"message": f"✅ Model from {client_id} saved successfully!", "path": save_path})


@app.route("/download_global", methods=["GET"])
def download_global():
    """
    Endpoint for clients to download the latest global model.
    """
    if not os.path.exists(GLOBAL_MODEL_PATH):
        return jsonify({"error": "Global model not available yet."}), 404

    return send_file(GLOBAL_MODEL_PATH, as_attachment=True)


@app.route("/aggregate", methods=["POST"])
def aggregate_models():
    """
    Optional: Aggregate client models into a single global model.
    Here we just pick the latest uploaded model as global for simplicity.
    """
    # Find all client models
    client_models = glob.glob(os.path.join(UPLOAD_FOLDER, "*_global_model_weights.h5"))
    if not client_models:
        return jsonify({"error": "No client models available for aggregation."}), 404

    # Example: overwrite global with the latest model (can implement averaging later)
    latest_model = max(client_models, key=os.path.getctime)
    os.replace(latest_model, GLOBAL_MODEL_PATH)

    return jsonify({"message": f"✅ Global model updated from {latest_model}"})


if __name__ == "__main__":
    # Make sure server_models folder exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(host="0.0.0.0", port=5001)
