from flask import Flask, request, send_file, jsonify
import pickle
import os
import numpy as np

app = Flask(__name__)

# ----------------------------
# Configuration
# ----------------------------
UPLOAD_FOLDER = "uploaded_weights"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
GLOBAL_WEIGHTS_FILE = os.path.join(UPLOAD_FOLDER, "global_weights.pkl")

# Temporary storage for client weights
client_weights_list = []

# Number of clients expected
NUM_CLIENTS = 3

# ----------------------------
# Endpoint: Upload client weights
# ----------------------------
@app.route("/upload_weights", methods=["POST"])
def upload_weights():
    global client_weights_list

    if "weights" not in request.files:
        return "No weights file uploaded", 400

    weights_file = request.files["weights"]
    client_id = request.form.get("client_id", "unknown_client")

    # Load client weights
    client_weights = pickle.load(weights_file)
    client_weights_list.append(client_weights)

    # Save individual client weights (optional)
    client_file_path = os.path.join(UPLOAD_FOLDER, f"{client_id}_weights.pkl")
    with open(client_file_path, "wb") as f:
        pickle.dump(client_weights, f)
    print(f"✅ Received weights from {client_id} saved at {client_file_path}")

    # Check if all clients have uploaded
    if len(client_weights_list) == NUM_CLIENTS:
        # Aggregate (average) weights
        aggregated_weights = []
        for layers in zip(*client_weights_list):
            aggregated_weights.append(np.mean(layers, axis=0))

        # Save global weights
        with open(GLOBAL_WEIGHTS_FILE, "wb") as f:
            pickle.dump(aggregated_weights, f)

        # Clear temporary list for next round
        client_weights_list = []

        print(f"✅ Aggregation complete. Global weights saved at {GLOBAL_WEIGHTS_FILE}")
        return jsonify({"message": "✅ Aggregation complete", "global_file": GLOBAL_WEIGHTS_FILE})

    return jsonify({"message": f"✅ Weights received from {client_id}, waiting for other clients..."})

# ----------------------------
# Endpoint: Download global weights
# ----------------------------
@app.route("/download_global", methods=["GET"])
def download_global():
    if not os.path.exists(GLOBAL_WEIGHTS_FILE):
        return "Global weights not available yet.", 404
    return send_file(GLOBAL_WEIGHTS_FILE, as_attachment=True)

# ----------------------------
# Run server
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
