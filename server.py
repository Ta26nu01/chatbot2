from flask import Flask, request, send_file, jsonify
import pickle
import os
import numpy as np

app = Flask(__name__)

# Folder to save aggregated weights
WEIGHTS_FOLDER = "uploaded_weights"
os.makedirs(WEIGHTS_FOLDER, exist_ok=True)
GLOBAL_WEIGHTS_FILE = os.path.join(WEIGHTS_FOLDER, "global_weights.pkl")

# Temporary storage for client weights
client_weights_list = []

# ----------------------------
# Endpoint: Upload client weights
# ----------------------------
@app.route("/upload_weights", methods=["POST"])
def upload_weights():
    global client_weights_list

    if "weights" not in request.files:
        return "No weights file uploaded", 400

    weights_file = request.files["weights"]
    client_weights = pickle.load(weights_file)
    client_weights_list.append(client_weights)

    # Check if all clients have sent weights (example: 3 clients)
    if len(client_weights_list) == 3:
        # Aggregate weights (simple average)
        aggregated_weights = []
        for layers in zip(*client_weights_list):
            aggregated_weights.append(np.mean(layers, axis=0))

        # Save aggregated global weights
        with open(GLOBAL_WEIGHTS_FILE, "wb") as f:
            pickle.dump(aggregated_weights, f)

        # Clear temporary list for next round
        client_weights_list = []

        return jsonify({"message": "✅ Aggregation complete", "global_file": GLOBAL_WEIGHTS_FILE})

    return jsonify({"message": "✅ Weights received, waiting for other clients..."})

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
    app.run(debug=True)
