from flask import Flask, request, jsonify
import pickle
import os
import numpy as np

app = Flask(__name__)

# ----------------------------
# Storage for client weights
# ----------------------------
UPLOAD_FOLDER = "uploaded_weights"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

client_weights = {}  # {client_id: weights}

# ----------------------------
# Home Route
# ----------------------------
@app.route("/", methods=["GET"])
def home():
    return "Federated server up ✅"

# ----------------------------
# Upload Weights
# ----------------------------
@app.route("/upload_weights", methods=["POST"])
def upload_weights():
    try:
        client_id = request.form.get("client_id")
        if not client_id:
            return jsonify({"error": "client_id missing"}), 400

        file = request.files["file"]
        weights = pickle.load(file)

        # Store in dictionary
        client_weights[client_id] = weights

        # Save a copy on disk
        save_path = os.path.join(UPLOAD_FOLDER, f"{client_id}_weights.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(weights, f)

        return jsonify({
            "message": f"✅ Weights received from {client_id}",
            "total_clients": len(client_weights)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ----------------------------
# Aggregate Weights
# ----------------------------
@app.route("/aggregate", methods=["GET"])
def aggregate():
    if not client_weights:
        return jsonify({"error": "No weights received yet"})

    try:
        # Convert dictionary values (weights list) into list
        all_weights = list(client_weights.values())

        # Perform simple FedAvg
        agg = [np.mean([w[i] for w in all_weights], axis=0) for i in range(len(all_weights[0]))]

        # Save aggregated weights
        global_path = os.path.join(UPLOAD_FOLDER, "global_weights.pkl")
        with open(global_path, "wb") as f:
            pickle.dump(agg, f)

        return jsonify({
            "message": "✅ Aggregation complete",
            "num_clients": len(client_weights),
            "global_file": global_path
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
