from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Directory to store uploaded weights
UPLOAD_DIR = "uploaded_weights"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.route("/", methods=["GET"])
def home():
    return "Federated server up ✅"


@app.route("/upload_weights", methods=["POST"])
def upload_weights():
    """
    Receive weights from clients and save them as files.
    Each client sends a pickle file.
    """
    try:
        client_id = request.form.get("client_id", "unknown")
        file = request.files["file"]

        save_path = os.path.join(UPLOAD_DIR, f"client_{client_id}_weights.pkl")

        # Save the uploaded file
        with open(save_path, "wb") as f:
            f.write(file.read())

        return jsonify({
            "message": f"✅ Weights received from Client {client_id}",
            "saved_file": save_path
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/aggregate", methods=["GET"])
def aggregate():
    """
    Load all saved client weights and perform federated averaging.
    """
    weight_files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith(".pkl")]

    if not weight_files:
        return jsonify({"error": "No weights received yet"}), 400

    all_weights = []

    for wf in weight_files:
        with open(os.path.join(UPLOAD_DIR, wf), "rb") as f:
            weights = pickle.load(f)
            all_weights.append(weights)

    # Perform simple federated averaging
    averaged = [np.mean([w[i] for w in all_weights], axis=0) for i in range(len(all_weights[0]))]

    # Save aggregated global weights
    global_path = os.path.join(UPLOAD_DIR, "global_weights.pkl")
    with open(global_path, "wb") as f:
        pickle.dump(averaged, f)

    return jsonify({
        "message": "✅ Aggregation complete",
        "num_clients": len(all_weights),
        "global_file": global_path
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
