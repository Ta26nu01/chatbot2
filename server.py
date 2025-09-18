from flask import Flask, request, jsonify, send_file
import os
import pickle
import numpy as np

app = Flask(__name__)

# ----------------------------
# Configuration
# ----------------------------
UPLOAD_FOLDER = "uploaded_weights"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

GLOBAL_WEIGHTS_FILE = os.path.join(UPLOAD_FOLDER, "global_weights.pkl")

# Store client weights temporarily
client_weights = {}
expected_clients = 3   # 🔹 Change this to number of clients in your setup
uploaded_count = 0


# ----------------------------
# Homepage route
# ----------------------------
@app.route("/", methods=["GET"])
def home():
    return "🚀 Federated Learning Server is running! Use /upload_weights to upload, /aggregate to aggregate, and /download_global to fetch the global model."


# ----------------------------
# Upload weights from client
# ----------------------------
@app.route("/upload_weights", methods=["POST"])
def upload_weights():
    global uploaded_count

    if "file" not in request.files or "client_id" not in request.form:
        return jsonify({"error": "Missing file or client_id"}), 400

    file = request.files["file"]
    client_id = request.form["client_id"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save client weights
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    with open(filepath, "rb") as f:
        weights = pickle.load(f)
        client_weights[client_id] = weights

    uploaded_count += 1
    print(f"📥 Received weights from {client_id} ({uploaded_count}/{expected_clients})")

    # If all clients have uploaded, aggregate automatically
    if uploaded_count >= expected_clients:
        aggregate_and_save()
        return jsonify({
            "message": "✅ All clients uploaded. Aggregation complete!",
            "global_file": GLOBAL_WEIGHTS_FILE
        })

    return jsonify({"message": f"✅ Weights from {client_id} received. Waiting for more clients..."})


# ----------------------------
# Manual aggregation trigger
# ----------------------------
@app.route("/aggregate", methods=["GET"])
def aggregate_manual():
    if len(client_weights) < expected_clients:
        return jsonify({
            "message": f"⏳ Waiting for more clients. Received {len(client_weights)} of {expected_clients}."
        }), 400

    aggregate_and_save()
    return jsonify({
        "message": "✅ Manual aggregation complete!",
        "global_file": GLOBAL_WEIGHTS_FILE
    })


# ----------------------------
# Download global weights
# ----------------------------
@app.route("/download_global", methods=["GET"])
def download_global():
    if not os.path.exists(GLOBAL_WEIGHTS_FILE):
        return jsonify({"error": "No global weights available yet"}), 404
    return send_file(GLOBAL_WEIGHTS_FILE, as_attachment=True)


# ----------------------------
# Helper: Aggregate and Save
# ----------------------------
def aggregate_and_save():
    global client_weights, uploaded_count

    print("⚡ Aggregating weights from clients...")
    aggregated_weights = []
    for layers in zip(*client_weights.values()):
        aggregated_weights.append(np.mean(layers, axis=0))

    with open(GLOBAL_WEIGHTS_FILE, "wb") as f:
        pickle.dump(aggregated_weights, f)

    print(f"✅ Aggregated global weights saved at {GLOBAL_WEIGHTS_FILE}")

    # Reset for next round
    client_weights = {}
    uploaded_count = 0


# ----------------------------
# Run the server
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
