from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Store weights from clients
client_weights = []

@app.route("/", methods=["GET"])
def home():
    return "Federated server up ✅"

@app.route("/upload_weights", methods=["POST"])
def upload_weights():
    try:
        file = request.files["file"]
        weights = pickle.load(file)
        client_weights.append(weights)
        return jsonify({"message": "✅ Weights received", "total_clients": len(client_weights)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/aggregate", methods=["GET"])
def aggregate():
    if not client_weights:
        return jsonify({"error": "No weights received yet"})
    
    # Simple averaging aggregation
    agg = [np.mean([w[i] for w in client_weights], axis=0) for i in range(len(client_weights[0]))]
    
    return jsonify({"message": "✅ Aggregation complete", "layers": len(agg)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
