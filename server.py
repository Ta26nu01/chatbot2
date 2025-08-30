from flask import Flask, request, jsonify
import os, numpy as np

app = Flask(__name__)

# in-memory buffer for this demo
client_updates = []
EXPECTED_CLIENTS = int(os.getenv("EXPECTED_CLIENTS", "3"))  # change later if needed

@app.get("/")
def home():
    return "✅ federated server up"

@app.post("/upload")
def upload():
    """
    clients POST json like:
    {"client_id":"1","weights":[0.1,0.2,0.3]}
    (we’ll switch to real keras weight lists next)
    """
    data = request.get_json(force=True, silent=False)
    client_id = data["client_id"]
    weights  = np.array(data["weights"], dtype=float)
    client_updates.append(weights)
    print(f"received from client {client_id}: shape={weights.shape}")

    if len(client_updates) >= EXPECTED_CLIENTS:
        agg = np.mean(np.stack(client_updates, axis=0), axis=0)
        client_updates.clear()
        return jsonify({"status":"aggregated","global_weights": agg.tolist()})
    else:
        return jsonify({"status":"waiting","received":len(client_updates),"need":EXPECTED_CLIENTS})

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
