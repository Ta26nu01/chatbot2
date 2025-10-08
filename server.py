from flask import Flask, request, send_file, jsonify
import os

app = Flask(__name__)

# ---------------- CONFIG ---------------- #
UPLOAD_FOLDER = "client_uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- UPLOAD ROUTE ---------------- #
@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Upload route for clients to send their weights and tokenizer.
    Expects:
        - 'weights' : weights.h5
        - 'tokenizer': tokenizer.pkl
        - 'client_id': unique identifier for the client
    """
    client_id = request.form.get('client_id', 'unknown')
    
    # Handle weights file
    weights_file = request.files.get('weights')
    if weights_file:
        weights_filename = f"{client_id}_weights.h5"
        weights_file.save(os.path.join(UPLOAD_FOLDER, weights_filename))
    else:
        return jsonify({"status": "error", "message": "Weights file missing"}), 400

    # Handle tokenizer file
    tokenizer_file = request.files.get('tokenizer')
    if tokenizer_file:
        tokenizer_filename = f"{client_id}_tokenizer.pkl"
        tokenizer_file.save(os.path.join(UPLOAD_FOLDER, tokenizer_filename))
    else:
        return jsonify({"status": "error", "message": "Tokenizer file missing"}), 400

    return jsonify({
        "status": "success",
        "uploaded_files": [weights_filename, tokenizer_filename]
    })

# ---------------- DOWNLOAD ROUTE ---------------- #
@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """
    Download a file by name from the uploads folder.
    """
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    else:
        return jsonify({"status": "error", "message": "File not found"}), 404

# ---------------- RUN SERVER ---------------- #
if __name__ == "__main__":
    # On Render, use host='0.0.0.0' and default port (from env)
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port)
