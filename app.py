from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile, os
from suggestions import generate_suggestions
 
app = Flask(__name__)
CORS(app)  # allows your HTML page to call this API
 
@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
 
    uploaded = request.files["file"]
    # save to a temp file so generate_suggestions can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        uploaded.save(tmp.name)
        tmp_path = tmp.name
 
    try:
        results = generate_suggestions(tmp_path)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(tmp_path)
 
if __name__ == "__main__":
    app.run(debug=True, port=5000)
