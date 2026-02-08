from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle

app = Flask(__name__)
CORS(app)

# =========================
# LOAD ASL MODEL
# =========================

with open("asl_model.pkl", "rb") as f:
    model = pickle.load(f)

def normalize_landmarks(sample):
    sample = np.array(sample).astype(float)
    sample = sample.reshape(21, 3)

    wrist = sample[0]
    sample = sample - wrist

    max_dist = np.max(np.linalg.norm(sample, axis=1))
    if max_dist == 0:
        max_dist = 1

    sample = sample / max_dist
    return sample.flatten()

# =========================
# WORD BUILDER STATE
# =========================

current_word = ""


# =========================
# PREDICT LETTER
# =========================

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json.get("landmarks")

    if not data or len(data) != 63:
        return jsonify({"error": "Invalid landmark data"}), 400

    normalized = normalize_landmarks(data)
    prediction = model.predict([normalized])[0]

    return jsonify({"prediction": prediction})


# =========================
# ADD LETTER TO WORD
# =========================

@app.route("/add_letter", methods=["POST"])
def add_letter():
    global current_word
    letter = request.json.get("letter", "").strip()

    if len(letter) != 1:
        return jsonify({"error": "Invalid letter"}), 400

    current_word += letter
    return jsonify({"word": current_word})


# =========================
# INSERT SPACE
# =========================

@app.route("/add_space", methods=["POST"])
def add_space():
    global current_word
    current_word += " "
    return jsonify({"word": current_word})


# =========================
# GET CURRENT WORD
# =========================

@app.route("/get_word", methods=["GET"])
def get_word():
    return jsonify({"word": current_word})


# =========================
# RESET WORD
# =========================

@app.route("/reset_word", methods=["POST"])
def reset_word():
    global current_word
    current_word = ""
    return jsonify({"word": current_word})


# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)