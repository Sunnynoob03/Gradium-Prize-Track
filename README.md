# ASL Recognition + TTS (SigntoSpeech)

This repository is a small demo that recognizes American Sign Language (letters) from MediaPipe hand landmarks, builds a word client-side, and exposes a Flask backend to predict letters and manage the current word.

**Files of interest**
- `server.py` — Flask backend that loads `asl_model.pkl` and exposes `/predict`, `/add_letter`, `/get_word`, `/reset_word` endpoints.
- `index.html` — Front-end using MediaPipe Hands, sends landmarks to the backend and shows the current word.
- `train_asl.py` — Script to train a RandomForest model from `asl_data.csv` and produce `asl_model.pkl`.
- `.env.example` — Example environment file (do not commit secrets).

## Requirements
- Python 3.8+
- See `requirements.txt` for Python dependencies.

## Quickstart (local)

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# or cmd
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Run the Flask server (this serves the model endpoints):

```bash
python server.py
```

3. Open `index.html` in your browser (Chrome/Edge recommended). The front-end uses your webcam via MediaPipe and sends landmark data to `http://127.0.0.1:5000/predict`.

Notes:
- The repository already contains `asl_model.pkl`. If you want to retrain the model, run:

```bash
python train_asl.py
```

- The front-end also contains code for TTS using an external API. Do NOT commit API keys. Use `.env` (see `.env.example`) to store secrets and ensure `.env` remains in `.gitignore`.
