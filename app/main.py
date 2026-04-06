"""
Cat Detector — FastAPI Backend

The browser now handles the webcam directly and POSTs frames here for
prediction. This means it works on mobile too — no server-side camera needed.

Endpoints:
  GET  /                  → frontend
  POST /predict_frame     → receives a JPEG frame, returns prediction
  GET  /status            → last prediction result
  GET  /history           → recent detection history
  POST /alert             → manually trigger WhatsApp alert

Run:
  pip install fastapi uvicorn pillow sqlalchemy python-dotenv twilio tensorflow
  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Open:
  http://localhost:8000             (desktop)
  http://<your-local-ip>:8000       (mobile on same WiFi)
"""
import io
import sys
import time
import threading
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from tensorflow.keras.applications.efficientnet import preprocess_input

sys.path.append(str(Path(__file__).parent.parent))
from app.database import DetectionDB
from src.notifications.twilio_alert import send_cat_alert

# ── Setup ─────────────────────────────────────────────────────────────────────

app        = FastAPI(title="Cat Detector")
STATIC_DIR = Path(__file__).parent.parent / "static"
MODEL_PATH = "models/best_binary_model.keras"
IMG_SIZE   = (224, 224)
THRESHOLD  = 0.5

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

print(f"Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model ready.")

db = DetectionDB()

# Shared last detection (updated by /predict_frame)
_lock  = threading.Lock()
_state = {
    "label"     : "—",
    "confidence": 0.0,
    "is_cat"    : False,
    "timestamp" : time.time(),
}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "index.html").read_text()


@app.post("/predict_frame")
async def predict_frame(file: UploadFile = File(...)):
    """
    Receive a JPEG frame from the browser, run the model, return the result.
    Also saves cat detections to the history DB.
    History only records new detections when the label is different from the last one, to avoid duplicates when the cat is still in view.
    """
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    prob   = float(model.predict(arr, verbose=0)[0][0])
    is_cat = prob < THRESHOLD
    conf   = (1 - prob) if is_cat else prob
    label  = "Cat" if is_cat else "Not a cat"

    prev_label = _state["label"]

    needs_update = False


    # save to history
    if is_cat and prev_label != "Cat":  # new cat detected
        needs_update = True
    elif not is_cat and prev_label != "Not a cat": # new non-cat detected
        needs_update = True
    elif is_cat and prev_label == "Cat" and (prob - _state["confidence"] > 0.1): # confidence increased significantly for cat
        needs_update = True


    if needs_update:
        with _lock:
            _state.update({
                "label"     : label,
                "confidence": conf,
                "is_cat"    : is_cat,
                "timestamp" : time.time(),
            })
        db.save(label, conf, "predict_frame")

    return JSONResponse({
        "label"     : label,
        "confidence": round(conf * 100, 1),
        "is_cat"    : is_cat,
    })


@app.get("/status")
def status():
    with _lock:
        s = dict(_state)
    return JSONResponse({
        "label"     : s["label"],
        "confidence": round(s["confidence"] * 100, 1),
        "is_cat"    : s["is_cat"],
        "timestamp" : s["timestamp"],
    })


@app.get("/history")
def history(limit: int = 20):
    return JSONResponse({"detections": db.get_recent(limit)})


@app.post("/alert")
def trigger_alert():
    with _lock:
        s = dict(_state)
    if not s["is_cat"]:
        return JSONResponse({"success": False, "message": "No cat currently detected"})
    success = send_cat_alert(source="manual (web app)")
    return JSONResponse({
        "success": success,
        "message": "WhatsApp alert sent!" if success else "Alert failed — check .env",
    })