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
import os
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

# sys.path.append(str(Path(__file__).parent.parent))
from app.database import DetectionDB
# from src.notifications.twilio_alert import send_cat_alert

from huggingface_hub import hf_hub_download

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re
import sqlite3
from src.notifications.twilio_alert import send_whatsapp_alert


# ---- Setup ---------------------------------------

app        = FastAPI(title="Cat Detector")
STATIC_DIR = Path(__file__).parent.parent / "static"
# MODEL_PATH = "models/best_binary_model.keras"
IMG_SIZE   = (224, 224)
THRESHOLD  = 0.5
DB_PATH = "users.db"

model_path = hf_hub_download(
    repo_id="wiwiwashere/meow",
    filename="best_binary_model.keras"
)
model = None

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# print(f"Loading model from {MODEL_PATH}...")
# model = tf.keras.models.load_model(MODEL_PATH)
# print("Model ready.")

db = DetectionDB()

# Shared last detection (updated by /predict_frame)
_lock  = threading.Lock()

_state = {
    "label"     : "—",
    "confidence": 0.0,
    "is_cat"    : False,
    "timestamp" : time.time(),
}

# model_path = hf_hub_download(
#     repo_id="wiwiwashere/meow",
#     filename="best_binary_model.keras"
# )
# model = tf.keras.models.load_model(model_path)


class SignupRequest(BaseModel):
    phone: str

class AlertRequest(BaseModel):
    phone: str
    label: str
    confidence: float | None = None



# --- Routes --------------------------------------------
@app.on_event("startup")
def load_model():
    global model

    model_path = hf_hub_download(
        repo_id="wiwiwashere/meow",
        filename="best_binary_model.keras",
        token=os.getenv("HF_TOKEN")  # okay if None for public repos
    )

    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")


@app.get("/", response_class=HTMLResponse)
def index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return index_file.read_text(encoding="utf-8")


@app.post("/predict_frame")
async def predict_frame(file: UploadFile = File(...)):
    """
    Receive a JPEG frame from the browser, run the model, return the result.
    Also saves cat detections to the history DB.
    History only records new detections when the label is different from the last one, to avoid duplicates when the cat is still in view.
    """
    global model

    if model is None:
        raise HTTPException(status_code=503, detail="Model is still loading")

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB").resize(IMG_SIZE)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

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
        "label": label,
        "confidence": round(conf * 100, 1),
        "is_cat" : is_cat,
    })


@app.get("/status")
def status():
    with _lock:
        s = dict(_state)

    return JSONResponse({
        "label" : s["label"],
        "confidence": round(s["confidence"] * 100, 1),
        "is_cat" : s["is_cat"],
        "timestamp" : s["timestamp"],
    })


@app.get("/history")
def history(limit: int = 20):
    return JSONResponse({"detections": db.get_recent(limit)})



@app.post("/alert")
def triggerAlert():
    with _lock:
        s = dict(_state)

    label = "cat" if s["is_cat"] else "no cat"

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT phone FROM whatsapp_subscribers")
    users = cur.fetchall()
    conn.close()

    if not users:
        return {
            "success": False,
            "message": "No signed-up WhatsApp users found."
        }

    results = []

    for (phone,) in users:
        try:
            msg = send_whatsapp_alert(phone, label, s["confidence"])
            results.append({"phone": phone, "sid": msg.sid})
        except Exception as e:
            results.append({"phone": phone, "error": str(e)})

    any_sent = any("sid" in r for r in results)

    return {
        "success": any_sent,
        "message": label if any_sent else "Alert failed for all users.",
        "results": results
    }



def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS whatsapp_subscribers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phone TEXT UNIQUE NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()

def normalize_phone(phone: str) -> str:
    phone = phone.strip()
    if not re.fullmatch(r"\+\d{10,15}", phone):
        raise ValueError("Phone number must be in E.164 format, like +13525551234")
    return phone

@app.post("/signup-whatsapp")
def signup_whatsapp(req: SignupRequest):
    try:
        phone = normalize_phone(req.phone)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT OR IGNORE INTO whatsapp_subscribers (phone) VALUES (?)",
        (phone,)
    )
    conn.commit()
    conn.close()

    return {"ok": True, "phone": phone}


@app.post("/clear-users")
def clear_users():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM whatsapp_subscribers")
    conn.commit()
    conn.close()

    return {"ok": True, "message": "All users cleared"}