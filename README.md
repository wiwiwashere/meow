# MEOW — Real-Time Cat Detection System
 
A full-stack deep learning project that answers one question: **is there a cat in frame?**
 
**Check out the project:** https://meow.up.railway.app
 
---
 
## Overview
 
MEOW is a binary image classifier built with transfer learning on a custom dataset of ~9,000 images. It runs as a web app. When a cat is detected, it automatically sends a WhatsApp notification to all registered subscribers.
 
---
 
## Features
 
- **Live webcam detection** - browser captures frames and sends them to the backend for classification every 1.5 seconds
- **Mobile friendly** - works on phone browsers, no app install needed
- **WhatsApp alerts** - auto-notifies subscribers via Twilio when a cat appears, with a 60-second cooldown to prevent spam
- **Detection history** - logs every new detection to SQLite, viewable in the app
- **Smart state tracking** - only saves to history on label changes or significant confidence jumps, avoiding duplicate entries
---
 
## Tech Stack
 
| Layer | Technology |
|---|---|
| Model | EfficientNetB0 (transfer learning, fine-tuned) |
| Backend | FastAPI + Uvicorn |
| Frontend | HTML / CSS / JavaScript |
| Notifications | Twilio WhatsApp API |
| Database | SQLite |
| Model hosting | Hugging Face Hub |
| Deployment | Railway |
 
---
 
## Model
 
- **Architecture:** EfficientNetB0 pretrained on ImageNet, fine-tuned for binary classification (cat vs. not_cat)
- **Dataset:** ~9,000 images - cats, dogs, and felidae (lions, tigers, etc.)
- **Training:** Two-phase transfer learning
  - Phase 1: frozen backbone, train head only
  - Phase 2: unfreeze top 20 layers, fine-tune at `lr=1e-5`
- **Classes:** `cat`(0) & `not_cat` (1); dogs and felidae merged as hard negatives
- **Split:** 80% train / 10% val / 10% test
---
 
## API Endpoints
 
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web frontend |
| `POST` | `/predict_frame` | Submit a JPEG frame, get prediction |
| `GET` | `/status` | Current detection state |
| `GET` | `/history` | Recent detection log |
| `POST` | `/alert` | Manually trigger WhatsApp alert |
| `POST` | `/signup-whatsapp` | Register phone for alerts |
| `POST` | `/clear-users` | Remove all subscribers |
 
---
 
## WhatsApp Alerts Setup
 
1. Open WhatsApp and message **+1 415 523 8886**
2. Send: `join aware-month`
3. You'll receive a confirmation saying you're now subscribed
4. Alerts fire automatically when a cat is detected (60s cooldown between messages)

---

## Demo Video

[![Watch the demo](https://img.youtube.com/vi/kaF1jJNUAdU/maxresdefault.jpg)](https://youtu.be/kaF1jJNUAdU)
