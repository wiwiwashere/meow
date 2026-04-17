// ----- state ---
let stream        = null;
let detectInterval = null;
const DETECT_MS   = 1500;   // send a frame for prediction every 1.5s

// elements
const video         = document.getElementById("feed");
const canvas        = document.getElementById("canvas");
const placeholder   = document.getElementById("feed-placeholder");
const camSelect     = document.getElementById("camera-select");
const permBtn       = document.getElementById("permission-btn");
const startBtn      = document.getElementById("start-btn");
const stopBtn       = document.getElementById("stop-btn");
const camError      = document.getElementById("cam-error");
const pill          = document.getElementById("status-pill");
const detLabel      = document.getElementById("det-label");
const detConf       = document.getElementById("det-conf");

const API_BASE = "https://meow-production-f89c.up.railway.app";


// ---- camera permission + device list --------

async function requestPermission() {
  camError.textContent = "";
  permBtn.disabled     = true;
  permBtn.textContent  = "Requesting...";

  try {
    // ask for permission
    // this triggers the browser prompt
    const tempStream = await navigator.mediaDevices.getUserMedia({ video: true });
    tempStream.getTracks().forEach(t => t.stop());   // release immediately

    await populateCameraList();

    permBtn.textContent = "Permission granted";
    startBtn.disabled   = false;
    camSelect.disabled  = false;

  } catch (err) {
    permBtn.disabled    = false;
    permBtn.textContent = "Allow camera access";
    camError.textContent = err.name === "NotAllowedError"
      ? "Camera access denied — please allow it in your browser settings."
      : `Error: ${err.message}`;
  }
}

async function populateCameraList() {
  const devices = await navigator.mediaDevices.enumerateDevices();
  const cameras = devices.filter(d => d.kind === "videoinput");

  camSelect.innerHTML = cameras.map((cam, i) =>
    `<option value="${cam.deviceId}">${cam.label || `Camera ${i + 1}`}</option>`
  ).join("");
}


// --- start / stop camera ---

async function startCamera() {
  if (stream) stopCamera();

  camError.textContent = "";
  const deviceId = camSelect.value;

  const constraints = {
    video: deviceId
      ? { deviceId: { exact: deviceId }, width: 640, height: 480 }
      : { width: 640, height: 480 }
  };

  try {
    stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject    = stream;
    placeholder.style.display = "none";

    startBtn.disabled = true;
    stopBtn.disabled  = false;

    // Start sending frames for prediction
    detectInterval = setInterval(sendFrame, DETECT_MS);

  } catch (err) {
    camError.textContent = `Could not start camera: ${err.message}`;
  }
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  video.srcObject = null;
  placeholder.style.display = "flex";

  clearInterval(detectInterval);
  detectInterval = null;

  startBtn.disabled = false;
  stopBtn.disabled  = true;

  pill.textContent = "No camera";
  pill.className   = "pill pill-idle";
  detLabel.textContent = "—";
  detConf.textContent  = "—";
}


// -- Send frame to backend for prediction ----------

async function sendFrame() {
  if (!stream || video.readyState < 2) return;

  // Draw current video frame to hidden canvas
  canvas.width  = 224;
  canvas.height = 224;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, 224, 224);

  // Convert to blob and POST to /predict_frame
  canvas.toBlob(async (blob) => {
    try {
      const form = new FormData();
      form.append("file", blob, "frame.jpg");

      const res  = await fetch(`${API_BASE}/predict_frame`, {
        method: "POST",
        body: form 
      });
      const data = await res.json();

      updateUI(data);
    } catch {
      // server unreachable
      // don't crash the loop
    }
  }, "image/jpeg", 0.8);
}

function updateUI(data) {
  detLabel.textContent = data.label;
  detConf.textContent  = `${data.confidence}% confidence`;

  if (data.is_cat) {
    pill.textContent = "Cat detected";
    pill.className   = "pill pill-cat";
  } else {
    pill.textContent = "No cat";
    pill.className   = "pill pill-not-cat";
  }
}


// -- WhatsApp alert -----

async function triggerAlert() {
  const btn = document.getElementById("alert-btn");
  const msg = document.getElementById("alert-msg");

  btn.disabled    = true;
  btn.textContent = "Sending...";
  msg.textContent = "";
  msg.className   = "alert-msg";

  try {
    const res  = await fetch(`${API_BASE}/alert`, { method: "POST" });

    if (!res.ok) {
      throw new Error(`Server error: ${res.status}`);
    }

    const data = await res.json();
    msg.textContent = data.message;
    msg.className   = data.success ? "alert-msg" : "alert-msg err";

  } catch (err) {
    msg.textContent = `Request failed: ${err.message}`;
    msg.className   = "alert-msg err";
  } finally {
    btn.disabled    = false;
    btn.textContent = "Send alert now";
  }
}


// -- Detection history -----

async function loadHistory() {
  const container = document.getElementById("history-list");
  try {
    const res  = await fetch("/history?limit=20");
    const data = await res.json();

    if (!data.detections.length) {
      container.innerHTML = "<p class='muted'>No detections yet.</p>";
      return;
    }

    container.innerHTML = data.detections.map(d => `
      <div class="history-item">
        <span>${d.label}</span>
        <span class="badge">${d.confidence}%</span>
        <span class="time">${d.time}</span>
      </div>
    `).join("");
  } catch {
    container.innerHTML = "<p class='muted'>Failed to load history.</p>";
  }
}

// ----instruction-------

function showSignupCard() {
  const card = document.getElementById("signup-card");
  if (card) card.style.display = "";
}

function hideSignup() {
  const card = document.getElementById("signup-card");
  if (card) card.style.display = "none";
}

async function submitSignup() {
  const phone = document.getElementById("signup-phone").value.trim();
  const msg = document.getElementById("signup-msg");

  if (!phone) {
    msg.textContent = "Please enter your WhatsApp number.";
    msg.classList.add("err");
    return;
  }

  try {
    const res = await fetch(`${API_BASE}/signup-whatsapp`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ phone })
    });

    const data = await res.json();

    if (!res.ok) {
      msg.textContent = data.detail || "Signup failed.";
      msg.classList.add("err");
      return;
    }

    localStorage.setItem("alertPhone", phone);
    msg.textContent = "Number saved. You can now receive alerts.";
    msg.classList.remove("err");

    setTimeout(() => hideSignup(), 900);
  } catch (err) {
    msg.textContent = "Could not save your number.";
    msg.classList.add("err");
  }
}

window.addEventListener("load", () => {
  const savedPhone = localStorage.getItem("alertPhone");
  if (!savedPhone) {
    showSignupCard();
  }
});

loadHistory();
setInterval(loadHistory, 10000);