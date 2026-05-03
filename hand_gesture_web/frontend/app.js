const API_BASE_URL = "";

const el = {
  modeCard: document.getElementById("mode-card"),
  modeValue: document.getElementById("mode-value"),
  sourceCard: document.getElementById("source-card"),
  sourceValue: document.getElementById("source-value"),
  gestureValue: document.getElementById("gesture-value"),
  stableGestureValue: document.getElementById("stable-gesture-value"),
  commandValue: document.getElementById("command-value"),
  robotStatusValue: document.getElementById("robot-status-value"),
  laptopCamera: document.getElementById("laptop-camera"),
  robotCamera: document.getElementById("robot-camera"),
  swipeSeriesValue: document.getElementById("swipe-series-value"),
  swipeDeltaValue: document.getElementById("swipe-delta-value"),
  landmarkToggleBtn: document.getElementById("landmark-toggle-btn"),
  connectionStatus: document.getElementById("connection-status"),
  message: document.getElementById("message"),
  buttons: document.querySelectorAll("[data-command]"),
};

let landmarkEnabled = false;

function setConnected(connected) {
  if (connected) {
    el.connectionStatus.textContent = "Server Connected";
    el.connectionStatus.classList.add("online");
    el.connectionStatus.classList.remove("offline");
    return;
  }
  el.connectionStatus.textContent = "Server Disconnected";
  el.connectionStatus.classList.remove("online");
  el.connectionStatus.classList.add("offline");
}

function setMessage(message, isError = false) {
  el.message.textContent = message || "";
  el.message.style.color = isError ? "#ff9d9d" : "#ffd67a";
}

function normalizeRobotStatus(value) {
  const raw = String(value || "").trim().toLowerCase();
  if (raw === "moving") return "Moving";
  if (raw === "stopped") return "Stopped";
  return value || "-";
}

function updateView(status) {
  const mode = String(status.mode || "AUTO").toUpperCase();
  const source = String(status.source || "gesture").toLowerCase();
  const gesture = String(status.gesture || "unknown");
  const stableGesture = String(status.stable_gesture || "unknown");
  const command = String(status.command || "NONE").toUpperCase();
  const robotStatus = normalizeRobotStatus(status.robot_status);

  el.modeValue.textContent = mode;
  el.gestureValue.textContent = gesture;
  el.stableGestureValue.textContent = stableGesture;
  el.commandValue.textContent = command;
  el.robotStatusValue.textContent = robotStatus;
  const swipeSeries =
    status.swipe_series ?? status.swipeSeries ?? status["swipe_series"] ?? "";
  const swipeDelta =
    status.swipe_delta_series ??
    status.swipeDeltaSeries ??
    status["swipe_delta_series"] ??
    "";
  if (el.swipeSeriesValue) {
    el.swipeSeriesValue.textContent = swipeSeries ? String(swipeSeries) : "-";
  }
  if (el.swipeDeltaValue) {
    el.swipeDeltaValue.textContent = swipeDelta ? String(swipeDelta) : "-";
  }

  el.modeCard.classList.remove("mode-auto", "mode-override");
  if (mode === "OVERRIDE") {
    el.modeCard.classList.add("mode-override");
  } else {
    el.modeCard.classList.add("mode-auto");
  }

  // 제어 입력 출처(gesture/web)를 텍스트와 색상으로 동시에 표시한다.
  if (el.sourceCard && el.sourceValue) {
    el.sourceCard.classList.remove("gesture", "web");
    if (source === "web") {
      el.sourceCard.classList.add("web");
      el.sourceValue.textContent = "Web";
    } else if (source === "gesture_client") {
      el.sourceCard.classList.add("gesture");
      el.sourceValue.textContent = "Gesture app";
    } else {
      el.sourceCard.classList.add("gesture");
      el.sourceValue.textContent = "Camera";
    }
  }
}

function updateLandmarkToggleButton() {
  if (!el.landmarkToggleBtn) return;
  el.landmarkToggleBtn.classList.remove("on", "off");
  if (landmarkEnabled) {
    el.landmarkToggleBtn.classList.add("on");
    el.landmarkToggleBtn.textContent = "Landmark ON";
  } else {
    el.landmarkToggleBtn.classList.add("off");
    el.landmarkToggleBtn.textContent = "Landmark OFF";
  }
}

async function getStatus() {
  const response = await fetch(`${API_BASE_URL}/status`, {
    method: "GET",
    cache: "no-store",
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch status: ${response.status}`);
  }
  return response.json();
}

async function refreshStatus() {
  try {
    const status = await getStatus();
    updateView(status);
    setConnected(true);
  } catch (error) {
    setConnected(false);
    setMessage("Server Disconnected", true);
  }
}

async function sendCommand(command) {
  el.buttons.forEach((button) => {
    button.disabled = true;
  });

  try {
    const response = await fetch(`${API_BASE_URL}/command`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ command }),
    });

    if (!response.ok) {
      throw new Error(`Command failed: ${response.status}`);
    }

    const status = await response.json();
    updateView(status);
    setConnected(true);
    setMessage(`Command sent: ${command}`);
  } catch (error) {
    setConnected(false);
    setMessage("Command send failed", true);
  } finally {
    el.buttons.forEach((button) => {
      button.disabled = false;
    });
  }
}

async function loadLandmarkState() {
  try {
    const response = await fetch(`${API_BASE_URL}/landmark`, { method: "GET" });
    if (!response.ok) return;
    const data = await response.json();
    landmarkEnabled = Boolean(data.enabled);
    updateLandmarkToggleButton();
  } catch (error) {
    // 랜드마크 상태 조회 실패는 스트림/상태 동작에 영향 없으므로 조용히 무시한다.
  }
}

async function toggleLandmark() {
  const nextState = !landmarkEnabled;
  try {
    const response = await fetch(`${API_BASE_URL}/landmark`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled: nextState }),
    });
    if (!response.ok) {
      throw new Error("Landmark toggle failed");
    }
    const data = await response.json();
    landmarkEnabled = Boolean(data.enabled);
    updateLandmarkToggleButton();
  } catch (error) {
    setMessage("Landmark toggle failed", true);
  }
}

function bindButtons() {
  el.buttons.forEach((button) => {
    button.addEventListener("click", () => {
      const command = button.dataset.command;
      if (!command) return;
      sendCommand(command);
    });
  });
}

function bindLandmarkToggleButton() {
  if (!el.landmarkToggleBtn) return;
  el.landmarkToggleBtn.addEventListener("click", () => {
    toggleLandmark();
  });
}

function startLaptopCamera() {
  if (el.laptopCamera) {
    el.laptopCamera.src = `${API_BASE_URL}/stream/laptop.mjpg`;
    el.laptopCamera.onerror = () => {
      setMessage("노트북 카메라 스트림 연결 실패", true);
    };
  }
}

function start() {
  bindButtons();
  bindLandmarkToggleButton();
  updateLandmarkToggleButton(); // 기본 상태는 OFF로 시작
  loadLandmarkState();
  startLaptopCamera();
  refreshStatus();
  window.setInterval(refreshStatus, 500);
}

start();

