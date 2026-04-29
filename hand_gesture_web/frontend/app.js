const els = {
  modeValue: document.getElementById("mode-value"),
  modeHint: document.getElementById("mode-hint"),
  gestureValue: document.getElementById("gesture-value"),
  confidenceValue: document.getElementById("confidence-value"),
  commandValue: document.getElementById("command-value"),
  robotStatusValue: document.getElementById("robot-status-value"),
  errorArea: document.getElementById("error-area"),
  cardMode: document.getElementById("card-mode"),
  errorTimer: null,
};

function setError(message) {
  if (!message) return;
  els.errorArea.hidden = false;
  els.errorArea.textContent = message;
}

function clearError() {
  els.errorArea.hidden = true;
  els.errorArea.textContent = "";
}

function confidenceToPercent(confidence) {
  const n = Number(confidence);
  if (!Number.isFinite(n)) return "-";
  return `${Math.round(n * 100)}%`;
}

function updateUI(status) {
  const mode = status.mode ?? "-";
  const gesture = status.gesture ?? "-";
  const confidence = status.confidence;
  const command = status.command ?? "-";
  const robotStatus = status.robot_status ?? "-";

  els.modeValue.textContent = mode;
  els.modeHint.textContent = mode === "AUTO" ? "제스처 인식 결과로 제어합니다." : "수동 명령이 우선합니다.";
  els.cardMode.dataset.mode = mode;

  els.gestureValue.textContent = gesture;
  els.confidenceValue.textContent = confidenceToPercent(confidence);
  els.commandValue.textContent = command;
  els.robotStatusValue.textContent =
    robotStatus === "moving" ? "Moving" : robotStatus === "stopped" ? "Stopped" : String(robotStatus);
  els.robotStatusValue.dataset.status = String(robotStatus).toLowerCase();
}

async function fetchStatus() {
  try {
    const res = await fetch("/status", { method: "GET" });
    if (!res.ok) {
      throw new Error(`상태 조회 실패: HTTP ${res.status}`);
    }
    const status = await res.json();
    clearError();
    updateUI(status);
  } catch (err) {
    setError(err?.message || "상태 조회 중 오류가 발생했습니다.");
  }
}

async function postCommand(command) {
  try {
    const res = await fetch("/command", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ command }),
    });
    if (!res.ok) {
      let detail = "";
      try {
        const data = await res.json();
        detail = data?.detail ? ` (${data.detail})` : "";
      } catch (_) {}
      throw new Error(`명령 전송 실패: HTTP ${res.status}${detail}`);
    }
    clearError();
    // 버튼 클릭 시 즉시 반영 (서버 상태가 갱신된 값을 다시 가져옴)
    await fetchStatus();
  } catch (err) {
    setError(err?.message || "명령 전송 중 오류가 발생했습니다.");
  }
}

function wireButtons() {
  document.querySelectorAll("[data-command]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const cmd = btn.getAttribute("data-command");
      if (!cmd) return;
      await postCommand(cmd);
    });
  });
}

function startPolling() {
  // 즉시 1회, 이후 1초마다 갱신
  fetchStatus();
  setInterval(fetchStatus, 1000);
}

wireButtons();
startPolling();

