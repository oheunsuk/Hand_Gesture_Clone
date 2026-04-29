from __future__ import annotations

import asyncio
import random
import threading
import time
from dataclasses import dataclass
from typing import Literal, Optional

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# ----------------------------
# Robot state (in-memory)
# ----------------------------

Mode = Literal["AUTO", "OVERRIDE"]
RobotStatus = Literal["moving", "stopped"]
Command = Literal["STOP", "RESUME", "LEFT", "RIGHT"]


@dataclass
class RobotState:
    mode: Mode = "AUTO"
    gesture: str = "palm"
    confidence: float = 0.93
    command: Command = "STOP"
    robot_status: RobotStatus = "stopped"
    last_updated_ts: float = time.time()
    last_manual_ts: Optional[float] = None


robot_state = RobotState()
state_lock = threading.Lock()


ALLOWED_COMMANDS: set[str] = {"STOP", "RESUME", "LEFT", "RIGHT"}
ALLOWED_MODES: set[str] = {"AUTO", "OVERRIDE"}
GESTURE_TO_COMMAND: dict[str, Command] = {
    # Example mapping: gestures -> robot commands
    "palm": "STOP",
    "thumb": "RESUME",
    "left": "LEFT",
    "right": "RIGHT",
}
GESTURES: list[str] = ["palm", "thumb", "left", "right"]


def _set_state(**kwargs) -> None:
    """Thread-safe update for the in-memory state."""
    with state_lock:
        for k, v in kwargs.items():
            setattr(robot_state, k, v)
        robot_state.last_updated_ts = time.time()


def _compute_robot_status_from_command(command: Command) -> RobotStatus:
    if command == "STOP":
        return "stopped"
    return "moving"


def _build_status_payload() -> dict:
    # Must match the UI requirement field names.
    with state_lock:
        return {
            "mode": robot_state.mode,
            "gesture": robot_state.gesture,
            "confidence": robot_state.confidence,
            "command": robot_state.command,
            "robot_status": robot_state.robot_status,
        }


# ----------------------------
# Background "gesture recognition"
# ----------------------------

async def gesture_recognition_loop() -> None:
    """
    In a real system, replace this loop with actual gesture recognition results.
    This is a functional placeholder so the UI works end-to-end.
    """
    while True:
        await asyncio.sleep(1.0)

        # OVERRIDE는 잠깐 유지 후 AUTO로 복귀합니다(데모/UX 목적).
        with state_lock:
            if robot_state.mode != "AUTO":
                if robot_state.last_manual_ts is not None:
                    manual_age = time.time() - robot_state.last_manual_ts
                    if manual_age >= 5.0:
                        robot_state.mode = "AUTO"
                        robot_state.last_manual_ts = None
                if robot_state.mode != "AUTO":
                    # While OVERRIDE: keep command/robot status as set by the operator.
                    continue

        gesture = random.choice(GESTURES)
        confidence = round(random.uniform(0.75, 0.99), 2)
        command = GESTURE_TO_COMMAND[gesture]
        robot_status = _compute_robot_status_from_command(command)
        _set_state(
            gesture=gesture,
            confidence=confidence,
            command=command,
            robot_status=robot_status,
        )


# ----------------------------
# FastAPI app
# ----------------------------

app = FastAPI(title="Gesture Robot Control UI")


@app.on_event("startup")
async def _startup() -> None:
    # Start the background loop that updates state every 1 second.
    asyncio.create_task(gesture_recognition_loop())


@app.get("/status")
def get_status():
    return JSONResponse(content=_build_status_payload())


@app.post("/command")
def post_command(payload: dict = Body(...)):
    """
    Expected payload:
      { "command": "STOP" | "RESUME" | "LEFT" | "RIGHT" }

    On any manual command, the system switches to OVERRIDE immediately.
    """
    if not isinstance(payload, dict) or "command" not in payload:
        raise HTTPException(status_code=400, detail="Missing 'command' field")

    command = payload.get("command")
    if not isinstance(command, str) or command not in ALLOWED_COMMANDS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid command. Allowed: {sorted(ALLOWED_COMMANDS)}",
        )

    command_lit: Command = command  # type narrowing for readability
    robot_status = _compute_robot_status_from_command(command_lit)

    # When operator overrides, we stop using gesture recognition for commands.
    _set_state(
        mode="OVERRIDE",
        command=command_lit,
        robot_status=robot_status,
        # Gesture display: indicate that command is manual.
        gesture="manual",
        confidence=1.0,
        last_manual_ts=time.time(),
    )

    return JSONResponse(content={"ok": True, "status": _build_status_payload()})


# ----------------------------
# Static frontend
# ----------------------------

from pathlib import Path  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = (THIS_DIR.parent / "frontend").resolve()

if FRONTEND_DIR.exists():
    # Important: mount static AFTER API routes are registered, so /status and /command keep working.
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
else:
    # Fallback: keep API working even if frontend isn't present yet.
    pass

