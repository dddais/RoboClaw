"""
Mock x2robot Bridge Server
===========================
Simulates the x2robot_bridge_server HTTP API on localhost:8766.
No ROS or shared memory required — generates synthetic camera images
and emulates running_mode state transitions.

Usage:
    python mock_x2robot_bridge.py [--port 8766] [--auto-complete-steps 8]
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import cv2
import numpy as np
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MockX2Robot] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

IMG_W, IMG_H = 640, 480


class TaskState(str, Enum):
    IDLE = "idle"
    AUTONOMOUS = "autonomous"
    TELEOP = "teleop"


@dataclass
class RobotState:
    running_mode: int = 0
    prompt: str = ""
    inference_ip: str = ""
    inference_port: int = 0
    step_interval: float = 1.5
    current_step: int = 0
    total_steps: int = 0
    start_time: float = 0.0
    left_arm_pos: list[float] = field(default_factory=lambda: [0.0] * 7)
    right_arm_pos: list[float] = field(default_factory=lambda: [0.0] * 7)


auto_complete_steps: int = 8
robot = RobotState()
_step_ticker: asyncio.Task | None = None


def _ok(data: Any = None, message: str = "ok") -> JSONResponse:
    return JSONResponse({"success": True, "data": data, "message": message})


def _fail(message: str, status: int = 400) -> JSONResponse:
    return JSONResponse(
        {"success": False, "data": None, "message": message},
        status_code=status,
    )


def _generate_mock_frame(label: str) -> np.ndarray:
    color_map = {
        "Left Wrist": ((60, 180, 120), (120, 60, 180)),
        "Head": ((180, 120, 60), (60, 120, 180)),
        "Right Wrist": ((120, 60, 180), (180, 180, 60)),
    }
    top, bot = color_map.get(label, ((128, 128, 128), (64, 64, 64)))
    img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    for row in range(IMG_H):
        alpha = row / max(IMG_H - 1, 1)
        for c in range(3):
            img[row, :, c] = int(top[c] * (1 - alpha) + bot[c] * alpha)
    ts = time.strftime("%H:%M:%S")
    cv2.putText(img, f"[MOCK] {label}", (20, IMG_H // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(img, ts, (20, IMG_H // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    return img


def _encode_b64(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


async def _step_ticker_loop() -> None:
    global robot
    try:
        while robot.running_mode == 1:
            await asyncio.sleep(robot.step_interval)
            if robot.running_mode != 1:
                break
            robot.current_step += 1
            logger.info("step %d / %d", robot.current_step, robot.total_steps)
            if robot.total_steps > 0 and robot.current_step >= robot.total_steps:
                robot.running_mode = 0
                logger.info("task auto-completed after %d steps", robot.current_step)
                break
    except asyncio.CancelledError:
        pass


def _start_ticker():
    global _step_ticker
    if _step_ticker and not _step_ticker.done():
        _step_ticker.cancel()
    _step_ticker = asyncio.get_event_loop().create_task(_step_ticker_loop())


def _stop_ticker():
    global _step_ticker
    if _step_ticker and not _step_ticker.done():
        _step_ticker.cancel()
        _step_ticker = None


# ---- Endpoints ----

async def cameras_latest(request: Request) -> JSONResponse:
    return _ok({
        "left_wrist": _encode_b64(_generate_mock_frame("Left Wrist")),
        "head": _encode_b64(_generate_mock_frame("Head")),
        "right_wrist": _encode_b64(_generate_mock_frame("Right Wrist")),
        "timestamp": time.time(),
    })


async def cameras_concatenated(request: Request) -> JSONResponse:
    left = _generate_mock_frame("Left Wrist")
    head = _generate_mock_frame("Head")
    right = _generate_mock_frame("Right Wrist")
    concat = cv2.hconcat([left, head, right])
    return _ok({
        "concatenated_image": _encode_b64(concat),
        "timestamp": time.time(),
    })


async def task_start(request: Request) -> JSONResponse:
    global robot
    robot.running_mode = 1
    robot.current_step = 0
    robot.start_time = time.time()
    _start_ticker()
    logger.info("task_start: running_mode=1")
    return _ok("running_mode set to 1 (autonomous)")


async def task_stop(request: Request) -> JSONResponse:
    global robot
    _stop_ticker()
    robot.running_mode = 0
    logger.info("task_stop: running_mode=0")
    return _ok("running_mode set to 0 (idle)")


async def task_reset(request: Request) -> JSONResponse:
    global robot
    _stop_ticker()
    robot.running_mode = 0
    robot.current_step = 0
    robot.total_steps = 0
    robot.left_arm_pos = [0.0] * 7
    robot.right_arm_pos = [0.0] * 7
    logger.info("task_reset: arms reset to home")
    return _ok("running_mode set to 0 and arms reset to home pose")


async def task_set_params(request: Request) -> JSONResponse:
    global robot
    body = await request.json()
    params = body.get("evaluate_params", {})
    if not params:
        return _fail("missing evaluate_params")

    robot.prompt = params.get("prompt", robot.prompt)
    policy = params.get("policy", {})
    robot.inference_ip = policy.get("host", robot.inference_ip)
    robot.inference_port = policy.get("port", robot.inference_port)
    robot.step_interval = params.get("step_interval", robot.step_interval)
    robot.current_step = 0
    robot.total_steps = auto_complete_steps

    logger.info(
        "set_params: prompt=%r  policy=%s:%s  step_interval=%.1f",
        robot.prompt, robot.inference_ip, robot.inference_port, robot.step_interval,
    )
    return _ok({
        "prompt": robot.prompt,
        "policy": {"host": robot.inference_ip, "port": robot.inference_port},
        "step_interval": robot.step_interval,
    })


async def task_set_prompt(request: Request) -> JSONResponse:
    global robot
    body = await request.json()
    prompt = body.get("prompt")
    if not prompt:
        return _fail("missing 'prompt'")
    robot.prompt = prompt
    logger.info("set_prompt: %r", robot.prompt)
    return _ok({"prompt": robot.prompt})


async def task_get_prompt(request: Request) -> JSONResponse:
    return _ok({"prompt": robot.prompt})


async def get_status(request: Request) -> JSONResponse:
    mode_text = {0: "idle", 1: "autonomous", 2: "teleop"}.get(robot.running_mode, "unknown")
    return _ok({
        "state": mode_text,
        "running_mode": robot.running_mode,
        "prompt": robot.prompt,
        "current_step": robot.current_step,
        "total_steps": robot.total_steps,
        "inference_server": f"{robot.inference_ip}:{robot.inference_port}" if robot.inference_ip else "",
        "left_arm_pos": robot.left_arm_pos,
        "right_arm_pos": robot.right_arm_pos,
    })


async def emergency_stop_ep(request: Request) -> JSONResponse:
    global robot
    _stop_ticker()
    robot.running_mode = 0
    logger.warning("EMERGENCY STOP triggered")
    return _ok("emergency stop executed — running_mode set to 0")


async def health(request: Request) -> JSONResponse:
    return _ok({
        "ros_available": False,
        "cameras_connected": 3,
        "running_mode": robot.running_mode,
    })


async def root(request: Request) -> JSONResponse:
    return JSONResponse({
        "service": "MockX2RobotBridge",
        "status": "running",
        "running_mode": robot.running_mode,
    })


routes = [
    Route("/", root, methods=["GET"]),
    Route("/cameras/latest", cameras_latest, methods=["GET"]),
    Route("/cameras/concatenated", cameras_concatenated, methods=["GET"]),
    Route("/task/start", task_start, methods=["POST"]),
    Route("/task/stop", task_stop, methods=["POST"]),
    Route("/task/reset", task_reset, methods=["POST"]),
    Route("/task/set_params", task_set_params, methods=["POST"]),
    Route("/task/set_prompt", task_set_prompt, methods=["POST"]),
    Route("/task/prompt", task_get_prompt, methods=["GET"]),
    Route("/status", get_status, methods=["GET"]),
    Route("/task/emergency_stop", emergency_stop_ep, methods=["POST"]),
    Route("/health", health, methods=["GET"]),
]

starlette_app = Starlette(routes=routes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock x2robot Bridge Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8766, help="Listen port")
    parser.add_argument(
        "--auto-complete-steps", type=int, default=8,
        help="Steps after which a running task auto-completes",
    )
    args = parser.parse_args()

    global auto_complete_steps
    auto_complete_steps = args.auto_complete_steps

    logger.info(
        "Starting MockX2RobotBridge on %s:%d (auto_complete_steps=%d)",
        args.host, args.port, auto_complete_steps,
    )
    uvicorn.run(starlette_app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
