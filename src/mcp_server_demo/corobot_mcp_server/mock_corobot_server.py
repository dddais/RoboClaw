"""
Mock CoRobot HTTP Server
========================
Simulates the Agibot G01 CoRobot control API on localhost:8765.

Provides a state-machine based task lifecycle so the full Agent → MCP → Robot
call chain can be exercised without real hardware.

Task lifecycle:
    IDLE  →(start)→  RUNNING  →(auto-complete / stop)→  IDLE
                                  ↓(reset)
                                IDLE

Usage:
    python mock_corobot_server.py [--port 8765] [--auto-complete-steps 8]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

logging.basicConfig(level=logging.INFO, format="%(asctime)s [MockCoRobot] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class TaskState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskContext:
    state: TaskState = TaskState.IDLE
    prompt: str = ""
    policy_host: str = "127.0.0.1"
    policy_port: int = 8001
    step_interval: float = 1.5
    current_step: int = 0
    total_steps: int = 0
    start_time: float = 0.0
    evaluate_params: dict[str, Any] = field(default_factory=dict)


auto_complete_steps: int = 8
task = TaskContext()
_step_ticker: asyncio.Task | None = None


def _ok(data: Any = None, message: str = "ok") -> JSONResponse:
    return JSONResponse({"success": True, "data": data, "message": message})


def _fail(message: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"success": False, "data": None, "message": message}, status_code=status)


async def _step_ticker_loop() -> None:
    """Background coroutine that advances task.current_step while RUNNING."""
    global task
    try:
        while task.state == TaskState.RUNNING:
            await asyncio.sleep(task.step_interval)
            if task.state != TaskState.RUNNING:
                break
            task.current_step += 1
            logger.info("step %d / %d", task.current_step, task.total_steps)
            if task.total_steps > 0 and task.current_step >= task.total_steps:
                task.state = TaskState.COMPLETED
                logger.info("task auto-completed after %d steps", task.current_step)
                break
    except asyncio.CancelledError:
        pass


def _start_ticker() -> None:
    global _step_ticker
    if _step_ticker and not _step_ticker.done():
        _step_ticker.cancel()
    _step_ticker = asyncio.get_event_loop().create_task(_step_ticker_loop())


def _stop_ticker() -> None:
    global _step_ticker
    if _step_ticker and not _step_ticker.done():
        _step_ticker.cancel()
        _step_ticker = None


# ---- Endpoints ----

async def set_evaluate_params(request: Request) -> JSONResponse:
    global task
    body = await request.json()
    params = body.get("evaluate_params", {})
    if not params:
        return _fail("missing evaluate_params")

    task.prompt = params.get("prompt", task.prompt)
    policy = params.get("policy", {})
    task.policy_host = policy.get("host", task.policy_host)
    task.policy_port = policy.get("port", task.policy_port)
    task.step_interval = params.get("step_interval", task.step_interval)
    task.evaluate_params = params
    task.current_step = 0
    task.total_steps = auto_complete_steps

    logger.info(
        "set_evaluate_params: prompt=%r  policy=%s:%s  step_interval=%.1f  total_steps=%d",
        task.prompt, task.policy_host, task.policy_port, task.step_interval, task.total_steps,
    )
    return _ok(
        {
            "prompt": task.prompt,
            "policy_host": task.policy_host,
            "policy_port": task.policy_port,
            "step_interval": task.step_interval,
            "total_steps": task.total_steps,
        },
        "evaluate params set",
    )


async def start_task(request: Request) -> JSONResponse:
    global task
    if task.state == TaskState.RUNNING:
        return _ok("task already running")
    task.state = TaskState.RUNNING
    task.current_step = 0
    task.start_time = time.time()
    _start_ticker()
    logger.info("start_task: prompt=%r", task.prompt)
    return _ok("task started")


async def stop_task(request: Request) -> JSONResponse:
    global task
    _stop_ticker()
    prev = task.state
    task.state = TaskState.IDLE
    logger.info("stop_task: %s → IDLE", prev)
    return _ok(f"task stopped (was {prev.value})")


async def reset_task(request: Request) -> JSONResponse:
    global task
    _stop_ticker()
    task.state = TaskState.IDLE
    task.current_step = 0
    task.total_steps = 0
    task.start_time = 0.0
    logger.info("reset_task: all state cleared")
    return _ok("task reset to initial state")


async def get_status(request: Request) -> JSONResponse:
    elapsed = time.time() - task.start_time if task.start_time else 0.0
    status_data = {
        "state": task.state.value,
        "prompt": task.prompt,
        "current_step": task.current_step,
        "total_steps": task.total_steps,
        "elapsed_s": round(elapsed, 2),
        "policy_host": task.policy_host,
        "policy_port": task.policy_port,
        "step_interval": task.step_interval,
    }
    return _ok(status_data)


async def set_prompt(request: Request) -> JSONResponse:
    global task
    body = await request.json()
    prompt = body.get("prompt")
    if not prompt:
        return _fail("missing 'prompt'")
    task.prompt = prompt
    logger.info("set_prompt: %r", task.prompt)
    return _ok({"prompt": task.prompt})


async def get_prompt(request: Request) -> JSONResponse:
    return _ok({"prompt": task.prompt})


async def root(request: Request) -> JSONResponse:
    return JSONResponse({
        "service": "MockCoRobotServer",
        "status": "running",
        "task_state": task.state.value,
        "endpoints": [
            "POST /set_evaluate_params",
            "POST /system/start_policytask",
            "POST /system/stop_policytask",
            "POST /system/reset_policytask",
            "GET  /status",
            "POST /set_prompt",
            "GET  /get_prompt",
        ],
    })


routes = [
    Route("/", root, methods=["GET"]),
    Route("/set_evaluate_params", set_evaluate_params, methods=["POST"]),
    Route("/system/start_policytask", start_task, methods=["POST"]),
    Route("/system/stop_policytask", stop_task, methods=["POST"]),
    Route("/system/reset_policytask", reset_task, methods=["POST"]),
    Route("/status", get_status, methods=["GET"]),
    Route("/set_prompt", set_prompt, methods=["POST"]),
    Route("/get_prompt", get_prompt, methods=["GET"]),
]

app = Starlette(routes=routes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock CoRobot HTTP Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765, help="Listen port (default: 8765)")
    parser.add_argument(
        "--auto-complete-steps", type=int, default=8,
        help="Steps after which a running task auto-completes (default: 8)",
    )
    args = parser.parse_args()

    global auto_complete_steps
    auto_complete_steps = args.auto_complete_steps

    logger.info("Starting MockCoRobotServer on %s:%d (auto_complete_steps=%d)", args.host, args.port, auto_complete_steps)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
