from __future__ import annotations

import asyncio
import base64
import contextlib
import gc
import logging
import os
from pathlib import Path
import re
import signal
import sys
import termios
import textwrap
import traceback
import tty
from enum import Enum
from typing import AsyncGenerator

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agent_demo.interaction_layer.auto_inference_prompt import AUTO_INFERENCE_PROMPT
from agent_demo.agent_layer.agent_components.agent_tools.local_skill_registry import LocalSkillRegistry
from agent_demo.agent_layer.agent_core import ImgActAgent
from agent_demo.agent_layer.agent_prompt import ImgActAgentPrompt
from agent_demo.common.response_formatter import format_response_text
from agent_demo.common.root_logger import setup_root_logging
from agent_demo.interaction_layer.feishu_long_connection import start_feishu_long_connection
from agent_demo.interaction_layer.local_skill_support import prepare_agent_message
from agent_demo.machine_layer.dataloader_factory import create_robot_dataloader
from agent_demo.types.agent_types import ActAgentState, BaseAgentCard, ChatAPIConfig, TextParam

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[4]
os.makedirs("./applog/", exist_ok=True)
setup_root_logging(default_log_path="./applog/", console_output=False)

ORGANIZE_COMPLETE_MSG = (
    "The task to organize and clean the table has been successfully completed. "
    "All four sub-tasks and their reset steps were executed without issues."
)

ROLLOUT_COMPLETE_MSG = (
    "The rollout task has been successfully completed. "
    "All steps (forward, reset, reverse, reset) were executed without issues."
)
LONG_HORIZON_SKILL_NAME = "long-horizon-execution"

CHAT_SCROLL_STEP = 3
MOUSE_SCROLL_UP_BUTTON = 64
MOUSE_SCROLL_DOWN_BUTTON = 65
MOUSE_EVENT_MODIFIER_MASK = 4 | 8 | 16 | 32
MOUSE_TRACKING_ENABLE = "\x1b[?1000h\x1b[?1006h\x1b[?1015h"
MOUSE_TRACKING_DISABLE = "\x1b[?1000l\x1b[?1006l\x1b[?1015l"
CHAT_CODE_BLOCK_PLACEHOLDER = "[code omitted]"
SIGNATURE_LINE_LIMIT = 2
ASSISTANT_SALIENT_KEYWORDS = (
    "complete",
    "completed",
    "success",
    "successful",
    "failed",
    "error",
    "summary",
    "result",
    "status",
    "done",
    "finished",
)


def _create_robot_dataloader():
    return create_robot_dataloader()


def _compact_chat_message(content: str, role: str) -> str:
    if not content:
        return ""

    return content.replace("\r\n", "\n").strip()


def _signature_chat_message(content: str, role: str) -> str:
    if not content:
        return ""

    text = content.replace("\r\n", "\n").strip()
    text = re.sub(r"```[\w+-]*\n.*?```", CHAT_CODE_BLOCK_PLACEHOLDER, text, flags=re.DOTALL)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{2,}", "\n", text)

    compact_lines: list[str] = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        if line.startswith("Final response: "):
            line = line[len("Final response: ") :]
        if compact_lines and compact_lines[-1] == line:
            continue
        compact_lines.append(line)

    if not compact_lines:
        return ""

    if role != "assistant":
        return "\n".join(compact_lines[:SIGNATURE_LINE_LIMIT])

    salient_lines = [
        line for line in compact_lines if any(keyword in line.lower() for keyword in ASSISTANT_SALIENT_KEYWORDS)
    ]
    bullet_lines = [line for line in compact_lines if re.match(r"^(\d+\.|[-*])\s", line)]
    plain_lines = [line for line in compact_lines if line != CHAT_CODE_BLOCK_PLACEHOLDER]

    if salient_lines:
        visible_lines = salient_lines[:SIGNATURE_LINE_LIMIT]
        if (
            len(visible_lines) == 1
            and ("status" in visible_lines[0].lower() or visible_lines[0].endswith(":"))
            and bullet_lines
        ):
            visible_lines.append(bullet_lines[0])
    elif bullet_lines:
        visible_lines = bullet_lines[:SIGNATURE_LINE_LIMIT]
    elif plain_lines:
        visible_lines = plain_lines[:SIGNATURE_LINE_LIMIT]
    else:
        visible_lines = compact_lines[:SIGNATURE_LINE_LIMIT]

    return "\n".join(visible_lines)


def _collapse_chat_entries(chat_history: list[dict[str, str]], max_messages: int | None) -> list[dict[str, str | int]]:
    collapsed: list[dict[str, str | int]] = []
    if max_messages is None:
        source_messages = chat_history
    else:
        source_messages = chat_history[-max_messages:]

    for message in source_messages:
        compact_content = _compact_chat_message(message["content"], message["role"])
        if not compact_content:
            continue

        if (
            collapsed
            and collapsed[-1]["role"] == message["role"]
            and _chat_message_signature(
                _signature_chat_message(str(collapsed[-1]["content"]), str(collapsed[-1]["role"]))
            )
            == _chat_message_signature(_signature_chat_message(compact_content, message["role"]))
        ):
            collapsed[-1]["repeat"] = int(collapsed[-1]["repeat"]) + 1
            continue

        collapsed.append(
            {
                "role": message["role"],
                "content": compact_content,
                "repeat": 1,
            }
        )

    return collapsed


def _chat_message_signature(content: str) -> str:
    normalized_lines = []
    for line in content.splitlines():
        normalized = line.strip().rstrip(" .,:;!?").casefold()
        if normalized:
            normalized_lines.append(normalized)
    return "\n".join(normalized_lines)


def _is_rollout_completed(record) -> bool:
    progress_payload = _get_rollout_progress_payload(record)
    if not progress_payload:
        return bool(record.result_preview)
    tasks = progress_payload.get("tasks", [])
    return bool(tasks) and all(task.get("status") == "completed" for task in tasks)


def _get_run_skill_payload(record) -> dict | None:
    task_result = record.task_result
    if record.tool_name != "run_skill" or not isinstance(task_result, dict):
        return None
    if not isinstance(task_result.get("skill_name"), str):
        return None
    return task_result


def _is_long_horizon_skill_record(record) -> bool:
    skill_payload = _get_run_skill_payload(record)
    return bool(skill_payload and skill_payload.get("skill_name") == LONG_HORIZON_SKILL_NAME)


def _get_rollout_progress_payload(record) -> dict | None:
    task_result = record.task_result
    if isinstance(task_result, dict) and task_result.get("type") == "rollout_progress":
        return task_result

    skill_payload = _get_run_skill_payload(record)
    structured_response = skill_payload.get("structured_response") if skill_payload else None
    if isinstance(structured_response, dict) and structured_response.get("type") == "rollout_progress":
        return structured_response
    return None


def _get_tool_display_name(record) -> str:
    if _is_long_horizon_skill_record(record):
        return f"{record.tool_name} · {LONG_HORIZON_SKILL_NAME}"
    return record.tool_name


def _has_active_completed_rollout(session_inst) -> bool:
    try:
        if not getattr(session_inst, "initialized", False):
            return False
        for record in reversed(session_inst.agent.get_tool_call_history()):
            if (record.tool_name == "rollout_task" or _is_long_horizon_skill_record(record)) and _is_rollout_completed(
                record
            ):
                return getattr(record, "_rollout_msg_sent", False)
    except Exception:
        return False
    return False


def _maybe_override_organize_msg(session_inst, bot_msg: str) -> str:
    try:
        if not getattr(session_inst, "initialized", False):
            return bot_msg
        for record in reversed(session_inst.agent.get_tool_call_history()):
            if record.tool_name == "organize_and_clean_table" and record.result_preview:
                if getattr(record, "_organize_msg_sent", False):
                    return bot_msg
                record._organize_msg_sent = True
                return ORGANIZE_COMPLETE_MSG
    except Exception:
        return bot_msg
    return bot_msg


def _maybe_override_rollout_msg(session_inst, bot_msg: str) -> str:
    try:
        if not getattr(session_inst, "initialized", False):
            return bot_msg
        for record in reversed(session_inst.agent.get_tool_call_history()):
            if (record.tool_name == "rollout_task" or _is_long_horizon_skill_record(record)) and _is_rollout_completed(
                record
            ):
                if getattr(record, "_rollout_msg_sent", False):
                    return bot_msg
                record._rollout_msg_sent = True
                return ROLLOUT_COMPLETE_MSG
    except Exception:
        return bot_msg
    return bot_msg


class UIState(Enum):
    IDLE = "idle"
    WAITING_INFER = "waiting_infer"
    INFERING = "infering"
    DONE = "done"
    FAILED = "failed"


class UIEvent(Enum):
    USER_INPUT = "user_input"
    INFER_START = "infer_start"
    INFER_SUCCESS = "infer_success"
    INFER_FAIL = "infer_fail"


class UIStateMachine:
    def __init__(self):
        self.state = UIState.IDLE
        self._state_lock = asyncio.Lock()
        self._infer_done = asyncio.Event()
        self._infer_done.set()

    async def send_event(self, event: UIEvent) -> bool:
        async with self._state_lock:
            if event == UIEvent.USER_INPUT and self.state in [UIState.IDLE, UIState.DONE, UIState.FAILED]:
                self.state = UIState.WAITING_INFER
                self._infer_done.clear()
                return True
            if event == UIEvent.INFER_START and self.state == UIState.WAITING_INFER:
                self.state = UIState.INFERING
                return True
            if event == UIEvent.INFER_SUCCESS and self.state == UIState.INFERING:
                self.state = UIState.DONE
                self._infer_done.set()
                return True
            if event == UIEvent.INFER_FAIL and self.state == UIState.INFERING:
                self.state = UIState.FAILED
                self._infer_done.set()
                return True
            return False

    async def wait_inference_done(self) -> None:
        await self._infer_done.wait()

    def get_state_display(self) -> str:
        return {
            UIState.IDLE: "Idle",
            UIState.WAITING_INFER: "Waiting for inference",
            UIState.INFERING: "Inferring",
            UIState.DONE: "Inference completed",
            UIState.FAILED: "Inference failed",
        }.get(self.state, "Unknown")


class Session:
    def __init__(self):
        self.initialized = False
        self._init_lock = asyncio.Lock()
        self._run_once_lock = asyncio.Lock()
        self._run_once_executing = False
        self.config_error: str | None = None
        self.a2d_warning: str | None = None
        self._agent_card: BaseAgentCard | None = None
        self.agent: ImgActAgent | None = None

        try:
            robot_dataloader, self.a2d_warning = _create_robot_dataloader()
            self._agent_card = BaseAgentCard(
                silence=False,
                config=ChatAPIConfig.resolve_runtime_default(),
                service_config_path=str(REPO_ROOT / "src/agent_demo/config/ormcp_services.json"),
                skill_paths=[str(REPO_ROOT / "skills")],
                agent_memory_prompt=ImgActAgentPrompt.init_memory_prompt,
                robot_dataloader=robot_dataloader,
            )
            self.agent = ImgActAgent(agent_card=self._agent_card)
        except Exception as exc:
            self.config_error = str(exc)
            logger.error("LLM configuration error: %s", self.config_error)

    async def ensure_initialized(self) -> None:
        if self.config_error:
            raise RuntimeError(self.config_error)
        if self.agent is None:
            raise RuntimeError("Agent is not available")
        if self.initialized:
            return

        async with self._init_lock:
            if self.initialized:
                return
            await self.agent.init_agent()
            self.initialized = True

    async def _run_agent_once(
        self,
        message: str,
        clear_history: bool = True,
        on_text_delta=None,
        on_status=None,
    ) -> str:
        if self.config_error:
            raise RuntimeError(f"API configuration error: {self.config_error}")

        await self.ensure_initialized()
        assert self.agent is not None

        async with self._run_once_lock:
            self._run_once_executing = True
            try:
                if self.agent.state != ActAgentState.READY:
                    logger.warning("[Session] Agent state is %s before run_once", self.agent.state)

                try:
                    res: TextParam | None = await self.agent.run_once(
                        message,
                        clear_history=clear_history,
                        on_text_delta=on_text_delta,
                        on_status=on_status,
                    )
                    return res.text if res else "No result returned."
                except Exception as exc:
                    error_str = str(exc)
                    if (
                        "messages with role 'tool'" in error_str
                        and "must be a response to a preceeding message with 'tool_calls'" in error_str
                    ):
                        logger.warning("[Session] Cleaning orphaned tool messages and retrying run_once")
                        self.agent.current_task_node._cleanup_orphaned_tool_messages()
                        retry_res: TextParam | None = await self.agent.run_once(
                            message,
                            clear_history=clear_history,
                            on_text_delta=on_text_delta,
                            on_status=on_status,
                        )
                        return retry_res.text if retry_res else "No result returned."
                    logger.error("Error calling agent.run_once: %s", exc)
                    logger.error("Full traceback:\n%s", traceback.format_exc())
                    raise
            finally:
                self._run_once_executing = False

    async def run_once(self, message: str, clear_history: bool = True) -> str:
        return await self._run_agent_once(message, clear_history=clear_history)

    async def run_once_stream(
        self,
        message: str,
        clear_history: bool = True,
    ) -> AsyncGenerator[dict[str, str], None]:
        stream_queue: asyncio.Queue[dict[str, str]] = asyncio.Queue()

        async def emit_text_delta(delta: str) -> None:
            if delta:
                await stream_queue.put({"type": "text_delta", "delta": delta})

        async def emit_status(status: str) -> None:
            if status:
                await stream_queue.put({"type": "status", "text": status})

        async def runner() -> None:
            try:
                final_text = await self._run_agent_once(
                    message,
                    clear_history=clear_history,
                    on_text_delta=emit_text_delta,
                    on_status=emit_status,
                )
                await stream_queue.put({"type": "final", "text": final_text})
            except Exception as exc:
                await stream_queue.put({"type": "error", "text": str(exc)})
            finally:
                await stream_queue.put({"type": "done"})

        runner_task = asyncio.create_task(runner())
        try:
            while True:
                event = await stream_queue.get()
                event_type = event.get("type")
                if event_type == "done":
                    break
                if event_type == "error":
                    raise RuntimeError(event.get("text", "Unknown error"))
                yield event
        finally:
            await runner_task

    async def shutdown(self) -> None:
        agent_shutdown_error: BaseException | None = None
        shutdown_interrupted = False
        try:
            if self.initialized and self.agent is not None:
                shutdown_task = asyncio.create_task(self.agent.shutdown())
                try:
                    await asyncio.shield(shutdown_task)
                except asyncio.CancelledError:
                    shutdown_interrupted = True
                    try:
                        await shutdown_task
                    except BaseException as exc:
                        agent_shutdown_error = exc
                except BaseException as exc:
                    agent_shutdown_error = exc
        finally:
            try:
                if self._agent_card is not None and self._agent_card.robot_dataloader is not None:
                    self._agent_card.robot_dataloader.shutdown()
            except Exception:
                logger.warning("Robot dataloader shutdown reported an error during exit.")
            gc.collect()
            with contextlib.suppress(BaseException):
                await asyncio.sleep(0.1)
            if shutdown_interrupted and agent_shutdown_error is None:
                logger.info("Agent shutdown completed after cancellation was requested.")
            elif isinstance(agent_shutdown_error, asyncio.CancelledError):
                logger.info("Agent shutdown cancellation suppressed during exit.")
            elif agent_shutdown_error is not None:
                logger.warning(
                    "Agent shutdown reported %s during exit: %s",
                    type(agent_shutdown_error).__name__,
                    agent_shutdown_error,
                )


class OlympusTUI:
    COMMANDS = [
        ("/exit", "Exit the TUI"),
        ("/quit", "Exit the TUI"),
        ("/clear", "Clear chat history"),
        ("/help", "Show available commands"),
    ]

    def __init__(self, session: Session | None = None, skill_registry: LocalSkillRegistry | None = None):
        self.console = Console()
        self.session = session or Session()
        self.skill_registry = skill_registry or LocalSkillRegistry(workspace_root=str(REPO_ROOT))
        self.ui_state_machine = UIStateMachine()
        self.user_input_processing = asyncio.Event()
        self.stop_event = asyncio.Event()
        self.chat_history: list[dict[str, str]] = []
        self.status_message = "Starting..."
        self.max_messages = 14
        self.max_tool_groups = 8
        self.input_buffer = ""
        self.input_queue: asyncio.Queue[str] = asyncio.Queue()
        self._stdin_fd: int | None = None
        self._stdin_attrs: list | None = None
        self._escape_sequence_active = False
        self._escape_sequence_buffer = ""
        self._chat_scroll_offset = 0
        self._chat_auto_scroll = True
        self._completion_menu_index = 0
        self._skill_completion_state: dict[str, object] | None = None
        self._installed_signals: list[signal.Signals] = []

    def _append_chat_message(self, role: str, content: str) -> None:
        if role == "assistant":
            compact_content = _compact_chat_message(content, role)
            if not compact_content:
                return
            signature = _chat_message_signature(_signature_chat_message(compact_content, role))
            for previous in reversed(self.chat_history):
                if previous["role"] != "assistant":
                    continue
                previous_signature = _chat_message_signature(
                    _signature_chat_message(_compact_chat_message(previous["content"], "assistant"), "assistant")
                )
                if previous_signature == signature:
                    return
                break
        self.chat_history.append({"role": role, "content": content})
        if self._chat_auto_scroll:
            self._chat_scroll_offset = 0

    def _update_last_assistant_message(self, content: str) -> None:
        for message in reversed(self.chat_history):
            if message["role"] == "assistant":
                message["content"] = content
                if self._chat_auto_scroll:
                    self._chat_scroll_offset = 0
                return
        self.chat_history.append({"role": "assistant", "content": content})
        if self._chat_auto_scroll:
            self._chat_scroll_offset = 0

    def _follow_chat_tail(self) -> None:
        self._chat_auto_scroll = True
        self._chat_scroll_offset = 0

    def _estimate_chat_viewport(self) -> tuple[int, int]:
        console_width = max(self.console.size.width, 40)
        console_height = max(self.console.size.height, 12)
        input_size = 3 if self._get_completion_state() is None else 8
        body_height = max(console_height - input_size, 3)
        chat_width = max((console_width * 3) // 5 - 4, 20)
        chat_height = max(body_height - 2, 3)
        return chat_width, chat_height

    def _build_chat_lines(self, viewport_width: int) -> list[tuple[str, str]]:
        chat_lines: list[tuple[str, str]] = []
        for message in _collapse_chat_entries(self.chat_history, None):
            is_user = message["role"] == "user"
            role_label = "You" if is_user else "Agent"
            role_style = "bold green" if is_user else "bold yellow"
            indent = " " * (len(role_label) + 2)
            logical_lines = str(message["content"]).splitlines() or [""]

            for line_index, logical_line in enumerate(logical_lines):
                initial_indent = f"{role_label}: " if line_index == 0 else indent
                wrapped_lines = textwrap.wrap(
                    logical_line,
                    width=viewport_width,
                    initial_indent=initial_indent,
                    subsequent_indent=indent,
                    replace_whitespace=False,
                    drop_whitespace=False,
                    break_long_words=False,
                    break_on_hyphens=False,
                )
                if not wrapped_lines:
                    wrapped_lines = [initial_indent.rstrip()]
                for wrapped_line in wrapped_lines:
                    chat_lines.append((wrapped_line, role_style))

            if int(message["repeat"]) > 1:
                chat_lines.append((f"{indent}(repeated x{message['repeat']})", "dim"))
            chat_lines.append(("", ""))

        if chat_lines and chat_lines[-1][0] == "":
            chat_lines.pop()
        return chat_lines

    def _get_chat_view(self) -> tuple[list[tuple[str, str]], int, int]:
        viewport_width, viewport_height = self._estimate_chat_viewport()
        chat_lines = self._build_chat_lines(viewport_width)
        total_lines = len(chat_lines)
        max_offset = max(total_lines - viewport_height, 0)
        if self._chat_auto_scroll:
            self._chat_scroll_offset = 0
        else:
            self._chat_scroll_offset = min(self._chat_scroll_offset, max_offset)
        start_index = max(total_lines - viewport_height - self._chat_scroll_offset, 0)
        end_index = total_lines - self._chat_scroll_offset if self._chat_scroll_offset else total_lines
        visible_lines = chat_lines[start_index:end_index]
        return visible_lines, total_lines, viewport_height

    def _scroll_chat(self, delta_lines: int) -> None:
        _, viewport_height = self._estimate_chat_viewport()
        total_lines = len(self._build_chat_lines(self._estimate_chat_viewport()[0]))
        max_offset = max(total_lines - viewport_height, 0)
        if max_offset == 0:
            self._chat_scroll_offset = 0
            self._chat_auto_scroll = True
            return

        self._chat_scroll_offset = max(0, min(self._chat_scroll_offset + delta_lines, max_offset))
        self._chat_auto_scroll = self._chat_scroll_offset == 0

    def _scroll_chat_page(self, direction: int) -> None:
        _, viewport_height = self._estimate_chat_viewport()
        page_step = max(viewport_height - 2, CHAT_SCROLL_STEP)
        self._scroll_chat(direction * page_step)

    def _scroll_chat_to_top(self) -> None:
        _, viewport_height = self._estimate_chat_viewport()
        total_lines = len(self._build_chat_lines(self._estimate_chat_viewport()[0]))
        self._chat_scroll_offset = max(total_lines - viewport_height, 0)
        self._chat_auto_scroll = self._chat_scroll_offset == 0

    def _parse_mouse_event(self, sequence: str) -> tuple[int, int, int] | None:
        sgr_match = re.fullmatch(r"\x1b\[<(\d+);(\d+);(\d+)([mM])", sequence)
        if sgr_match:
            return int(sgr_match.group(1)), int(sgr_match.group(2)), int(sgr_match.group(3))

        urxvt_match = re.fullmatch(r"\x1b\[(\d+);(\d+);(\d+)M", sequence)
        if urxvt_match:
            return int(urxvt_match.group(1)), int(urxvt_match.group(2)), int(urxvt_match.group(3))

        return None

    def _handle_mouse_event(self, sequence: str) -> bool:
        mouse_event = self._parse_mouse_event(sequence)
        if mouse_event is None:
            return False

        button_code = mouse_event[0] & ~MOUSE_EVENT_MODIFIER_MASK
        if button_code == MOUSE_SCROLL_UP_BUTTON:
            self._scroll_chat(CHAT_SCROLL_STEP)
            return True
        if button_code == MOUSE_SCROLL_DOWN_BUTTON:
            self._scroll_chat(-CHAT_SCROLL_STEP)
            return True
        return False

    def _is_partial_escape_sequence(self, sequence: str) -> bool:
        known_sequences = {
            "\x1b[A",
            "\x1b[B",
            "\x1b[C",
            "\x1b[D",
            "\x1bOA",
            "\x1bOB",
            "\x1bOC",
            "\x1bOD",
            "\x1b[5~",
            "\x1b[6~",
            "\x1b[H",
            "\x1b[F",
            "\x1bOH",
            "\x1bOF",
        }
        if any(known.startswith(sequence) for known in known_sequences):
            return True
        if sequence.startswith("\x1b[") and len(sequence) <= 32:
            return re.fullmatch(r"\x1b\[(?:<)?[\d;]*[mM]?", sequence) is not None
        return False

    def _request_stop(self, status_message: str | None = None) -> None:
        if status_message:
            self.status_message = status_message
        self.input_buffer = ""
        with contextlib.suppress(asyncio.QueueFull):
            self.input_queue.put_nowait("")
        if self.stop_event.is_set():
            return
        self.stop_event.set()

    async def _sleep_until_stop(self, timeout: float) -> bool:
        try:
            await asyncio.wait_for(self.stop_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return self.stop_event.is_set()

    async def _stop_background_tasks(self, tasks: list[asyncio.Task]) -> None:
        if not tasks:
            return
        logger.info("[TUI] Stopping %d background task(s).", len(tasks))
        try:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5)
        except asyncio.TimeoutError:
            logger.warning("[TUI] Background tasks did not stop in time; cancelling.")
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("[TUI] Background tasks stopped.")

    async def _arm_auto_inference(self) -> None:
        if self.stop_event.is_set() or self.user_input_processing.is_set():
            return
        if self.ui_state_machine.state in {UIState.IDLE, UIState.DONE, UIState.FAILED}:
            await self.ui_state_machine.send_event(UIEvent.USER_INPUT)

    async def _read_input(self) -> str:
        return await self.input_queue.get()

    def _get_active_skill_token(self) -> tuple[int, int, str] | None:
        match = re.search(r"(^|\s)(\$[A-Za-z0-9-]*)$", self.input_buffer)
        if match is None:
            return None
        return match.start(2), match.end(2), match.group(2)

    def _get_completion_state(self) -> dict[str, object] | None:
        stripped_buffer = self.input_buffer.strip()
        if self.input_buffer.startswith("/") and " " not in stripped_buffer:
            # Slash commands never reuse skill completion state.
            self._skill_completion_state = None
            if stripped_buffer == "/":
                items = [
                    {"label": name, "description": description, "replacement": name}
                    for name, description in self.COMMANDS
                ]
            else:
                items = [
                    {"label": name, "description": description, "replacement": name}
                    for name, description in self.COMMANDS
                    if name.startswith(stripped_buffer)
                ]
            if not items:
                return None
            return {
                "kind": "command",
                "start": 0,
                "end": len(self.input_buffer),
                "token": stripped_buffer,
                "items": items,
                "hint": "Use Up/Down to select, Tab or Right to fill, Enter to execute",
                "selection_status": "Command selected. Press Enter to execute.",
            }

        skill_token = self._get_active_skill_token()
        if skill_token is None:
            # No active $skill token => clear any cached skill completion state.
            self._skill_completion_state = None
            return None

        start, end, token = skill_token

        # For skill completions we keep the suggestion list stable while the
        # user navigates with arrow keys, so moving Up/Down cycles through the
        # same items even after a suggestion has been previewed in the input
        # buffer.
        if self._skill_completion_state is not None and self._skill_completion_state.get("kind") == "skill":
            items = self._skill_completion_state["items"]
        else:
            skills = self.skill_registry.suggest(token[1:])
            if not skills:
                self._skill_completion_state = None
                return None

            items = [
                {
                    "label": f"${skill.name}",
                    "description": (
                        (skill.description[:120] + "...") if len(skill.description) > 120 else skill.description
                    ),
                    "replacement": f"${skill.name}",
                }
                for skill in skills
            ]

        state = {
            "kind": "skill",
            "start": start,
            "end": end,
            "token": token,
            "items": items,
            "hint": "Use Up/Down to select, Tab or Right to fill, Enter to send",
            "selection_status": "Skill selected. Press Enter to send.",
        }
        self._skill_completion_state = state
        return state

    def _sync_completion_menu_index(self) -> None:
        state = self._get_completion_state()
        if state is None:
            self._completion_menu_index = 0
            return
        self._completion_menu_index %= len(state["items"])

    def _move_completion_selection(self, delta: int) -> None:
        state = self._get_completion_state()
        if state is None:
            self._completion_menu_index = 0
            return
        # Always wrap within the number of known items so that repeated
        # Up/Down presses cycle through the same suggestions, even when the
        # underlying input buffer has been updated to show a preview.
        self._completion_menu_index = (self._completion_menu_index + delta) % len(state["items"])

    def _apply_completion_selection(self) -> bool:
        state = self._get_completion_state()
        if state is None:
            return False
        # Clamp the index to the current items length in case the list of
        # suggestions changed.
        self._completion_menu_index %= len(state["items"])
        item = state["items"][self._completion_menu_index]
        start = state["start"]
        end = state["end"]
        self.input_buffer = f"{self.input_buffer[:start]}{item['replacement']}{self.input_buffer[end:]}"
        return True

    def _completion_is_exact_match(self) -> bool:
        state = self._get_completion_state()
        if state is None:
            return False
        token = self.input_buffer[state["start"] : state["end"]].strip()
        return token in {item["replacement"] for item in state["items"]}

    def _handle_escape_sequence(self, sequence: str) -> None:
        if sequence in {"\x1b[A", "\x1bOA"}:
            if self._get_completion_state() is not None:
                # Move selection up and preview the selected completion in the input
                # buffer so that the user sees the currently focused item.
                self._move_completion_selection(-1)
                self._apply_completion_selection()
            else:
                self._scroll_chat(CHAT_SCROLL_STEP)
        elif sequence in {"\x1b[B", "\x1bOB"}:
            if self._get_completion_state() is not None:
                # Move selection down and preview the selected completion.
                self._move_completion_selection(1)
                self._apply_completion_selection()
            else:
                self._scroll_chat(-CHAT_SCROLL_STEP)
        elif sequence in {"\x1b[C", "\x1bOC"}:
            # Right arrow keeps the existing behavior of applying the current
            # selection without changing the index.
            self._apply_completion_selection()
        elif sequence == "\x1b[5~":
            self._scroll_chat_page(1)
        elif sequence == "\x1b[6~":
            self._scroll_chat_page(-1)
        elif sequence in {"\x1b[H", "\x1bOH"}:
            self._scroll_chat_to_top()
        elif sequence in {"\x1b[F", "\x1bOF"}:
            self._follow_chat_tail()
        else:
            self._handle_mouse_event(sequence)

    def _consume_stdin_bytes(self) -> None:
        if self._stdin_fd is None or self.stop_event.is_set():
            return

        try:
            data = os.read(self._stdin_fd, 1024)
        except OSError as exc:
            self._request_stop(f"Input error: {exc}")
            return

        if not data:
            self._request_stop("Input closed.")
            return

        for char in data.decode(errors="ignore"):
            if self._escape_sequence_active:
                self._escape_sequence_buffer += char
                # Handle common cursor key escape sequences explicitly so that
                # raw characters like "[" or "A/B/C/D" are not appended to
                # the input buffer when pressing arrow keys.
                if self._escape_sequence_buffer in {
                    "\x1b[A",
                    "\x1b[B",
                    "\x1b[C",
                    "\x1b[D",
                    "\x1bOA",
                    "\x1bOB",
                    "\x1bOC",
                    "\x1bOD",
                    "\x1b[5~",
                    "\x1b[6~",
                    "\x1b[H",
                    "\x1b[F",
                    "\x1bOH",
                    "\x1bOF",
                } or self._parse_mouse_event(self._escape_sequence_buffer):
                    self._escape_sequence_active = False
                    self._handle_escape_sequence(self._escape_sequence_buffer)
                    self._escape_sequence_buffer = ""
                    continue

                if not self._is_partial_escape_sequence(self._escape_sequence_buffer):
                    self._escape_sequence_active = False
                    self._escape_sequence_buffer = ""
                continue

            if char == "\x1b":
                self._escape_sequence_active = True
                self._escape_sequence_buffer = char
                continue

            if char in {"\r", "\n"}:
                completion_state = self._get_completion_state()
                line = self.input_buffer.strip()
                if completion_state is not None and not self._completion_is_exact_match():
                    if self._apply_completion_selection():
                        self.status_message = str(completion_state["selection_status"])
                    continue
                self.input_buffer = ""
                self._completion_menu_index = 0
                if line:
                    self.input_queue.put_nowait(line)
                continue

            if char == "\x04":
                self._request_stop("Input closed.")
                return

            if char == "\x03":
                self._request_stop("Shutting down...")
                return

            if char == "\t":
                completion_state = self._get_completion_state()
                if completion_state is not None and self._apply_completion_selection():
                    self.status_message = str(completion_state["selection_status"])
                continue

            if char in {"\x08", "\x7f"}:
                self.input_buffer = self.input_buffer[:-1]
                # Backspace changes the active token; clear cached skill
                # completion state so that suggestions are recomputed for the
                # new prefix.
                self._skill_completion_state = None
                self._sync_completion_menu_index()
                continue

            if char.isprintable():
                self.input_buffer += char
                # Typing changes the prefix; clear cached skill completion
                # state so that we recompute suggestions based on the current
                # token.
                self._skill_completion_state = None
                self._sync_completion_menu_index()

    def _install_stdin_reader(self) -> None:
        if not sys.stdin.isatty():
            self.status_message = "stdin is not a TTY; interactive input is unavailable."
            return

        self._stdin_fd = sys.stdin.fileno()
        self._stdin_attrs = termios.tcgetattr(self._stdin_fd)
        tty.setcbreak(self._stdin_fd)
        attrs = termios.tcgetattr(self._stdin_fd)
        attrs[3] &= ~(termios.ECHO | termios.ISIG)
        termios.tcsetattr(self._stdin_fd, termios.TCSANOW, attrs)
        with contextlib.suppress(Exception):
            sys.stdout.write(MOUSE_TRACKING_ENABLE)
            sys.stdout.flush()
        asyncio.get_running_loop().add_reader(self._stdin_fd, self._consume_stdin_bytes)

    def _restore_stdin_reader(self) -> None:
        if self._stdin_fd is None:
            return

        loop = asyncio.get_running_loop()
        with contextlib.suppress(Exception):
            loop.remove_reader(self._stdin_fd)
        if self._stdin_attrs is not None:
            with contextlib.suppress(Exception):
                termios.tcsetattr(self._stdin_fd, termios.TCSANOW, self._stdin_attrs)
        with contextlib.suppress(Exception):
            sys.stdout.write(MOUSE_TRACKING_DISABLE)
            sys.stdout.flush()
        self._stdin_fd = None
        self._stdin_attrs = None

    def _prepare_message_for_agent(self, message: str) -> str | None:
        services = None
        session_agent = getattr(self.session, "agent", None)
        if session_agent is not None:
            service_manager = getattr(session_agent, "service_manager", None)
            services = getattr(service_manager, "_services_register_list", None)

        prepared_message = prepare_agent_message(message, self.skill_registry, services=services)
        if not prepared_message.expanded_message.requested_skills:
            return message

        if prepared_message.error_message:
            self._append_chat_message(
                "assistant",
                prepared_message.error_message,
            )
            self.status_message = prepared_message.status_message or "Requested local skill not found."
            return None

        self.status_message = prepared_message.status_message or self.status_message
        return prepared_message.message

    async def _handle_user_message(self, message: str) -> None:
        if self.ui_state_machine.state != UIState.WAITING_INFER:
            if not await self.ui_state_machine.send_event(UIEvent.USER_INPUT):
                await self.ui_state_machine.wait_inference_done()
                if self.ui_state_machine.state != UIState.WAITING_INFER:
                    await self.ui_state_machine.send_event(UIEvent.USER_INPUT)

        self.user_input_processing.set()
        self._follow_chat_tail()
        self._append_chat_message("user", message)
        try:
            agent_message = self._prepare_message_for_agent(message)
            if agent_message is None:
                return
            if not await self.ui_state_machine.send_event(UIEvent.INFER_START):
                self.status_message = "Input queued while agent was not ready."
                return

            assistant_text = ""
            status_text = "Thinking..."
            final_text = ""
            self._append_chat_message("assistant", "_Thinking..._")

            run_once_stream = getattr(self.session, "run_once_stream", None)
            if callable(run_once_stream):
                async for event in run_once_stream(agent_message):
                    if event["type"] == "status":
                        status_text = event.get("text", status_text) or status_text
                        self.status_message = status_text
                    elif event["type"] == "text_delta":
                        assistant_text += event.get("delta", "")
                    elif event["type"] == "final":
                        final_text = event.get("text", "") or ""

                    streaming_content = assistant_text or "_Thinking..._"
                    if status_text and status_text != "Thinking...":
                        streaming_content = f"{streaming_content}\n\n[status] {status_text}"
                    self._update_last_assistant_message(streaming_content)
            else:
                final_text = await self.session.run_once(agent_message)

            final_text = final_text or assistant_text or "No result returned."
            final_text = _maybe_override_organize_msg(self.session, final_text)
            final_text = _maybe_override_rollout_msg(self.session, final_text)
            self._update_last_assistant_message(format_response_text(final_text))
            await self.ui_state_machine.send_event(UIEvent.INFER_SUCCESS)
            self.status_message = "Last user message processed."
        except asyncio.CancelledError:
            self.status_message = "Stopping..."
            raise
        except Exception as exc:
            if self.stop_event.is_set():
                self.status_message = "Stopping..."
                return
            await self.ui_state_machine.send_event(UIEvent.INFER_FAIL)
            self._append_chat_message("assistant", f"Error: {type(exc).__name__}: {exc}")
            self.status_message = "User message failed."
        finally:
            self.user_input_processing.clear()
            if not self.stop_event.is_set():
                await self._arm_auto_inference()

    async def _handle_slash_command(self, command: str) -> None:
        if command in {"/exit", "/quit"}:
            self._request_stop("Shutting down...")
            return

        if command == "/clear":
            self.chat_history.clear()
            self._follow_chat_tail()
            self.status_message = "Chat history cleared."
            return

        if command == "/help":
            skills = self.skill_registry.list_skills()
            skill_text = ", ".join(f"${skill.name}" for skill in skills[:8]) or "(none found)"
            if len(skills) > 8:
                skill_text += ", ..."
            help_text = "\n".join(f"{name}: {description}" for name, description in self.COMMANDS)
            help_text += (
                f"\n\nLocal skills:\nUse $skill-name in your message to attach a local skill.\nAvailable: {skill_text}"
            )
            self._append_chat_message("assistant", help_text)
            self.status_message = "Command help shown."
            return

        self.status_message = f"Unknown command: {command}"

    def _build_state_panel(self) -> Panel:
        agent_state = "Not initialized"
        current_tool = "-"
        tool_status = "-"
        if self.session.initialized:
            try:
                agent_state = self.session.agent.state.name
            except Exception:
                agent_state = "Unknown"
            try:
                tool_history = self.session.agent.get_tool_call_history()
            except Exception:
                tool_history = []
            if tool_history:
                last = tool_history[-1]
                current_tool = f"{last.service_name}.{last.tool_name} (#{last.step_index})"
                tool_status = "Completed" if last.result_preview else "Running"
            elif agent_state == ActAgentState.ACT.name:
                tool_status = "Waiting for tool list"

        note = "Ready"
        if self.user_input_processing.is_set():
            note = "Processing user input; auto inference paused"
        elif self.ui_state_machine.state == UIState.IDLE:
            note = "Waiting for the first submitted message"
        elif self.ui_state_machine.state == UIState.WAITING_INFER:
            note = "Auto inference checks every 5 seconds"

        rows = Table.grid(expand=True)
        rows.add_column(style="cyan", ratio=1)
        rows.add_column(ratio=2)
        rows.add_row("UI State", self.ui_state_machine.get_state_display())
        rows.add_row("Executing", "Yes" if self.session._run_once_executing else "No")
        rows.add_row("Agent", agent_state)
        rows.add_row("Tool", current_tool)
        rows.add_row("Tool Status", tool_status)
        if getattr(self.session, "config_error", None):
            rows.add_row("API", str(self.session.config_error))
        if getattr(self.session, "a2d_warning", None):
            rows.add_row("A2D", str(self.session.a2d_warning))
        rows.add_row("Note", note)
        rows.add_row("Status", self.status_message)
        return Panel(rows, title="Status", border_style="cyan")

    def _build_status_lines(self) -> str:
        agent_state = "Not initialized"
        current_tool = "-"
        tool_status = "-"
        if self.session.initialized:
            try:
                agent_state = self.session.agent.state.name
            except Exception:
                agent_state = "Unknown"
            try:
                tool_history = self.session.agent.get_tool_call_history()
            except Exception:
                tool_history = []
            if tool_history:
                last = tool_history[-1]
                current_tool = f"{last.service_name}.{last.tool_name} (#{last.step_index})"
                tool_status = "Completed" if last.result_preview else "Running"
            elif agent_state == ActAgentState.ACT.name:
                tool_status = "Waiting for tool list"

        note = "Ready"
        if self.user_input_processing.is_set():
            note = "Processing user input; auto inference paused"
        elif self.ui_state_machine.state == UIState.IDLE:
            note = "Waiting for the first submitted message"
        elif self.ui_state_machine.state == UIState.WAITING_INFER:
            note = "Auto inference checks every 5 seconds"

        lines = [
            f"UI State: {self.ui_state_machine.get_state_display()}",
            f"Executing: {'Yes' if self.session._run_once_executing else 'No'}",
            f"Agent: {agent_state}",
            f"Tool: {current_tool}",
            f"Tool Status: {tool_status}",
            f"Note: {note}",
            f"Status: {self.status_message}",
        ]
        if getattr(self.session, "a2d_warning", None):
            lines.insert(5, f"A2D: {self.session.a2d_warning}")
        return "\n".join(lines)

    def _summarize_task_result(self, record) -> str | None:
        task_result = record.task_result
        if not task_result:
            return None

        if record.tool_name == "detect_tasks_from_image" and "categories" in task_result:
            parts = []
            for category in task_result.get("categories", [])[:3]:
                task_names = ", ".join(task.get("task_name", "unknown") for task in category.get("tasks", [])[:3])
                parts.append(f"{category.get('category', 'unknown')}: {task_names or 'no tasks'}")
            summary = task_result.get("summary")
            if summary:
                parts.append(f"summary: {summary}")
            return "\n".join(parts)

        progress_payload = task_result
        if _is_long_horizon_skill_record(record):
            progress_payload = _get_rollout_progress_payload(record) or {}

        if progress_payload.get("type") in {"organize_progress", "rollout_progress"}:
            tasks = progress_payload.get("tasks", [])
            completed = sum(1 for task in tasks if task.get("status") == "completed")
            preview = [f"{task.get('status', '?')}: {task.get('description', 'unknown')}" for task in tasks[:4]]
            return f"{completed}/{len(tasks)} completed\n" + "\n".join(preview)

        if _is_long_horizon_skill_record(record):
            response_text = task_result.get("response", "")
            if isinstance(response_text, str) and response_text.strip():
                first_line = response_text.strip().splitlines()[0]
                return first_line[:160] + ("..." if len(first_line) > 160 else "")

        return None

    def _build_tool_flow_panel(self) -> Panel:
        if not self.session.initialized:
            return Panel("Waiting for agent initialization...", title="Tool Flow", border_style="magenta")

        try:
            history = self.session.agent.get_tool_call_history()
        except Exception:
            history = []

        if not history:
            return Panel("Waiting for tool calls...", title="Tool Flow", border_style="magenta")

        grouped: dict[str, list] = {}
        for record in history[-32:]:
            grouped.setdefault(record.service_name, []).append(record)

        blocks = []
        service_items = list(grouped.items())[-self.max_tool_groups :]
        for service_name, records in service_items:
            service_table = Table.grid(expand=True)
            service_table.add_column(ratio=3)
            service_table.add_column(ratio=2)
            service_table.add_column(ratio=5)

            previous_tool = None
            repeat_count = 0
            for record in records[-6:]:
                repeat_count = repeat_count + 1 if record.tool_name == previous_tool else 1
                previous_tool = record.tool_name
                label = f"#{record.step_index} {_get_tool_display_name(record)}"
                if repeat_count > 1:
                    label += f" x{repeat_count}"
                tool_status = "Completed" if record.result_preview else "Running"
                summary = self._summarize_task_result(record)
                details = f"status: {tool_status}"
                if summary:
                    compact_summary = summary[:120] + ("..." if len(summary) > 120 else "")
                    details += f"\nsummary: {compact_summary}"
                service_table.add_row(
                    Text(label, style="bold"),
                    record.timestamp.split(" ")[-1],
                    details,
                )

            blocks.append(Panel(service_table, title=service_name, border_style="magenta"))

        return Panel(Group(*blocks), title="Tool Flow", border_style="magenta")

    def _build_chat_panel(self) -> Panel:
        if not self.chat_history:
            return Panel(
                "Type below, then press Enter to send.\nCommands start with /. Skills start with $. Example: $local-skill-demo",
                title="Chat",
                border_style="green",
                subtitle="Wheel/PgUp/PgDn/Home/End to scroll",
            )

        transcript = Text()
        visible_lines, total_lines, viewport_height = self._get_chat_view()
        for line, style in visible_lines:
            transcript.append(line, style=style or None)
            transcript.append("\n")

        if transcript.plain.endswith("\n"):
            transcript.rstrip()

        scroll_state = "Bottom"
        if total_lines > viewport_height:
            if self._chat_scroll_offset > 0:
                start_line = max(total_lines - viewport_height - self._chat_scroll_offset + 1, 1)
                end_line = min(start_line + len(visible_lines) - 1, total_lines)
                scroll_state = f"Lines {start_line}-{end_line}/{total_lines}"
            else:
                scroll_state = f"Bottom {max(total_lines - viewport_height + 1, 1)}-{total_lines}/{total_lines}"

        return Panel(
            transcript,
            title="Chat",
            border_style="green",
            subtitle=f"{scroll_state} | Wheel/PgUp/PgDn/Home/End to scroll",
        )

    def _build_input_panel(self) -> Panel:
        prompt = Text("> ", style="bold cyan")
        if self.input_buffer:
            prompt.append(self.input_buffer)
        else:
            prompt.append("Type here, or use /commands and $skills", style="dim")
        prompt.append("|", style="bold cyan")
        completion_state = self._get_completion_state()
        if completion_state is None:
            return Panel(prompt, title="Input", border_style="blue")

        self._sync_completion_menu_index()
        menu = Table.grid(expand=True)
        menu.add_column(ratio=2)
        menu.add_column(ratio=5)
        for index, item in enumerate(completion_state["items"]):
            style = "bold black on cyan" if index == self._completion_menu_index else "white"
            menu.add_row(Text(str(item["label"]), style=style), Text(str(item["description"]), style=style))

        hint = Text(str(completion_state["hint"]), style="dim")
        return Panel(Group(prompt, menu, hint), title="Input", border_style="blue")

    def render(self):
        layout = Layout()
        input_size = 3 if self._get_completion_state() is None else 8
        layout.split_column(Layout(name="body", ratio=8), Layout(name="input", size=input_size))
        layout["body"].split_row(Layout(name="chat", ratio=3), Layout(name="side", ratio=2))
        layout["side"].split_column(Layout(name="status", ratio=1), Layout(name="tools", ratio=2))
        layout["chat"].update(self._build_chat_panel())
        layout["status"].update(self._build_state_panel())
        layout["tools"].update(self._build_tool_flow_panel())
        layout["input"].update(self._build_input_panel())
        return layout

    async def _input_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                line = (await self._read_input()).strip()
            except asyncio.CancelledError:
                break
            except EOFError:
                self._request_stop("Input closed.")
                break
            except Exception as exc:
                self._request_stop(f"Input error: {exc}")
                break

            if not line:
                continue
            if line.startswith("/"):
                await self._handle_slash_command(line.lower())
                if self.stop_event.is_set():
                    break
                continue
            await self._handle_user_message(line)

    async def _auto_inference_loop(self) -> None:
        while not self.stop_event.is_set():
            if await self._sleep_until_stop(5):
                break
            if self.ui_state_machine.state != UIState.WAITING_INFER:
                continue
            if self.user_input_processing.is_set():
                continue
            if not self.session.initialized:
                continue

            try:
                dataloader = self.session._agent_card.robot_dataloader
                if dataloader is None:
                    continue
                a2d_data = await dataloader.get_latest_concatenate_image_base64()
                if a2d_data is None:
                    self.status_message = "Auto inference skipped: no image data."
                    continue

                await self.session.agent.add_img_data_to_memory()
                if self.user_input_processing.is_set() or self.ui_state_machine.state != UIState.WAITING_INFER:
                    continue

                if not await self.ui_state_machine.send_event(UIEvent.INFER_START):
                    continue

                bot_msg = await self.session.run_once(AUTO_INFERENCE_PROMPT, clear_history=False)
                bot_msg = _maybe_override_organize_msg(self.session, bot_msg)
                bot_msg = _maybe_override_rollout_msg(self.session, bot_msg)
                formatted = format_response_text(bot_msg)
                if not _has_active_completed_rollout(self.session):
                    self._append_chat_message("assistant", formatted)
                await self.ui_state_machine.send_event(UIEvent.INFER_SUCCESS)
                self.status_message = "Auto inference completed."
            except asyncio.CancelledError:
                break
            except Exception as exc:
                if self.stop_event.is_set():
                    break
                logger.warning("Auto inference failed: %s", exc, exc_info=True)
                await self.ui_state_machine.send_event(UIEvent.INFER_FAIL)
                self.status_message = f"Auto inference failed: {type(exc).__name__}"
            finally:
                if not self.stop_event.is_set():
                    await self._arm_auto_inference()

    async def _refresh_loop(self, live: Live) -> None:
        while not self.stop_event.is_set():
            live.update(self.render())
            if await self._sleep_until_stop(0.2):
                break
        live.update(self.render())

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._request_stop, "Shutting down...")
                self._installed_signals.append(sig)
            except NotImplementedError:
                logger.warning("Signal handlers are not supported on this platform")

    def _remove_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in self._installed_signals:
            with contextlib.suppress(Exception):
                loop.remove_signal_handler(sig)
        self._installed_signals.clear()

    async def run(self) -> None:
        tasks: list[asyncio.Task] = []
        feishu_long_connection = None
        self._install_signal_handlers()
        self.status_message = "Initializing agent..."
        await self.session.ensure_initialized()
        if self.session.a2d_warning:
            self.status_message = "Ready for input in chat-only mode. Type text, use /commands, or attach $skills."
        else:
            self.status_message = "Ready for input. Type text, use /commands, or attach $skills."
        self._install_stdin_reader()

        def _format_bot_msg_for_feishu(raw: str) -> str:
            msg = _maybe_override_organize_msg(self.session, raw)
            msg = _maybe_override_rollout_msg(self.session, msg)
            return format_response_text(msg)

        def _build_status_for_feishu() -> str:
            return self._build_status_lines()

        async def _get_latest_snapshot_bytes() -> bytes | None:
            try:
                dataloader = self.session._agent_card.robot_dataloader
                if dataloader is None:
                    return None
                a2d_data = await dataloader.get_latest_concatenate_image_base64()
                if a2d_data and a2d_data.concatenated_image_base64:
                    return base64.b64decode(a2d_data.concatenated_image_base64)
            except Exception as exc:
                logger.warning("[FeishuLongConnection] Failed to get snapshot: %s", exc, exc_info=True)
            return None

        try:
            feishu_long_connection = start_feishu_long_connection(
                session_inst=self.session,
                ui_state_machine=self.ui_state_machine,
                user_input_processing=self.user_input_processing,
                chat_history=self.chat_history,
                format_bot_msg=_format_bot_msg_for_feishu,
                build_status_text=_build_status_for_feishu,
                get_image_bytes=_get_latest_snapshot_bytes,
                source_name="tui",
            )
        except Exception as exc:
            logger.warning("[FeishuLongConnection] Startup skipped: %s", exc, exc_info=True)

        try:
            with Live(self.render(), console=self.console, screen=True, refresh_per_second=10) as live:
                tasks = [
                    asyncio.create_task(self._input_loop()),
                    asyncio.create_task(self._auto_inference_loop()),
                    asyncio.create_task(self._refresh_loop(live)),
                ]
                await self.stop_event.wait()
        finally:
            logger.info("[TUI] Finalizing run loop.")
            self._request_stop("Shutting down...")
            self._restore_stdin_reader()
            self._remove_signal_handlers()
            await self._stop_background_tasks(tasks)
            if feishu_long_connection is not None:
                await feishu_long_connection.aclose()
            logger.info("[TUI] Starting session shutdown.")
            await self.session.shutdown()
            logger.info("[TUI] Session shutdown finished.")


async def main() -> None:
    app = OlympusTUI()
    await app.run()
    logger.info("[TUI] main() completed.")


def _flush_and_exit(code: int = 0) -> None:
    with contextlib.suppress(Exception):
        sys.stdout.flush()
    with contextlib.suppress(Exception):
        sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, asyncio.CancelledError):
        _flush_and_exit(0)
    else:
        _flush_and_exit(0)
