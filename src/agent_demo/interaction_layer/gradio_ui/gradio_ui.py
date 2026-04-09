from agent_demo.interaction_layer.auto_inference_prompt import AUTO_INFERENCE_PROMPT
from agent_demo.agent_layer.agent_components.agent_tools.local_skill_registry import LocalSkillRegistry
from agent_demo.types.agent_types import ChatAPIConfig, BaseAgentCard, TextParam
from agent_demo.agent_layer.agent_core import ImgActAgent
from agent_demo.agent_layer.agent_prompt import ImgActAgentPrompt
from agent_demo.agent_layer.agent_core.img_act_agent.img_act_agent import ToolCallRecord
from agent_demo.common.response_formatter import format_response_text
from agent_demo.common.root_logger import setup_root_logging
from agent_demo.machine_layer.dataloader_factory import create_robot_dataloader
from pathlib import Path
import traceback
import logging
import whisper
import gradio as gr
import asyncio
import webbrowser
import base64
import json
import time
import re
from typing import Any
from enum import Enum
from html import escape
import plotly.graph_objects as go
import os

from agent_demo.interaction_layer.feishu_long_connection import start_feishu_long_connection
from agent_demo.interaction_layer.local_skill_support import PreparedAgentMessage, prepare_agent_message
from typing import AsyncGenerator

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[4]
setup_root_logging(default_log_path="./applog/")


# Fixed message for organize_and_clean_table completion
ORGANIZE_COMPLETE_MSG = (
    "The task to organize and clean the table has been successfully completed. "
    "All four sub-tasks and their reset steps were executed without issues."
)

# Fixed message for rollout_task completion
ROLLOUT_COMPLETE_MSG = (
    "The rollout task has been successfully completed. "
    "All steps (forward, reset, reverse, reset) were executed without issues."
)
LONG_HORIZON_SKILL_NAME = "long-horizon-execution"


def _create_robot_dataloader():
    return create_robot_dataloader()


def _get_run_skill_payload(record: ToolCallRecord) -> dict | None:
    task_result = record.task_result
    if record.tool_name != "run_skill" or not isinstance(task_result, dict):
        return None
    if not isinstance(task_result.get("skill_name"), str):
        return None
    return task_result


def _is_long_horizon_skill_record(record: ToolCallRecord) -> bool:
    skill_payload = _get_run_skill_payload(record)
    return bool(skill_payload and skill_payload.get("skill_name") == LONG_HORIZON_SKILL_NAME)


def _get_rollout_progress_payload(record: ToolCallRecord) -> dict | None:
    task_result = record.task_result
    if isinstance(task_result, dict) and task_result.get("type") == "rollout_progress":
        return task_result

    skill_payload = _get_run_skill_payload(record)
    structured_response = skill_payload.get("structured_response") if skill_payload else None
    if isinstance(structured_response, dict) and structured_response.get("type") == "rollout_progress":
        return structured_response
    return None


def _get_tool_display_name(record: ToolCallRecord) -> str:
    if _is_long_horizon_skill_record(record):
        return f"{record.tool_name} · {LONG_HORIZON_SKILL_NAME}"
    return record.tool_name


def _should_hide_tool_parameters(record: ToolCallRecord) -> bool:
    return record.tool_name == "organize_and_clean_table" or _is_long_horizon_skill_record(record)


def _maybe_override_organize_msg(session_inst, bot_msg: str) -> str:
    """
    If the most recent tool call was organize_and_clean_table and it completed,
    return the fixed completion message instead of the LLM's response.
    Only fires once per tool call (uses a flag on the record to avoid duplicates).
    """
    try:
        if not getattr(session_inst, 'initialized', False):
            return bot_msg
        tool_history = session_inst.agent.get_tool_call_history()
        if not tool_history:
            return bot_msg
        # Check if any tool in history is organize_and_clean_table with a completed result
        for record in reversed(tool_history):
            if record.tool_name == "organize_and_clean_table" and record.result_preview:
                # Only override once — skip if already consumed
                if getattr(record, '_organize_msg_sent', False):
                    return bot_msg
                record._organize_msg_sent = True
                return ORGANIZE_COMPLETE_MSG
    except Exception:
        pass
    return bot_msg


def _is_rollout_completed(record) -> bool:
    """检查 rollout_task 的 ToolCallRecord 是否已全部完成"""
    tr = _get_rollout_progress_payload(record)
    if not tr:
        # 没有进度信息时退回到 result_preview 检查
        return bool(record.result_preview)
    tasks = tr.get("tasks", [])
    return bool(tasks) and all(t.get("status") == "completed" for t in tasks)


def _has_active_completed_rollout(session_inst) -> bool:
    """Check if there's a completed rollout_task whose completion message was already sent."""
    try:
        if not getattr(session_inst, 'initialized', False):
            return False
        tool_history = session_inst.agent.get_tool_call_history()
        if not tool_history:
            return False
        for record in reversed(tool_history):
            if (record.tool_name == "rollout_task" or _is_long_horizon_skill_record(record)) and _is_rollout_completed(
                record
            ):
                return getattr(record, '_rollout_msg_sent', False)
        return False
    except Exception:
        return False


def _maybe_override_rollout_msg(session_inst, bot_msg: str) -> str:
    """
    If the most recent tool call was rollout_task and it completed,
    return the fixed completion message instead of the LLM's response.
    Only fires once per tool call (uses a flag on the record to avoid duplicates).
    """
    try:
        if not getattr(session_inst, 'initialized', False):
            return bot_msg
        tool_history = session_inst.agent.get_tool_call_history()
        if not tool_history:
            return bot_msg
        for record in reversed(tool_history):
            if (record.tool_name == "rollout_task" or _is_long_horizon_skill_record(record)) and _is_rollout_completed(
                record
            ):
                if getattr(record, '_rollout_msg_sent', False):
                    return bot_msg
                record._rollout_msg_sent = True
                return ROLLOUT_COMPLETE_MSG
    except Exception:
        pass
    return bot_msg


def snapshot_chat_history(chat_history: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return a detached copy so background updates are visible to Gradio polling."""
    return [{"role": message.get("role", ""), "content": message.get("content", "")} for message in chat_history]


# ======================== UI 状态机定义 ========================
class UIState(Enum):
    """UI interaction state"""

    IDLE = "idle"  # Initialized, waiting for user input
    WAITING_INFER = "waiting_infer"  # Received user input, waiting to trigger inference
    INFERING = "infering"  # Inferring
    DONE = "done"  # Inference completed, waiting for new input
    FAILED = "failed"  # Inference failed, waiting for new input


class UIEvent(Enum):
    """UI interaction events"""

    USER_INPUT = "user_input"  # User input arrives
    INFER_START = "infer_start"  # State machine decides inference can start
    INFER_SUCCESS = "infer_success"  # Inference finished successfully
    INFER_FAIL = "infer_fail"  # Inference failed


class UIStateMachine:
    """UI interaction state machine"""

    def __init__(self):
        self.state = UIState.IDLE
        self._state_lock = asyncio.Lock()
        self._infer_done = asyncio.Event()
        self._infer_done.set()  # Initial state: inference considered done

    async def send_event(self, event: UIEvent) -> bool:
        """
        Send an event and return whether the state transition succeeded.
        - USER_INPUT: IDLE/DONE/FAILED → WAITING_INFER (other states return False)
        - INFER_START: WAITING_INFER → INFERING
        - INFER_SUCCESS: INFERING → DONE
        - INFER_FAIL: INFERING → FAILED
        """
        async with self._state_lock:
            if event == UIEvent.USER_INPUT:
                # Only accept new input in idle or done/failed state
                if self.state in [UIState.IDLE, UIState.DONE, UIState.FAILED]:
                    self.state = UIState.WAITING_INFER
                    self._infer_done.clear()
                    logger.debug(f"[UIStateMachine] USER_INPUT: {self.state.value}")
                    return True
                else:
                    # While inferring or waiting, do not accept new input (caller should wait)
                    logger.debug(f"[UIStateMachine] USER_INPUT rejected, current state: {self.state.value}")
                    return False

            elif event == UIEvent.INFER_START:
                if self.state == UIState.WAITING_INFER:
                    self.state = UIState.INFERING
                    logger.debug(f"[UIStateMachine] INFER_START: {self.state.value}")
                    return True
                return False

            elif event == UIEvent.INFER_SUCCESS:
                if self.state == UIState.INFERING:
                    self.state = UIState.DONE
                    self._infer_done.set()
                    logger.debug(f"[UIStateMachine] INFER_SUCCESS: {self.state.value}")
                    return True
                return False

            elif event == UIEvent.INFER_FAIL:
                if self.state == UIState.INFERING:
                    self.state = UIState.FAILED
                    self._infer_done.set()
                    logger.debug(f"[UIStateMachine] INFER_FAIL: {self.state.value}")
                    return True
                return False

            return False

    async def wait_inference_done(self):
        """Wait until inference is finished."""
        await self._infer_done.wait()

    def get_state_display(self) -> str:
        """Get display text for current UI state."""
        state_display = {
            UIState.IDLE: "✅ Idle",
            UIState.WAITING_INFER: "⏳ Waiting for inference",
            UIState.INFERING: "🔄 Inferring...",
            UIState.DONE: "✅ Inference completed",
            UIState.FAILED: "❌ Inference failed",
        }
        return state_display.get(self.state, "Unknown state")


def build_detailed_state_text(
    ui_state_machine: "UIStateMachine",
    session_inst: "Session",
    user_input_processing: asyncio.Event,
) -> str:
    """
    构建更详细的状态文本（不包含模型推理链内容）。
    展示：UI状态 + agent阶段 + 当前/最近工具调用。
    """
    from agent_demo.types.agent_types import ActAgentState

    lines: list[str] = []
    lines.append(ui_state_machine.get_state_display())

    executing = bool(getattr(session_inst, "_run_once_executing", False))
    lines.append(f"Executing: {'Yes' if executing else 'No'}")

    config_error = getattr(session_inst, "config_error", None)
    if config_error:
        lines.append("Agent: Configuration error")
        lines.append(f"API: {config_error}")
        return "\n".join(lines)
    a2d_warning = getattr(session_inst, "a2d_warning", None)
    if a2d_warning:
        lines.append(f"A2D: {a2d_warning}")

    # 初始化与 agent 阶段
    if not getattr(session_inst, "initialized", False):
        lines.append("Agent: Not initialized")
        return "\n".join(lines)

    agent_state = None
    try:
        agent_state = session_inst.agent.state
    except Exception:
        agent_state = None

    if agent_state is not None:
        state_name = agent_state.name if hasattr(agent_state, "name") else str(agent_state)
        lines.append(f"Phase: {state_name}")

    # 当前/最近工具
    try:
        tool_history = session_inst.get_tool_call_history()
    except Exception:
        tool_history = []

    if tool_history:
        last = tool_history[-1]
        tool_full_name = f"{last.service_name}.{last.tool_name} (#{last.step_index})"
        in_act = agent_state == ActAgentState.ACT if agent_state is not None else False
        if in_act and not last.result_preview:
            lines.append(f"Current tool: {tool_full_name}")
            lines.append("Tool status: Executing...")
        else:
            lines.append(f"Recent tool: {tool_full_name}")
            lines.append(f"Tool status: {'Completed' if last.result_preview else 'Waiting for result'}")
    else:
        if agent_state == ActAgentState.ACT:
            lines.append("Tool status: Ready to call/Waiting for tool list")

    # 解释自动推理为何可能不触发
    if user_input_processing.is_set():
        lines.append("Note: Processing user input, auto inference will be skipped")
    elif ui_state_machine.state == UIState.WAITING_INFER:
        lines.append("Note: Auto inference triggers every 5 seconds")

    return "\n".join(lines)


def build_streaming_assistant_content(assistant_text: str, status_text: str | None) -> str:
    """在流式输出期间构建 assistant 气泡内容。"""
    normalized_text = assistant_text or ""
    normalized_status = (status_text or "").strip()

    if normalized_text and normalized_status:
        return f"{normalized_text}\n\n_Status: {normalized_status}_"
    if normalized_text:
        return normalized_text
    if normalized_status:
        return f"_{normalized_status}_"
    return "_Thinking..._"


# from agent_demo.agent_layer.agent_core import ActAgent  # 原始实现（已切换为 ImgActAgent）
# from agent_demo.agent_layer.agent_prompt import ActAgentPrompt  # 原始实现（已切换为 ImgActAgentPrompt）


class Session:
    def __init__(self):
        self.initialized = False
        self._init_lock = asyncio.Lock()
        self._run_once_lock = asyncio.Lock()  # 防止并发调用 run_once
        self._run_once_executing = False  # 标记是否正在执行
        self.config_error: str | None = None
        self.a2d_warning: str | None = None
        self._agent_card: BaseAgentCard | None = None
        self.agent: ImgActAgent | None = None
        self.skill_registry = LocalSkillRegistry(workspace_root=str(REPO_ROOT))

        try:
            robot_dataloader, self.a2d_warning = _create_robot_dataloader()
            self._agent_card = BaseAgentCard(
                silence=False,
                config=ChatAPIConfig.resolve_runtime_default(),
                service_config_path=str(REPO_ROOT / "src/agent_demo/config/ormcp_services.json"),
                skill_paths=[str(REPO_ROOT / "skills")],
                agent_memory_prompt=ImgActAgentPrompt.init_memory_prompt,
                robot_dataloader=robot_dataloader,
                # agent_memory_prompt=ActAgentPrompt.init_memory_prompt,  # 原始实现
                # 注意：原始实现没有 robot_dataloader 参数
            )
            self.agent = ImgActAgent(agent_card=self._agent_card)
            self.skill_registry = LocalSkillRegistry(
                configured_paths=self._agent_card.skill_paths,
                workspace_root=str(REPO_ROOT),
            )
            # self.agent = ActAgent(agent_card=self._agent_card)  # 原始实现
        except Exception as e:
            self.config_error = str(e)
            logger.error("LLM configuration error: %s", self.config_error)

    def get_tool_call_history(self) -> list[ToolCallRecord]:
        if not self.initialized or self.agent is None:
            return []
        try:
            return self.agent.get_tool_call_history()
        except Exception:
            return []

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

    def prepare_message_for_agent(self, message: str) -> PreparedAgentMessage:
        services = self.agent.service_manager._services_register_list if self.agent is not None else None
        return prepare_agent_message(message, self.skill_registry, services=services)

    # async def tts_inference(self, content: str) -> None:
    #     url = "http://127.0.0.1:8000/tts_inference"
    #     headers = {"Content-Type": "application/json"}
    #     data = {"content": content}

    #     timeout = httpx.Timeout(connect=1.0, read=0.1, write=1.0, pool=1.0)  # 非常短的超时

    #     async with httpx.AsyncClient(timeout=timeout) as client:
    #         try:
    #             await client.post(url, headers=headers, json=data)
    #         except (httpx.RequestError, httpx.ReadTimeout):
    #             logger.info("Fire-and-forget: 请求发送但不等待响应。超时可忽略。")

    async def _run_agent_once(
        self,
        message: str,
        clear_history: bool = True,
        on_text_delta=None,
        on_status=None,
    ) -> str:
        # #region agent log
        try:
            with open("/home/agiuser/Project_Olympus_zyl/.cursor/debug.log", "a") as f:
                f.write(
                    json.dumps(
                        {
                            "id": f"log_{int(time.time()*1000)}",
                            "timestamp": int(time.time() * 1000),
                            "location": "gradio_ui.py:152",
                            "message": "Session.run_once entry",
                            "data": {
                                "message_preview": message[:50] if len(message) > 50 else message,
                                "clear_history": clear_history,
                            },
                            "runId": "debug",
                            "hypothesisId": "A",
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
        # #endregion
        if self.config_error:
            raise RuntimeError(f"API configuration error: {self.config_error}")

        await self.ensure_initialized()
        assert self.agent is not None

        # 使用锁保护，确保同一时间只有一个 run_once 在执行；用户输入优先，排队等待而不是直接跳过
        async with self._run_once_lock:
            self._run_once_executing = True

            try:
                # 检查 agent 状态，如果不在 READY 状态则记录警告
                # run_once 内部会处理状态恢复，所以这里只做检查和日志记录
                from agent_demo.types.agent_types import ActAgentState

                if self.agent.state != ActAgentState.READY:
                    logger.warning(
                        f"[Session] Agent 状态为 {self.agent.state}，不是 READY（可能是上次异常后未恢复），run_once 将尝试恢复"
                    )

                try:
                    # #region agent log
                    try:
                        with open("/home/agiuser/Project_Olympus_zyl/.cursor/debug.log", "a") as f:
                            f.write(
                                json.dumps(
                                    {
                                        "id": f"log_{int(time.time()*1000)}",
                                        "timestamp": int(time.time() * 1000),
                                        "location": "gradio_ui.py:172",
                                        "message": "Calling agent.run_once",
                                        "data": {"clear_history": clear_history},
                                        "runId": "debug",
                                        "hypothesisId": "B",
                                    }
                                )
                                + "\n"
                            )
                    except Exception:
                        pass
                    # #endregion
                    res: TextParam | None = await self.agent.run_once(
                        message,
                        clear_history=clear_history,
                        on_text_delta=on_text_delta,
                        on_status=on_status,
                    )
                    # if res is not None:
                    # asyncio.create_task(self.tts_inference(res.text))  # 用 create_task 推到后台执行
                    return res.text if res else "No result returned."
                except asyncio.CancelledError:
                    logger.info("[Session] run_once cancelled by caller.")
                    raise
                except Exception as e:
                    error_str = str(e)
                    # 检查是否是 tool 消息相关的错误
                    if (
                        "messages with role 'tool'" in error_str
                        and "must be a response to a preceeding message with 'tool_calls'" in error_str
                    ):
                        logger.warning(
                            f"[Session] 检测到 tool 消息配对错误，清理孤立 tool 消息并重试: {type(e).__name__}: {e}"
                        )
                        # 清理孤立的 tool 消息
                        self.agent.current_task_node._cleanup_orphaned_tool_messages()
                        # 重试一次
                        try:
                            # #region agent log
                            try:
                                with open("/home/agiuser/Project_Olympus_zyl/.cursor/debug.log", "a") as f:
                                    f.write(
                                        json.dumps(
                                            {
                                                "id": f"log_{int(time.time()*1000)}",
                                                "timestamp": int(time.time() * 1000),
                                                "location": "gradio_ui.py:190",
                                                "message": "Retry agent.run_once",
                                                "data": {"clear_history": clear_history},
                                                "runId": "debug",
                                                "hypothesisId": "B",
                                            }
                                        )
                                        + "\n"
                                    )
                            except Exception:
                                pass
                            # #endregion
                            res: TextParam | None = await self.agent.run_once(
                                message,
                                clear_history=clear_history,
                                on_text_delta=on_text_delta,
                                on_status=on_status,
                            )
                            logger.info("[Session] Retry of agent.run_once succeeded")
                            return res.text if res else "No result returned."
                        except Exception as retry_e:
                            logger.error(f"[Session] 重试后仍然失败: {type(retry_e).__name__}: {retry_e}")
                            logger.error("Full traceback:\n" + traceback.format_exc())
                            raise retry_e
                    else:
                        logger.error(f"Error calling OpenAI API: {type(e).__name__}: {e}")
                        logger.error("Full traceback:\n" + traceback.format_exc())
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
            except Exception as e:
                await stream_queue.put({"type": "error", "text": str(e)})
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


class ASR_Model:
    def __init__(self, model_name: str = "base"):
        self.model = None
        try:
            self.model = whisper.load_model(model_name)
        except Exception as e:
            logger.warning("Whisper ASR model failed to load (voice input disabled): %s", e)

    def transcribe(self, audio_path: str) -> str:
        if self.model is None:
            return "[Voice input unavailable: Whisper model not loaded]"
        result = self.model.transcribe(audio_path)
        return str(result["text"])


# ======================== 工具调用流程渲染 ========================
# 服务图标映射
SERVICE_ICON_MAP = {
    "data_analyst_mcp_server": "📊",
    "robot": "🤖",
    "knowledge": "📚",
    "memory": "🧠",
    "default": "🔧",
}

# 服务颜色映射（CSS class）
SERVICE_COLOR_MAP = {
    "data_analyst_mcp_server": "service-data",
    "robot": "service-robot",
    "knowledge": "service-knowledge",
    "memory": "service-memory",
    "default": "service-default",
}


def get_service_icon(service_name: str) -> str:
    """根据服务名获取图标"""
    for key, icon in SERVICE_ICON_MAP.items():
        if key in service_name.lower():
            return icon
    return SERVICE_ICON_MAP["default"]


def get_service_color_class(service_name: str) -> str:
    """根据服务名获取颜色类"""
    for key, color_class in SERVICE_COLOR_MAP.items():
        if key in service_name.lower():
            return color_class
    return SERVICE_COLOR_MAP["default"]


def extract_visualization_path(tool_history: list[ToolCallRecord]) -> str | None:
    """
    从工具调用历史中提取HTML可视化文件路径

    Args:
        tool_history: 工具调用历史记录列表

    Returns:
        HTML文件路径，如果未找到则返回None
    """
    if not tool_history:
        return None

    # 查找数据分析相关的工具调用
    for record in reversed(tool_history):  # 从最新开始查找
        if record.service_name == "data_analyst_mcp_server":
            if record.tool_name in ["full_analysis", "generate_3d_visualization"]:
                # 尝试从result_preview中解析JSON
                try:
                    # result_preview可能被截断，先尝试解析
                    result_data = json.loads(record.result_preview)
                    html_path = result_data.get("visualization") or result_data.get("html_path")
                    if html_path:
                        return html_path
                except (json.JSONDecodeError, ValueError, AttributeError):
                    # 如果解析失败，尝试从args_preview中获取output_path
                    try:
                        args_data = json.loads(record.args_preview)
                        html_path = args_data.get("output_path")
                        if html_path:
                            return html_path
                    except (json.JSONDecodeError, ValueError, AttributeError):
                        pass

                # 如果都失败了，使用默认路径
                return "/tmp/multi_tsne_3d.html"

    return None


def extract_plotly_figure_from_html(html_path: str) -> go.Figure | None:
    """
    从Plotly生成的HTML文件中提取Figure对象

    Args:
        html_path: HTML文件路径

    Returns:
        Plotly Figure对象，如果提取失败则返回None
    """

    # 检查文件是否存在
    if not os.path.exists(html_path):
        logger.warning(f"HTML file not found: {html_path}")
        return None

    try:
        # 方法1: 尝试使用plotly.io.from_json直接读取（如果HTML包含JSON）
        # 但plotly.io不支持直接从HTML读取，所以我们需要手动提取

        # 读取HTML文件内容
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # Plotly HTML文件使用 Plotly.newPlot('id', data_array, layout_obj, config_obj)
        # 我们需要提取data数组和layout对象

        # 方法2: 提取Plotly.newPlot的第二个和第三个参数（data和layout）
        # 由于数据可能很大且包含嵌套结构，需要使用平衡括号匹配
        plotly_match = re.search(r'Plotly\.newPlot\s*\(', html_content)
        if plotly_match:
            start_pos = plotly_match.end()
            # 跳过第一个参数（id字符串）
            pos = start_pos
            depth = 0
            in_string = False
            escape_next = False

            # 跳过第一个参数（通常是字符串ID）
            while pos < len(html_content):
                char = html_content[pos]
                if escape_next:
                    escape_next = False
                    pos += 1
                    continue
                if char == '\\':
                    escape_next = True
                    pos += 1
                    continue
                if char == '"' or char == "'":
                    in_string = not in_string
                if not in_string:
                    if char == '(':
                        depth += 1
                    elif char == ')':
                        depth -= 1
                        if depth < 0:
                            break
                    elif char == ',' and depth == 0:
                        # 找到第一个逗号，跳过第一个参数
                        pos += 1
                        # 跳过空白
                        while pos < len(html_content) and html_content[pos] in ' \n\t':
                            pos += 1
                        break
                pos += 1

            # 现在pos指向第二个参数（data数组）的开始
            # 提取data数组（使用平衡括号）
            data_start = pos
            depth = 0
            in_string = False
            escape_next = False
            bracket_type = None  # '[' or '{'

            while pos < len(html_content):
                char = html_content[pos]
                if escape_next:
                    escape_next = False
                    pos += 1
                    continue
                if char == '\\':
                    escape_next = True
                    pos += 1
                    continue
                if char == '"' or char == "'":
                    in_string = not in_string
                if not in_string:
                    if char == '[':
                        if depth == 0:
                            bracket_type = '['
                        depth += 1
                    elif char == ']':
                        depth -= 1
                        if depth == 0 and bracket_type == '[':
                            data_str = html_content[data_start : pos + 1]
                            pos += 1
                            # 跳过空白和逗号
                            while pos < len(html_content) and html_content[pos] in ' ,\n\t':
                                pos += 1
                            break
                pos += 1

            # 提取layout对象
            layout_start = pos
            depth = 0
            in_string = False
            escape_next = False
            bracket_type = None

            while pos < len(html_content):
                char = html_content[pos]
                if escape_next:
                    escape_next = False
                    pos += 1
                    continue
                if char == '\\':
                    escape_next = True
                    pos += 1
                    continue
                if char == '"' or char == "'":
                    in_string = not in_string
                if not in_string:
                    if char == '{':
                        if depth == 0:
                            bracket_type = '{'
                        depth += 1
                    elif char == '}':
                        depth -= 1
                        if depth == 0 and bracket_type == '{':
                            layout_str = html_content[layout_start : pos + 1]
                            break
                pos += 1

            # 尝试解析JSON并创建Figure
            if 'data_str' in locals() and 'layout_str' in locals():
                try:
                    data = json.loads(data_str)
                    layout = json.loads(layout_str)
                    fig = go.Figure(data=data, layout=layout)
                    logger.info(f"Successfully extracted Plotly figure from HTML: {html_path}")
                    return fig
                except json.JSONDecodeError as e:
                    logger.debug(f"Failed to parse JSON from Plotly.newPlot: {e}")
                except Exception as e:
                    logger.debug(f"Failed to create Figure: {e}")

        # 方法3: 如果Plotly.newPlot格式不匹配，尝试查找window.PLOTLYENV中的数据
        # 或者查找包含完整figure数据的变量
        window_pattern = r'window\.PLOTLYENV\s*=\s*({.*?});'
        window_match = re.search(window_pattern, html_content, re.DOTALL)

        if window_match:
            try:
                env_data = json.loads(window_match.group(1))
                # 尝试从环境变量中提取figure数据
                if 'data' in env_data and 'layout' in env_data:
                    fig = go.Figure(data=env_data['data'], layout=env_data['layout'])
                    return fig
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

        # 如果所有方法都失败，记录警告
        logger.warning(
            f"Could not extract Plotly data from HTML: {html_path}. The HTML structure may be different than expected."
        )
        logger.debug(f"HTML file size: {len(html_content)} bytes")
        return None

    except Exception as e:
        logger.error(f"Error extracting Plotly figure from HTML {html_path}: {e}", exc_info=True)
        return None


def load_plotly_figure(html_path: str | None) -> go.Figure | None:
    """
    加载Plotly可视化图表

    Args:
        html_path: HTML文件路径，如果为None则返回None

    Returns:
        Plotly Figure对象，如果加载失败则返回None
    """
    if html_path is None:
        return None

    return extract_plotly_figure_from_html(html_path)


def render_task_tree(task_result: dict) -> str:
    """
    Render structured task detection result as a tree-like HTML view.
    Expected structure (keys unchanged to keep compatibility):
    {
      "categories": [
        {
          "category": "...",
          "tasks": [...],
          "reason": "..."
        },
        ...
      ],
      "summary": "..."
    }
    """
    if not task_result or "categories" not in task_result:
        return ""

    categories = task_result.get("categories", [])
    if not categories:
        return '<div class="task-tree-empty">No tasks to execute</div>'

    html_parts = ['<div class="task-tree">']

    for category in categories:
        category_name = category.get("category", "Unknown category")
        tasks = category.get("tasks", [])
        reason = category.get("reason", "")

        if not tasks:
            continue

        html_parts.append(
            f'''
            <details class="task-category" open>
                <summary class="task-category-summary">
                    <span class="task-category-name">{escape(category_name)}</span>
                    <span class="task-count">({len(tasks)} items)</span>
                </summary>
                <ul class="task-list">
            '''
        )

        for task in tasks:
            html_parts.append(f'<li class="task-item">{escape(task)}</li>')

        if reason:
            html_parts.append(f'<li class="task-reason"><em>{escape(reason)}</em></li>')

        html_parts.append('</ul></details>')

    summary = task_result.get("summary", "")
    if summary:
        html_parts.append(f'<div class="task-summary">{escape(summary)}</div>')

    html_parts.append('</div>')
    return ''.join(html_parts)


def render_organize_progress(task_result: dict) -> str:
    """
    Render organize_and_clean_table subtask progress as a live status card.
    task_result format:
    {
      "type": "organize_progress",
      "total": 4,
      "current_index": 1,
      "tasks": [
        {"description": "...", "category": "...", "status": "completed"/"in_progress"/"pending", "timeout": 10},
        ...
      ]
    }
    """
    if not task_result or task_result.get("type") != "organize_progress":
        return ""

    tasks = task_result.get("tasks", [])
    total = task_result.get("total", len(tasks))
    if not tasks:
        return ""

    completed_count = sum(1 for t in tasks if t.get("status") == "completed")
    progress_pct = int((completed_count / total) * 100) if total > 0 else 0

    html_parts = ['<div class="organize-progress">']

    # Progress bar
    html_parts.append(
        f'''
        <div class="organize-progress-header">
            <span class="organize-title">🧹 Organize & Clean Table</span>
            <span class="organize-counter">{completed_count}/{total} completed</span>
        </div>
        <div class="organize-progress-bar-bg">
            <div class="organize-progress-bar-fill" style="width: {progress_pct}%;"></div>
        </div>
        '''
    )

    # Task list
    html_parts.append('<ul class="organize-task-list">')
    for i, task in enumerate(tasks):
        status = task.get("status", "pending")
        desc = escape(task.get("description", ""))
        category = escape(task.get("category", ""))
        is_reset = category == "Reset"

        if status == "completed":
            icon = "✅"
            css_class = "organize-task-completed"
        elif status == "in_progress":
            icon = "🔄" if not is_reset else "🔁"
            css_class = "organize-task-active"
        else:
            icon = "⏳"
            css_class = "organize-task-pending"

        # Reset 任务使用额外的 CSS class 以区分样式
        if is_reset:
            css_class += " organize-task-reset"

        html_parts.append(
            f'''
            <li class="organize-task-item {css_class}">
                <span class="organize-task-icon">{icon}</span>
                <div class="organize-task-info">
                    <span class="organize-task-desc">{desc}</span>
                    <span class="organize-task-meta">{category}</span>
                </div>
            </li>
            '''
        )
    html_parts.append('</ul>')
    html_parts.append('</div>')
    return ''.join(html_parts)


def render_rollout_progress(task_result: dict) -> str:
    """
    Render rollout_task subtask progress as a live status card.
    task_result format:
    {
      "type": "rollout_progress",
      "total": 3,
      "current_index": 1,
      "tasks": [
        {"description": "...", "category": "...", "status": "completed"/"in_progress"/"pending", "timeout": 10},
        ...
      ]
    }
    """
    if not task_result or task_result.get("type") != "rollout_progress":
        return ""

    tasks = task_result.get("tasks", [])
    total = task_result.get("total", len(tasks))
    if not tasks:
        return ""

    completed_count = sum(1 for t in tasks if t.get("status") == "completed")
    progress_pct = int((completed_count / total) * 100) if total > 0 else 0

    html_parts = ['<div class="organize-progress">']

    # Progress bar
    html_parts.append(
        f'''
        <div class="organize-progress-header">
            <span class="organize-title">🔁 Rollout Task</span>
            <span class="organize-counter">{completed_count}/{total} completed</span>
        </div>
        <div class="organize-progress-bar-bg">
            <div class="organize-progress-bar-fill" style="width: {progress_pct}%;"></div>
        </div>
        '''
    )

    # Task list
    html_parts.append('<ul class="organize-task-list">')
    for i, task in enumerate(tasks):
        status = task.get("status", "pending")
        desc = escape(task.get("description", ""))
        category = escape(task.get("category", ""))
        timeout = task.get("timeout", 0)
        is_reset = category == "Reset"

        if status == "completed":
            icon = "✅"
            css_class = "organize-task-completed"
        elif status == "in_progress":
            icon = "🔄" if not is_reset else "🔁"
            css_class = "organize-task-active"
        else:
            icon = "⏳"
            css_class = "organize-task-pending"

        # Reset 任务使用额外的 CSS class 以区分样式
        if is_reset:
            css_class += " organize-task-reset"

        html_parts.append(
            f'''
            <li class="organize-task-item {css_class}">
                <span class="organize-task-icon">{icon}</span>
                <div class="organize-task-info">
                    <span class="organize-task-desc">{desc}</span>
                    <span class="organize-task-meta">{category} · {timeout}s</span>
                </div>
            </li>
            '''
        )
    html_parts.append('</ul>')
    html_parts.append('</div>')
    return ''.join(html_parts)


def render_tool_flow(tool_history: list[ToolCallRecord]) -> str:
    """
    Render tool call history as HTML.
    Grouped by service and shown as a branched timeline view.
    """
    if not tool_history:
        return (
            "<div class=\"tool-flow-container\">"
            "<p style='color: #888; padding: 20px; text-align: center;'>"
            "Waiting for tool calls..."
            "</p></div>"
        )

    # 按服务分组
    service_groups: dict[str, list[ToolCallRecord]] = {}
    for record in tool_history:
        service_name = record.service_name
        if service_name not in service_groups:
            service_groups[service_name] = []
        service_groups[service_name].append(record)

    # 生成 HTML
    html_parts = ['<div class="tool-flow-container">']

    for service_name, records in service_groups.items():
        icon = get_service_icon(service_name)
        color_class = get_service_color_class(service_name)

        # 对该服务下的调用按“连续相同 tool_name”分组
        grouped_records: list[list[ToolCallRecord]] = []
        current_group: list[ToolCallRecord] = []
        for record in records:
            if not current_group:
                current_group.append(record)
            else:
                if record.tool_name == current_group[-1].tool_name:
                    current_group.append(record)
                else:
                    grouped_records.append(current_group)
                    current_group = [record]
        if current_group:
            grouped_records.append(current_group)

        # Service header
        html_parts.append(
            f'''
            <div class="service-branch {color_class}">
                <div class="service-header">
                    <span class="service-icon">{icon}</span>
                    <span class="service-name">{escape(service_name)}</span>
                    <span class="service-count">({len(records)} calls)</span>
                </div>
                <div class="service-timeline">
        '''
        )

        # All tool calls under this service (grouped by consecutive identical tool_name, collapsible)
        for group_index, group in enumerate(grouped_records):
            is_last_group = group_index == len(grouped_records) - 1

            # Single-call group: render directly
            if len(group) == 1:
                record = group[0]
                show_parameters = not _should_hide_tool_parameters(record)
                # 如果是任务检测工具且有结构化结果，使用任务树渲染
                task_tree_html = ""
                if record.tool_name == "detect_tasks_from_image" and record.task_result:
                    task_tree_html = render_task_tree(record.task_result)
                elif record.tool_name == "organize_and_clean_table" and record.task_result:
                    task_tree_html = render_organize_progress(record.task_result)
                elif record.tool_name == "rollout_task":
                    task_tree_html = render_rollout_progress(record.task_result)
                elif _is_long_horizon_skill_record(record):
                    task_tree_html = render_rollout_progress(_get_rollout_progress_payload(record) or {})

                html_parts.append(
                    f'''
                    <div class="tool-call-card">
                        <div class="tool-call-line {'last' if is_last_group else ''}"></div>
                        <div class="tool-call-content">
                            <div class="tool-call-header">
                                <span class="tool-step">#{record.step_index}</span>
                                <span class="tool-name">{escape(_get_tool_display_name(record))}</span>
                                <span class="tool-time">{escape(record.timestamp)}</span>
                            </div>
                            <div class="tool-call-details">
                                {f'<div class="tool-args"><strong>Parameters:</strong><pre class="tool-args-preview">{escape(record.args_preview)}</pre></div>' if show_parameters else ''}
                                <div class="tool-result">
                                    <strong>Result:</strong>
                                    {task_tree_html if task_tree_html else f'<pre class="tool-result-preview">{escape(record.result_preview) if record.result_preview else "Running..."}</pre>'}
                                </div>
                            </div>
                        </div>
                    </div>
                    '''
                )
            else:
                first_record = group[0]
                last_record = group[-1]
                call_count = len(group)
                if first_record.step_index == last_record.step_index:
                    step_range = f"#{first_record.step_index}"
                else:
                    step_range = f"#{first_record.step_index}-{last_record.step_index}"

                # Collapsed group: outer card with details listing each call
                html_parts.append(
                    f'''
                    <div class="tool-call-card tool-call-group">
                        <div class="tool-call-line {'last' if is_last_group else ''}"></div>
                        <div class="tool-call-content">
                            <details>
                                <summary>
                                    <span class="tool-step">{escape(step_range)}</span>
                                    <span class="tool-name">{escape(first_record.tool_name)}</span>
                                    <span class="tool-group-count">({call_count} consecutive calls)</span>
                                </summary>
                                <div class="tool-call-group-body">
                    '''
                )

                # Each call detail within the group
                for record in group:
                    show_parameters = not _should_hide_tool_parameters(record)
                    task_tree_html = ""
                    if record.tool_name == "detect_tasks_from_image" and record.task_result:
                        task_tree_html = render_task_tree(record.task_result)
                    elif record.tool_name == "organize_and_clean_table" and record.task_result:
                        task_tree_html = render_organize_progress(record.task_result)
                    elif record.tool_name == "rollout_task":
                        task_tree_html = render_rollout_progress(record.task_result)
                    elif _is_long_horizon_skill_record(record):
                        task_tree_html = render_rollout_progress(_get_rollout_progress_payload(record) or {})

                    html_parts.append(
                        f'''
                                    <div class="tool-call-subcard">
                                        <div class="tool-call-header">
                                            <span class="tool-step">#{record.step_index}</span>
                                            <span class="tool-time">{escape(record.timestamp)}</span>
                                        </div>
                                        <div class="tool-call-details">
                                            {f'<div class="tool-args"><strong>Parameters:</strong><pre class="tool-args-preview">{escape(record.args_preview)}</pre></div>' if show_parameters else ''}
                                            <div class="tool-result">
                                                <strong>Result:</strong>
                                                {task_tree_html if task_tree_html else f'<pre class="tool-result-preview">{escape(record.result_preview) if record.result_preview else "Running..."}</pre>'}
                                            </div>
                                        </div>
                                    </div>
                        '''
                    )

                html_parts.append(
                    '''
                                </div>
                            </details>
                        </div>
                    </div>
                    '''
                )

        html_parts.append(
            '''
                </div>
            </div>
        '''
        )

    html_parts.append('</div>')

    # 添加 CSS 样式
    css = '''
    <style>
    .tool-flow-container {
        padding: 15px;
        max-height: 600px;
        overflow-y: auto;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .service-branch {
        margin-bottom: 25px;
        border-left: 3px solid #e0e0e0;
        padding-left: 15px;
    }
    
    .service-branch.service-data { border-left-color: #4CAF50; }
    .service-branch.service-robot { border-left-color: #2196F3; }
    .service-branch.service-knowledge { border-left-color: #FF9800; }
    .service-branch.service-memory { border-left-color: #9C27B0; }
    .service-branch.service-default { border-left-color: #757575; }
    
    .service-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 12px;
        font-weight: 600;
        font-size: 14px;
        color: #333;
    }
    
    .service-icon {
        font-size: 18px;
    }
    
    .service-name {
        color: #555;
    }
    
    .service-count {
        color: #888;
        font-size: 12px;
        font-weight: normal;
    }
    
    .service-timeline {
        position: relative;
        padding-left: 20px;
    }
    
    .tool-call-card {
        position: relative;
        margin-bottom: 15px;
    }
    
    .tool-call-line {
        position: absolute;
        left: -25px;
        top: 0;
        width: 2px;
        height: 100%;
        background: #ddd;
    }
    
    .tool-call-line.last {
        height: 20px;
    }
    
    .tool-call-card::before {
        content: '';
        position: absolute;
        left: -30px;
        top: 8px;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #2196F3;
        border: 2px solid white;
        box-shadow: 0 0 0 2px #ddd;
    }
    
    .tool-call-card.tool-call-group .tool-call-content {
        background: #f1f5fb;
    }
    
    .tool-call-content {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        transition: box-shadow 0.2s;
    }
    
    .tool-call-content:hover {
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    }
    
    .tool-call-group summary {
        display: flex;
        align-items: center;
        gap: 10px;
        cursor: pointer;
        list-style: none;
        outline: none;
    }
    
    .tool-call-group summary::-webkit-details-marker {
        display: none;
    }
    
    .tool-group-count {
        font-size: 11px;
        color: #888;
    }
    
    .tool-call-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 10px;
        padding-bottom: 8px;
        border-bottom: 1px solid #e0e0e0;
    }
    
    .tool-step {
        background: #2196F3;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
    }
    
    .tool-name {
        font-weight: 600;
        color: #333;
        flex: 1;
    }
    
    .tool-time {
        font-size: 11px;
        color: #888;
    }
    
    .tool-call-details {
        display: flex;
        flex-direction: column;
        gap: 10px;
    }
    
    .tool-args, .tool-result {
        font-size: 12px;
    }
    
    .tool-args strong, .tool-result strong {
        color: #555;
        display: block;
        margin-bottom: 4px;
    }
    
    .tool-call-group-body {
        margin-top: 8px;
        padding-top: 8px;
        border-top: 1px dashed #e0e0e0;
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    
    .tool-call-subcard {
        background: white;
        border-radius: 6px;
        padding: 8px;
        border: 1px solid #e0e0e0;
    }
    
    .tool-args-preview, .tool-result-preview {
        background: white;
        padding: 8px;
        border-radius: 4px;
        border: 1px solid #e0e0e0;
        margin: 0;
        font-size: 11px;
        color: #333;
        white-space: pre-wrap;
        word-wrap: break-word;
        max-height: 100px;
        overflow-y: auto;
    }
    
    .tool-result-preview {
        color: #4CAF50;
    }
    
    /* 任务树状图样式 */
    .task-tree {
        margin-top: 10px;
        padding: 12px;
        background: #f0f7ff;
        border-radius: 6px;
        border: 1px solid #b3d9ff;
    }
    
    .task-tree-empty {
        color: #888;
        font-style: italic;
        padding: 8px;
        text-align: center;
    }
    
    .task-category {
        margin-bottom: 10px;
    }
    
    .task-category-summary {
        display: flex;
        align-items: center;
        gap: 8px;
        cursor: pointer;
        padding: 8px;
        background: white;
        border-radius: 4px;
        border: 1px solid #b3d9ff;
        font-weight: 600;
        color: #2196F3;
        list-style: none;
        outline: none;
        transition: background-color 0.2s;
    }
    
    .task-category-summary:hover {
        background: #e3f2fd;
    }
    
    .task-category-summary::-webkit-details-marker {
        display: none;
    }
    
    .task-category[open] .task-category-summary {
        border-bottom-left-radius: 0;
        border-bottom-right-radius: 0;
        border-bottom: none;
    }
    
    .task-category-name {
        flex: 1;
    }
    
    .task-count {
        font-size: 12px;
        color: #666;
        font-weight: normal;
    }
    
    .task-list {
        margin: 0;
        padding: 8px 8px 8px 24px;
        background: white;
        border: 1px solid #b3d9ff;
        border-top: none;
        border-radius: 0 0 4px 4px;
        list-style: none;
    }
    
    .task-item {
        padding: 6px 8px;
        margin: 4px 0;
        background: #f5f5f5;
        border-radius: 4px;
        border-left: 3px solid #4CAF50;
        color: #333;
    }
    
    .task-reason {
        padding: 6px 8px;
        margin-top: 8px;
        font-size: 11px;
        color: #666;
        font-style: italic;
        border-top: 1px dashed #ddd;
        padding-top: 8px;
    }
    
    .task-summary {
        margin-top: 12px;
        padding: 8px;
        background: #e8f5e9;
        border-radius: 4px;
        border: 1px solid #4CAF50;
        font-weight: 600;
        color: #2e7d32;
        text-align: center;
    }
    /* Organize & Clean Table 进度样式 */
    .organize-progress {
        margin-top: 10px;
        padding: 14px;
        background: linear-gradient(135deg, #f0f7ff 0%, #e8f4f8 100%);
        border-radius: 8px;
        border: 1px solid #b3d9ff;
    }

    .organize-progress-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }

    .organize-title {
        font-weight: 700;
        font-size: 14px;
        color: #1a73e8;
    }

    .organize-counter {
        font-size: 12px;
        color: #5f6368;
        font-weight: 600;
    }

    .organize-progress-bar-bg {
        width: 100%;
        height: 6px;
        background: #e0e0e0;
        border-radius: 3px;
        margin-bottom: 12px;
        overflow: hidden;
    }

    .organize-progress-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #4CAF50, #66BB6A);
        border-radius: 3px;
        transition: width 0.5s ease;
    }

    .organize-task-list {
        list-style: none;
        padding: 0;
        margin: 0;
        display: flex;
        flex-direction: column;
        gap: 6px;
    }

    .organize-task-item {
        display: flex;
        align-items: flex-start;
        gap: 8px;
        padding: 8px 10px;
        border-radius: 6px;
        transition: all 0.3s ease;
    }

    .organize-task-completed {
        background: #e8f5e9;
        border: 1px solid #a5d6a7;
    }

    .organize-task-completed .organize-task-desc {
        text-decoration: line-through;
        color: #81c784;
    }

    .organize-task-active {
        background: #fff3e0;
        border: 1px solid #ffcc80;
        box-shadow: 0 0 8px rgba(255, 152, 0, 0.2);
        animation: pulse-border 2s infinite;
    }

    @keyframes pulse-border {
        0%, 100% { box-shadow: 0 0 4px rgba(255, 152, 0, 0.15); }
        50% { box-shadow: 0 0 12px rgba(255, 152, 0, 0.35); }
    }

    .organize-task-pending {
        background: #f5f5f5;
        border: 1px solid #e0e0e0;
        opacity: 0.65;
    }

    .organize-task-icon {
        font-size: 16px;
        flex-shrink: 0;
        margin-top: 1px;
    }

    .organize-task-info {
        display: flex;
        flex-direction: column;
        gap: 2px;
        min-width: 0;
    }

    .organize-task-desc {
        font-size: 12px;
        color: #333;
        line-height: 1.4;
        word-break: break-word;
    }

    .organize-task-meta {
        font-size: 10px;
        color: #888;
    }
    </style>
    '''

    return css + ''.join(html_parts)


if __name__ == "__main__":
    # 禁用右侧工具面板更新时的泛白虚化效果
    custom_css = """
    #tool_flow_view .pending {
        opacity: 1 !important;
    }
    #tool_flow_view.pending {
        opacity: 1 !important;
    }
    #tool_flow_view > .wrap {
        opacity: 1 !important;
        pointer-events: auto !important;
    }
    #tool_flow_view .generating {
        opacity: 1 !important;
    }
    #tool_flow_view.generating {
        opacity: 1 !important;
    }
    """
    auto_scroll_js = """
    () => {
        // Auto-scroll tool flow panel to bottom on content change
        const target = document.querySelector('#tool_flow_view');
        if (target && !target._autoScrollObserver) {
            const observer = new MutationObserver(() => {
                const container = target.querySelector('.tool-flow-container');
                if (container) container.scrollTop = container.scrollHeight;
            });
            observer.observe(target, { childList: true, subtree: true });
            target._autoScrollObserver = true;
        }
    }
    """
    with gr.Blocks(elem_id="root", fill_height=True, css=custom_css, js=auto_scroll_js) as demo:
        asr_model: ASR_Model = ASR_Model()
        session_inst: Session = Session()
        chat_history: list = []

        # 创建 UI 状态机
        ui_state_machine = UIStateMachine()

        # 用户输入处理标志：True 表示用户输入正在处理，自动推理应该跳过
        user_input_processing = asyncio.Event()
        user_input_processing.clear()  # 初始状态：没有用户输入处理中

        # Data analysis function
        async def perform_data_analysis():
            """Perform t-SNE analysis on LeRobot dataset and open in browser"""
            try:
                logger.info("Starting data analysis...")

                if session_inst.config_error:
                    return f"❌ API configuration error: {session_inst.config_error}"

                # Get the MCP service manager
                await session_inst.ensure_initialized()
                assert session_inst.agent is not None

                service_manager = session_inst.agent._service_manager

                # Check if data_analyst service is available
                service_names = [s.service_name for s in service_manager._services_register_list]
                if "data_analyst_mcp_server" not in service_names:
                    return "❌ 数据分析服务未初始化。请检查服务配置或日志。"

                # Call the full_analysis tool
                logger.info("Calling data_analyst_mcp_server.full_analysis...")
                result = await service_manager.tools_routing(
                    service_name="data_analyst_mcp_server",
                    tool_name="full_analysis",
                    tool_args={"feature_types": ["state", "action"], "output_path": "/tmp/tsne_analysis.html"},
                )

                # Parse result
                if result and result.text:
                    logger.info(f"Analysis result: {result.text}")
                    try:
                        result_data = json.loads(result.text)

                        html_path = result_data.get("visualization", "/tmp/tsne_analysis.html")

                        # Open in browser with better error handling
                        logger.info(f"Opening visualization in browser: {html_path}")
                        import os
                        import subprocess

                        # Check if file exists
                        if os.path.exists(html_path):
                            logger.info(f"HTML file found at: {html_path}")
                            try:
                                # Try to open with default browser
                                webbrowser.open(f"file://{html_path}")
                                logger.info("Browser opened successfully")
                            except Exception as browser_error:
                                logger.warning(f"Failed to open browser: {browser_error}")
                                # Try alternative methods
                                try:
                                    if os.name == "posix":  # Linux/Mac
                                        subprocess.Popen(["xdg-open", html_path])
                                    elif os.name == "nt":  # Windows
                                        os.startfile(html_path)
                                    logger.info("Opened HTML file using alternative method")
                                except Exception as alt_error:
                                    logger.warning(f"Alternative open method also failed: {alt_error}")
                        else:
                            logger.warning(f"HTML file not found at: {html_path}")

                        return (
                            "✅ Data analysis completed!\n\n"
                            f"Dataset: {result_data['dataset']['dataset_name']}\n"
                            f"Total samples: {result_data['dataset']['total_samples']}\n\n"
                            f"📊 t-SNE visualization: {html_path}\n"
                            "(An attempt has been made to open it in your browser)\n\n"
                            f"Analyzed features: {', '.join(result_data['analysis'].keys())}"
                        )
                    except json.JSONDecodeError:
                        # If result is not JSON, just return the text
                        return f"✅ Data analysis completed!\n\n{result.text}"
                else:
                    return "⚠️ Data analysis completed, but no result was returned."

            except Exception as e:
                logger.error(f"Data analysis error: {e}", exc_info=True)
                return f"❌ Data analysis failed: {str(e)}"

        # 主布局：左侧聊天区域 + 右侧工具调用流程
        with gr.Row():
            # 左侧：聊天区域
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    value=snapshot_chat_history(chat_history),
                    elem_id="chatbot",
                    type="messages",
                    show_label=False,
                    scale=1,
                )

                # 状态显示 / State display
                state_display = gr.Textbox(
                    value=ui_state_machine.get_state_display(),
                    label="Inference Status",
                    interactive=False,
                    scale=0,
                    lines=6,
                )

                chat_input = gr.MultimodalTextbox(
                    interactive=True,
                    file_count="multiple",
                    file_types=[".wav"],
                    placeholder="Enter message...",
                    show_label=False,
                    sources=["microphone"],
                    scale=0,
                )

            # 右侧：工具调用流程面板 / Tool call flow panel
            with gr.Column(scale=1):
                tool_flow_title = gr.Markdown("### 🛠 Tool Call Flow")
                tool_flow_view = gr.HTML(
                    value=(
                        "<div class='tool-flow-container'>"
                        "<p style='color: #888; padding: 20px; text-align: center;'>"
                        "Waiting for tool calls..."
                        "</p></div>"
                    ),
                    elem_id="tool_flow_view",
                    label="",
                    show_label=False,
                )

        # 机器人相机图像显示
        # with gr.Row():
        #     robot_image = gr.Image(
        #         label="机器人视角 (拼接图像)",
        #         show_label=True,
        #         type="numpy",
        #         height=400,
        #     )

        # Data Analysis Status
        with gr.Row():
            analysis_status = gr.Plot(
                label="Analysis Status",
                value=None,
            )

        def _build_chat_outputs():
            tool_history = session_inst.get_tool_call_history()
            tool_flow_html = render_tool_flow(tool_history)
            viz_path = extract_visualization_path(tool_history) if tool_history else None
            analysis_figure = load_plotly_figure(viz_path)
            return tool_history, tool_flow_html, analysis_figure

        async def _stream_user_message(user_message: str):
            prepared_message = session_inst.prepare_message_for_agent(user_message)
            chat_history.append({"role": "user", "content": user_message})
            if prepared_message.error_message:
                chat_history.append({"role": "assistant", "content": prepared_message.error_message})
                _, tool_flow_html, analysis_figure = _build_chat_outputs()
                yield chat_history, tool_flow_html, analysis_figure
                return

            agent_message = prepared_message.message or user_message
            chat_history.append({"role": "assistant", "content": "_Thinking..._"})

            assistant_text = ""
            status_text = prepared_message.status_message or "Thinking..."
            final_text = ""

            try:
                await ui_state_machine.send_event(UIEvent.INFER_START)
                async for event in session_inst.run_once_stream(agent_message):
                    if event["type"] == "status":
                        status_text = event.get("text", status_text) or status_text
                    elif event["type"] == "text_delta":
                        assistant_text += event.get("delta", "")
                    elif event["type"] == "final":
                        final_text = event.get("text", "") or ""

                    chat_history[-1]["content"] = build_streaming_assistant_content(assistant_text, status_text)
                    _, tool_flow_html, analysis_figure = _build_chat_outputs()
                    yield chat_history, tool_flow_html, analysis_figure

                final_text = final_text or assistant_text or "No result returned."
                final_text = _maybe_override_organize_msg(session_inst, final_text)
                final_text = _maybe_override_rollout_msg(session_inst, final_text)
                formatted_msg = format_response_text(final_text)
                chat_history[-1]["content"] = formatted_msg
                await ui_state_machine.send_event(UIEvent.INFER_SUCCESS)
            except Exception as e:
                logger.error(f"[chat] 推理失败: {e}")
                await ui_state_machine.send_event(UIEvent.INFER_FAIL)
                if assistant_text:
                    chat_history[-1]["content"] = f"{assistant_text}\n\n_推理失败: {e}_"
                else:
                    chat_history[-1]["content"] = f"推理失败: {e}"

            _, tool_flow_html, analysis_figure = _build_chat_outputs()
            yield chat_history, tool_flow_html, analysis_figure

        async def chat(user_input):
            """处理用户输入，发送 USER_INPUT 事件"""
            # 尝试转移到 WAITING_INFER 状态
            if not await ui_state_machine.send_event(UIEvent.USER_INPUT):
                logger.warning(f"[chat] 当前状态 {ui_state_machine.state.value} 不允许接收新输入，等待推理完成...")
                # 等待推理完成，然后重试
                await ui_state_machine.wait_inference_done()
                if not await ui_state_machine.send_event(UIEvent.USER_INPUT):
                    logger.error(f"[chat] 等待后仍无法接收输入，当前状态: {ui_state_machine.state.value}")
                    _, tool_flow_html, analysis_figure = _build_chat_outputs()
                    yield chat_history, tool_flow_html, analysis_figure
                    return

            # 设置用户输入处理标志，阻止自动推理
            user_input_processing.set()
            try:
                if user_input["files"]:
                    for file in user_input["files"]:
                        audio_path = file
                        user_transcript: str = asr_model.transcribe(audio_path)
                        async for payload in _stream_user_message(user_transcript):
                            yield payload
                    return
                if user_input["text"]:
                    async for payload in _stream_user_message(str(user_input["text"])):
                        yield payload
                    return
                else:
                    _, tool_flow_html, analysis_figure = _build_chat_outputs()
                    yield chat_history, tool_flow_html, analysis_figure
                    return
            finally:
                # 用户输入处理完成，清除标志，允许自动推理
                user_input_processing.clear()

        chat_msg = chat_input.submit(chat, [chat_input], [chatbot, tool_flow_view, analysis_status])
        chat_msg.then(lambda: "", None, [chat_input])  # 返回空字符串清空输入框内容
        chat_msg.then(lambda: ui_state_machine.get_state_display(), None, [state_display])  # 更新状态显示

        feishu_long_connection_holder: dict[str, Any] = {"runtime": None}

        async def _start_feishu_long_connection_once() -> None:
            if feishu_long_connection_holder["runtime"] is not None:
                return
            try:
                feishu_long_connection_holder["runtime"] = start_feishu_long_connection(
                    session_inst=session_inst,
                    ui_state_machine=ui_state_machine,
                    user_input_processing=user_input_processing,
                    chat_history=chat_history,
                    format_bot_msg=_format_bot_msg_for_feishu,
                    build_status_text=_build_status_for_feishu,
                    get_image_bytes=_get_latest_snapshot_bytes,
                    source_name="gradio_ui",
                )
            except Exception as e:
                logger.warning(f"[FeishuLongConnection] Startup skipped: {type(e).__name__}: {e}")

        async def _shutdown_feishu_long_connection() -> None:
            runtime = feishu_long_connection_holder["runtime"]
            if runtime is None:
                return
            try:
                await runtime.aclose()
            except Exception as e:
                logger.warning(f"[FeishuLongConnection] Shutdown failed: {type(e).__name__}: {e}")
            finally:
                feishu_long_connection_holder["runtime"] = None

        # 应用加载时初始化代理（确保定时取图前完成初始化）
        async def _on_load():
            if session_inst.config_error:
                logger.error("Skip Gradio agent init because API config is invalid: %s", session_inst.config_error)
                return None
            await session_inst.ensure_initialized()
            await _start_feishu_long_connection_once()
            return None

        demo.load(_on_load, inputs=None, outputs=None)

        def _format_bot_msg_for_feishu(raw: str) -> str:
            msg = _maybe_override_organize_msg(session_inst, raw)
            msg = _maybe_override_rollout_msg(session_inst, msg)
            return format_response_text(msg)

        def _build_status_for_feishu() -> str:
            return build_detailed_state_text(ui_state_machine, session_inst, user_input_processing)

        async def _get_latest_snapshot_bytes() -> bytes | None:
            try:
                if not session_inst.initialized:
                    await session_inst.agent.init_agent()
                    session_inst.initialized = True
                dataloader = session_inst._agent_card.robot_dataloader
                if dataloader is None:
                    return None
                a2d_data = await dataloader.get_latest_concatenate_image_base64()
                if a2d_data and a2d_data.concatenated_image_base64:
                    return base64.b64decode(a2d_data.concatenated_image_base64)
            except Exception as e:
                logger.warning(f"[FeishuBot] Failed to get snapshot: {type(e).__name__}: {e}")
            return None

        demo.app.add_event_handler("startup", _start_feishu_long_connection_once)
        demo.app.add_event_handler("shutdown", _shutdown_feishu_long_connection)

        # 高频图像获取：每0.2秒获取一次最新图像并更新显示（快速消费DDS消息，避免缓冲区溢出）
        # async def _tick_img():
        #     img_rgb = None
        #     try:
        #         if session_inst.initialized:
        #             # 获取图像数据
        #             a2d_data = await session_inst._agent_card.robot_dataloader.get_latest_concatenate_image_base64()
        #             if a2d_data is not None:
        #                 # 处理图像用于前端显示
        #                 if a2d_data.concatenated_image is not None:
        #                     img_rgb = cv2.cvtColor(a2d_data.concatenated_image, cv2.COLOR_BGR2RGB)
        #                 elif a2d_data.concatenated_image_base64:
        #                     try:
        #                         img_bytes = base64.b64decode(a2d_data.concatenated_image_base64)
        #                         nparr = np.frombuffer(img_bytes, np.uint8)
        #                         img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        #                         if img is not None:
        #                             img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #                     except Exception as e:
        #                         logger.warning(f"解码 base64 图像失败: {e}")
        #     except Exception as e:
        #         logger.warning(f"tick_img error: {e}")
        #
        #     # 只返回图像，不更新聊天历史
        #     return img_rgb

        # 低频自动推理：每5秒写入记忆并执行一次自动推理（仅在 WAITING_INFER 状态执行）
        async def _tick_inference():
            # #region agent log
            try:
                with open("/home/agiuser/Project_Olympus_zyl/.cursor/debug.log", "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "id": f"log_{int(time.time()*1000)}",
                                "timestamp": int(time.time() * 1000),
                                "location": "gradio_ui.py:728",
                                "message": "_tick_inference triggered",
                                "data": {
                                    "state": ui_state_machine.state.value,
                                    "user_input_processing": user_input_processing.is_set(),
                                },
                                "runId": "debug",
                                "hypothesisId": "B",
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # #endregion
            try:
                if session_inst.config_error:
                    tool_flow_html = (
                        "<div class='tool-flow-container'>"
                        f"<p style='color: #c62828; padding: 20px;'>API configuration error: {escape(session_inst.config_error)}</p>"
                        "</div>"
                    )
                    return chat_history, tool_flow_html, None

                # 状态机检查：只有在 WAITING_INFER 状态才执行推理
                if ui_state_machine.state != UIState.WAITING_INFER:
                    logger.debug(f"[自动推理] 当前状态 {ui_state_machine.state.value}，不执行推理")
                    tool_history = session_inst.get_tool_call_history()
                    tool_flow_html = render_tool_flow(tool_history) if tool_history else ""
                    # 检查是否有可视化文件
                    viz_path = extract_visualization_path(tool_history) if tool_history else None
                    analysis_figure = load_plotly_figure(viz_path)
                    return chat_history, tool_flow_html, analysis_figure

                # 检查是否有用户输入正在处理，如果有则跳过本次自动推理
                if user_input_processing.is_set():
                    logger.debug("[自动推理] 用户输入正在处理中，跳过本次自动推理")
                    tool_history = session_inst.get_tool_call_history()
                    tool_flow_html = render_tool_flow(tool_history) if tool_history else ""
                    # 检查是否有可视化文件
                    viz_path = extract_visualization_path(tool_history) if tool_history else None
                    analysis_figure = load_plotly_figure(viz_path)
                    return chat_history, tool_flow_html, analysis_figure

                if session_inst.initialized:
                    assert session_inst._agent_card is not None
                    assert session_inst.agent is not None
                    a2d_data = await session_inst._agent_card.robot_dataloader.get_latest_concatenate_image_base64()
                    if a2d_data is not None:
                        await session_inst.agent.add_img_data_to_memory()

                        # 再次检查用户输入状态（可能在写入记忆期间用户输入了）
                        if user_input_processing.is_set():
                            logger.debug("[自动推理] 用户输入已开始，跳过本次自动推理")
                            tool_history = session_inst.get_tool_call_history()
                            tool_flow_html = render_tool_flow(tool_history) if tool_history else ""
                            # 检查是否有可视化文件
                            viz_path = extract_visualization_path(tool_history) if tool_history else None
                            analysis_figure = load_plotly_figure(viz_path)
                            return chat_history, tool_flow_html, analysis_figure

                        # 再次检查状态机状态（可能已被用户输入改变）
                        if ui_state_machine.state != UIState.WAITING_INFER:
                            logger.debug(f"[自动推理] 状态已变更为 {ui_state_machine.state.value}，跳过推理")
                            tool_history = session_inst.get_tool_call_history()
                            tool_flow_html = render_tool_flow(tool_history) if tool_history else ""
                            # 检查是否有可视化文件
                            viz_path = extract_visualization_path(tool_history) if tool_history else None
                            analysis_figure = load_plotly_figure(viz_path)
                            return chat_history, tool_flow_html, analysis_figure

                        # 发送 INFER_START 事件
                        if await ui_state_machine.send_event(UIEvent.INFER_START):
                            logger.debug("[自动推理] 发送 INFER_START 事件，开始推理")
                            # #region agent log
                            try:
                                with open("/home/agiuser/Project_Olympus_zyl/.cursor/debug.log", "a") as f:
                                    f.write(
                                        json.dumps(
                                            {
                                                "id": f"log_{int(time.time()*1000)}",
                                                "timestamp": int(time.time() * 1000),
                                                "location": "gradio_ui.py:766",
                                                "message": "run_once starting (auto inference)",
                                                "data": {"prompt": AUTO_INFERENCE_PROMPT, "clear_history": False},
                                                "runId": "debug",
                                                "hypothesisId": "A",
                                            }
                                        )
                                        + "\n"
                                    )
                            except Exception:
                                pass
                            # #endregion

                            # 自动推理（不清空工具调用历史，保留之前的调用记录）
                            try:
                                # #region agent log
                                try:
                                    with open("/home/agiuser/Project_Olympus_zyl/.cursor/debug.log", "a") as f:
                                        f.write(
                                            json.dumps(
                                                {
                                                    "id": f"log_{int(time.time()*1000)}",
                                                    "timestamp": int(time.time() * 1000),
                                                    "location": "gradio_ui.py:898",
                                                    "message": "Calling Session.run_once with clear_history=False",
                                                    "data": {
                                                        "prompt_preview": (
                                                            AUTO_INFERENCE_PROMPT[:50]
                                                            if len(AUTO_INFERENCE_PROMPT) > 50
                                                            else AUTO_INFERENCE_PROMPT
                                                        )
                                                    },
                                                    "runId": "debug",
                                                    "hypothesisId": "A",
                                                }
                                            )
                                            + "\n"
                                        )
                                except Exception:
                                    pass
                                # #endregion
                                bot_msg = await session_inst.run_once(AUTO_INFERENCE_PROMPT, clear_history=False)
                                # #region agent log
                                try:
                                    with open("/home/agiuser/Project_Olympus_zyl/.cursor/debug.log", "a") as f:
                                        f.write(
                                            json.dumps(
                                                {
                                                    "id": f"log_{int(time.time()*1000)}",
                                                    "timestamp": int(time.time() * 1000),
                                                    "location": "gradio_ui.py:771",
                                                    "message": "run_once completed (auto inference)",
                                                    "data": {"result_length": len(bot_msg) if bot_msg else 0},
                                                    "runId": "debug",
                                                    "hypothesisId": "A",
                                                }
                                            )
                                            + "\n"
                                        )
                                except Exception:
                                    pass
                                # #endregion
                                # Check if organize_and_clean_table or rollout_task completed
                                bot_msg = _maybe_override_organize_msg(session_inst, bot_msg)
                                bot_msg = _maybe_override_rollout_msg(session_inst, bot_msg)
                                # 过滤JSON响应，转换为自然语言
                                formatted_msg = format_response_text(bot_msg)
                                # 如果 rollout 任务已完成且完成消息已发送，跳过自动推理的聊天追加
                                if not _has_active_completed_rollout(session_inst):
                                    chat_history.append({"role": "assistant", "content": formatted_msg})
                                logger.info(f"[自动推理] 推理成功: {formatted_msg}")

                                # 发送推理成功事件
                                await ui_state_machine.send_event(UIEvent.INFER_SUCCESS)

                                # 获取工具调用历史并渲染
                                tool_history = session_inst.get_tool_call_history()
                                tool_flow_html = render_tool_flow(tool_history)
                                # 检查是否有可视化文件
                                viz_path = extract_visualization_path(tool_history) if tool_history else None
                                analysis_figure = load_plotly_figure(viz_path)
                                return chat_history, tool_flow_html, analysis_figure

                            except Exception as e:
                                logger.warning(f"自动推理失败: {e}")
                                # 发送推理失败事件
                                await ui_state_machine.send_event(UIEvent.INFER_FAIL)
                                tool_history = session_inst.get_tool_call_history()
                                tool_flow_html = render_tool_flow(tool_history) if tool_history else ""
                                # 检查是否有可视化文件
                                viz_path = extract_visualization_path(tool_history) if tool_history else None
                                analysis_figure = load_plotly_figure(viz_path)
                                return chat_history, tool_flow_html, analysis_figure
                        else:
                            logger.debug("[自动推理] INFER_START 事件发送失败")
            except Exception as e:
                logger.warning(f"tick_inference error: {e}")
                tool_history = session_inst.get_tool_call_history()
                tool_flow_html = render_tool_flow(tool_history) if tool_history else ""
                # 检查是否有可视化文件
                viz_path = extract_visualization_path(tool_history) if tool_history else None
                analysis_figure = load_plotly_figure(viz_path)
                return chat_history, tool_flow_html, analysis_figure

            # 返回更新后的聊天历史和工具调用流程
            tool_history = session_inst.get_tool_call_history()
            tool_flow_html = render_tool_flow(tool_history) if tool_history else ""
            # 检查是否有可视化文件
            viz_path = extract_visualization_path(tool_history) if tool_history else None
            analysis_figure = load_plotly_figure(viz_path)
            return chat_history, tool_flow_html, analysis_figure

        # 高频图像更新：0.2秒（5 FPS），快速消费DDS消息
        # gr.Timer(0.2).tick(_tick_img, inputs=None, outputs=[robot_image])

        # 低频自动推理：5秒，减少计算负担
        gr.Timer(5.0).tick(_tick_inference, inputs=None, outputs=[chatbot, tool_flow_view, analysis_status])

        # 状态显示更新：每1秒更新一次（详细状态）
        async def _tick_state():
            return build_detailed_state_text(ui_state_machine, session_inst, user_input_processing)

        gr.Timer(1.0).tick(_tick_state, inputs=None, outputs=[state_display])

        # 聊天窗口实时刷新：让飞书长连接等后台写入也能显示到 Gradio 界面。
        async def _tick_chatbot_live():
            return snapshot_chat_history(chat_history)

        gr.Timer(1.0).tick(_tick_chatbot_live, inputs=None, outputs=[chatbot])

        # 工具流程面板实时刷新：每1秒更新一次
        async def _tick_tool_flow_live():
            if session_inst.config_error:
                return (
                    "<div class='tool-flow-container'>"
                    f"<p style='color: #c62828; padding: 20px;'>API configuration error: {escape(session_inst.config_error)}</p>"
                    "</div>"
                )
            if session_inst.initialized:
                tool_history = session_inst.get_tool_call_history()
                return render_tool_flow(tool_history)
            return '<div class="tool-flow-container"><p style="color: #888; padding: 20px; text-align: center;">等待工具调用...</p></div>'

        gr.Timer(1.0).tick(_tick_tool_flow_live, inputs=None, outputs=[tool_flow_view])

        # 卸载时清理资源
        def _on_unload():
            try:
                if session_inst._agent_card is not None and session_inst._agent_card.robot_dataloader is not None:
                    session_inst._agent_card.robot_dataloader.shutdown()
            except Exception:
                pass

        demo.unload(_on_unload)

    # Enable Gradio's request queue so async-generator yields are pushed to the UI incrementally.
    demo.queue()

    no_proxy = os.environ.get("no_proxy", "")
    if "localhost" not in no_proxy:
        os.environ["no_proxy"] = ",".join(filter(None, [no_proxy, "localhost,127.0.0.1,0.0.0.0"]))

    demo.launch(server_name="0.0.0.0", server_port=11233)
