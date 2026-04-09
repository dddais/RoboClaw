import uuid
import json
import logging
import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Optional, Awaitable, Callable
from datetime import datetime
from agent_demo.types.interaction_types import InteractionPackage
from ..base_agent.base_agent import BaseAgent
from agent_demo.types.agent_types import (
    OpenAISendMsg,
    OpenAIResponseMsg,
    ORMCP_TOOLS_SPLICE,
    TextParam,
    ActAgentState,
    BaseAgentCard,
)
from agent_demo.types.machine_layer import A2DData

logger = logging.getLogger(name=__name__)


StreamTextCallback = Callable[[str], Awaitable[None]]
StreamStatusCallback = Callable[[str], Awaitable[None]]


@dataclass
class ToolCallRecord:
    """工具调用记录"""

    id: str
    service_name: str
    tool_name: str
    args_preview: str
    result_preview: str = ""
    step_index: int = 0
    parent_step_index: Optional[int] = None
    timestamp: str = field(default_factory=lambda: datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
    task_result: Optional[dict] = None  # 用于存储任务判断的结构化结果


class ImgActAgent(BaseAgent):

    _agent_name: str = "Hephaestus"
    _MAX_STRUCTURED_SKILL_DELEGATIONS: int = 4

    def __init__(
        self,
        agent_card: BaseAgentCard,
    ):
        super().__init__(agent_card=agent_card)
        self._agent_card_ref.agent_name = self._agent_name
        self._agent_card_ref.agent_id = self._agent_name + "-" + uuid.uuid4().hex[:8]
        self._cancel_requested = asyncio.Event()
        # 工具调用历史记录
        self.tool_call_history: list[ToolCallRecord] = []
        self._tool_call_step_counter: int = 0
        # 标记当前轮用户输入是否为 reset，用于控制某些工具的自动调用
        self._last_user_input_was_reset: bool = False

    def request_cancel(self) -> None:
        """Best-effort cancellation signal for the currently running run_once."""
        self._cancel_requested.set()

    def clear_cancel(self) -> None:
        self._cancel_requested.clear()

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_requested.is_set()

    def _extract_json_object_from_text(self, text: str) -> dict | None:
        if not text:
            return None

        decoder = json.JSONDecoder()
        stripped = text.strip()
        candidates: list[str] = [stripped]
        candidates.extend(stripped[idx:] for idx, char in enumerate(stripped) if char == "{")

        for candidate in candidates:
            try:
                parsed, _end_idx = decoder.raw_decode(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        return None

    def _extract_structured_skill_delegation(self, messages: OpenAIResponseMsg) -> dict[str, object] | None:
        content = messages.first_choice.message.content
        if content is None:
            return None

        payload = self._extract_json_object_from_text(content.text)
        if not isinstance(payload, dict):
            return None

        status = str(payload.get("status", "")).strip().lower()
        next_skill = payload.get("next_skill")
        skill_args = payload.get("skill_args")
        if status != "continue" or not isinstance(next_skill, str) or not isinstance(skill_args, dict):
            return None

        source_skill = payload.get("selected_skill")
        if not isinstance(source_skill, str):
            source_skill = None

        return {
            "status": status,
            "next_skill": next_skill,
            "skill_args": skill_args,
            "source_skill": source_skill,
        }

    # ========= 核心方法 =========
    # user -> chat -> act -> response -> compress_memory
    async def init_agent(self):
        logger.info("[Init][Agent][Start]")
        await self._llm_client.init_client()
        await self._memory_manager.init_memory()
        await self._service_manager.init_services()
        await self._agent_tools.init_agent_tools()
        self._transition_state(ActAgentState.READY)
        logger.info("[Init][Agent][Done]")

    async def chat(
        self,
        messages: str,
        on_text_delta: StreamTextCallback | None = None,
    ) -> OpenAIResponseMsg:
        self._transition_state(ActAgentState.CHAT)
        await self._memory_manager.add_user_str_message(messages)
        res: OpenAIResponseMsg = await self._llm_client.sync_chat(
            send_package=OpenAISendMsg(contexts=self.current_contexts, tools_list=self.available_tools),
            on_text_delta=on_text_delta,
        )
        self._current_total_tokens = res.total_tokens
        await self._memory_manager.add_agent_message_type(msg=res.first_choice.message)
        return res

    async def act(self, messages: OpenAIResponseMsg) -> TextParam:
        self._transition_state(ActAgentState.ACT)
        res: TextParam = TextParam(text="未知错误")
        for tool_call in messages.first_choice.tool_calls:
            tool_call_id = tool_call.id
            service_name, tool_name = tool_call.function.name.split(ORMCP_TOOLS_SPLICE)
            tool_args: dict[str, object] = json.loads(tool_call.function.arguments)
            logger.info(f"[Routing Info] ---> [{service_name}][{tool_name}][{tool_args}]")
            self.display_deque.append(
                InteractionPackage(
                    content_type="analyze_response",
                    agent_id=self.agent_id,
                    content=f"[Routing Info] ---> [{service_name}][{tool_name}][{tool_args}]",
                )
            )

            # 创建工具调用记录（调用前）
            self._tool_call_step_counter += 1
            args_preview = json.dumps(tool_args, ensure_ascii=False)
            if len(args_preview) > 200:
                args_preview = args_preview[:197] + "..."

            tool_record = ToolCallRecord(
                id=tool_call_id,
                service_name=service_name,
                tool_name=tool_name,
                args_preview=args_preview,
                step_index=self._tool_call_step_counter,
            )
            self.tool_call_history.append(tool_record)

            # 执行工具调用
            # #region agent log
            try:
                with open("/home/agiuser/Project_Olympus_zyl/.cursor/debug.log", "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "id": f"log_{int(time.time()*1000)}",
                                "timestamp": int(time.time() * 1000),
                                "location": "img_act_agent.py:112",
                                "message": "tool routing before call",
                                "data": {"service_name": service_name, "tool_name": tool_name},
                                "runId": "debug",
                                "hypothesisId": "A",
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # #endregion

            # 特殊规则：在用户刚刚执行 reset 指令的这一轮对话中，不自动调用任务判断工具
            if (
                self._last_user_input_was_reset
                and service_name == self._agent_tools._service_name
                and tool_name == "detect_tasks_from_image"
            ):
                logger.info("[ImgActAgent.act] 跳过 detect_tasks_from_image 调用（上一轮用户输入为 reset）")
                # 直接返回说明性文本，避免真正调用任务判断
                res = TextParam(
                    text="已根据您的指令重置当前任务，本轮不自动进行图像任务判断。如需重新分析，请在重置后明确说明要分析的任务。"
                )
                # 更新工具调用记录
                result_preview = res.text
                if len(result_preview) > 200:
                    result_preview = result_preview[:197] + "..."
                tool_record.result_preview = result_preview
                await self._memory_manager.add_robot_call_back_text_message(res, tool_call_id)
                await self._agent_tools.flush_pending_context_injections()
                # 跳过本次循环的后续处理
                continue

            if self._service_manager.check_is_agent_service(service_name):  # agent service
                res = await self._agent_tools.tools_routing(service_name, tool_name, tool_args)
            else:  # mcp service
                res = await self._service_manager.tools_routing(service_name, tool_name, tool_args)
            # #region agent log
            try:
                with open("/home/agiuser/Project_Olympus_zyl/.cursor/debug.log", "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "id": f"log_{int(time.time()*1000)}",
                                "timestamp": int(time.time() * 1000),
                                "location": "img_act_agent.py:112",
                                "message": "tool routing after call",
                                "data": {"service_name": service_name, "tool_name": tool_name},
                                "runId": "debug",
                                "hypothesisId": "A",
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # #endregion

            # 更新工具调用记录（调用后，填充结果）
            result_preview = res.text if res else "无结果"
            if len(result_preview) > 200:
                result_preview = result_preview[:197] + "..."
            tool_record.result_preview = result_preview

            # 如果是任务判断工具，解析并存储结构化结果
            if tool_name == "detect_tasks_from_image" and res:
                try:
                    task_result = json.loads(res.text)
                    # 验证结构
                    if isinstance(task_result, dict) and "categories" in task_result:
                        tool_record.task_result = task_result
                        logger.info(f"[act] 已存储任务判断结果: {len(task_result.get('categories', []))} 个分类")
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"[act] 解析任务判断结果失败: {e}")
            elif tool_name == "run_skill" and res:
                try:
                    skill_result = json.loads(res.text)
                    if isinstance(skill_result, dict) and "skill_name" in skill_result:
                        response_text = skill_result.get("response")
                        if isinstance(response_text, str):
                            json_match = re.search(r"\{[\s\S]*\}|\[[\s\S]*\]", response_text.strip())
                            if json_match:
                                try:
                                    skill_result["structured_response"] = json.loads(json_match.group(0))
                                except json.JSONDecodeError:
                                    pass
                        tool_record.task_result = skill_result
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"[act] 解析 skill 执行结果失败: {e}")

            await self._memory_manager.add_robot_call_back_text_message(res, tool_call_id)
            await self._agent_tools.flush_pending_context_injections()

        return res

    async def response(
        self,
        on_text_delta: StreamTextCallback | None = None,
    ) -> OpenAIResponseMsg:
        self._transition_state(ActAgentState.RESPONSE)
        res: OpenAIResponseMsg = await self._llm_client.sync_chat(
            OpenAISendMsg(contexts=self.current_contexts, tools_list=self.available_tools),
            on_text_delta=on_text_delta,
        )
        self._current_total_tokens = res.total_tokens
        await self._memory_manager.add_agent_message_type(msg=res.first_choice.message)
        return res

    async def compress_memory(
        self,
        messages: OpenAIResponseMsg,
        on_text_delta: StreamTextCallback | None = None,
    ):
        if messages.need_compress:
            self._transition_state(ActAgentState.COMPRESS)
            await self._memory_manager.add_compress_request_message()
            res: OpenAIResponseMsg = await self._llm_client.sync_chat(
                OpenAISendMsg(contexts=self.current_contexts, tools_list=self.available_tools),
                on_text_delta=on_text_delta,
            )
            self._current_total_tokens = res.total_tokens
            await self._memory_manager.add_agent_message_type(msg=res.first_choice.message)
            await self._memory_manager.compress_current_memory()

        self._transition_state(ActAgentState.READY)

    async def reset(self):
        pass

    async def run_once(
        self,
        user_input: str,
        clear_history: bool = True,
        on_text_delta: StreamTextCallback | None = None,
        on_status: StreamStatusCallback | None = None,
    ) -> TextParam | None:
        # new run, clear old cancel request
        self.clear_cancel()

        # 标记本轮输入是否为 reset，用于控制后续工具调用行为
        normalized_input = user_input.strip().lower()
        self._last_user_input_was_reset = normalized_input == "reset"

        # #region agent log
        try:
            with open("/home/agiuser/Project_Olympus_zyl/.cursor/debug.log", "a") as f:
                f.write(
                    json.dumps(
                        {
                            "id": f"log_{int(time.time()*1000)}",
                            "timestamp": int(time.time() * 1000),
                            "location": "img_act_agent.py:151",
                            "message": "run_once entry",
                            "data": {
                                "user_input_preview": user_input[:50] if len(user_input) > 50 else user_input,
                                "clear_history": clear_history,
                                "state": str(self._state),
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
        # 在开始处理前，清理可能存在的孤立 tool 消息（作为安全网）
        self.current_task_node._cleanup_orphaned_tool_messages()

        # 只在明确要求时清空工具调用历史（用户输入时清空，自动推理时不清空）
        if clear_history:
            self.tool_call_history.clear()
            self._tool_call_step_counter = 0

        # 确保状态从 READY 开始
        if self._state != ActAgentState.READY:
            logger.warning(f"[ImgActAgent] 状态不是 READY ({self._state})，强制重置为 READY")
            self._state = ActAgentState.READY

        try:
            if on_status is not None:
                await on_status("Thinking...")

            res: OpenAIResponseMsg = await self.chat(user_input, on_text_delta=on_text_delta)
            logger.info(f"[😼 ]{res.first_choice.message}")

            tool_call_count = 0
            delegated_skill_count = 0
            while True:
                while res.has_tool_call:
                    if self.cancel_requested:
                        logger.info("[ImgActAgent] Cancellation requested, aborting run_once before tool execution.")
                        raise asyncio.CancelledError()
                    tool_call_count += 1

                    if on_status is not None:
                        tool_names = [
                            tool_call.function.name.split(ORMCP_TOOLS_SPLICE, 1)[-1]
                            for tool_call in (res.first_choice.tool_calls or [])
                        ]
                        tool_summary = ", ".join(tool_names[:2]) if tool_names else f"#{tool_call_count}"
                        if len(tool_names) > 2:
                            tool_summary += "..."
                        await on_status(f"Executing tool: {tool_summary}")

                    # #region agent log
                    try:
                        with open("/home/agiuser/Project_Olympus_zyl/.cursor/debug.log", "a") as f:
                            f.write(
                                json.dumps(
                                    {
                                        "id": f"log_{int(time.time()*1000)}",
                                        "timestamp": int(time.time() * 1000),
                                        "location": "img_act_agent.py:168",
                                        "message": "tool call iteration",
                                        "data": {
                                            "iteration": tool_call_count,
                                            "tool_calls_count": (
                                                len(res.first_choice.tool_calls) if res.first_choice.tool_calls else 0
                                            ),
                                        },
                                        "runId": "debug",
                                        "hypothesisId": "C",
                                    }
                                )
                                + "\n"
                            )
                    except Exception:
                        pass
                    # #endregion

                    tool_res: TextParam = await self.act(res)

                    self._agent_card_ref.display_deque.append(
                        InteractionPackage(
                            content_type="analyze_response",
                            agent_id=self._agent_card_ref.agent_id,
                            content=f"...........{tool_res}",
                        )
                    )
                    logger.info(f"[🤖 ]{tool_res}")

                    if on_status is not None:
                        await on_status("Summarizing tool result...")
                    res = await self.response(on_text_delta=on_text_delta)
                    logger.info(f"[😼 ]{res.first_choice.message}")

                delegation = self._extract_structured_skill_delegation(res)
                if delegation is None:
                    break

                if delegated_skill_count >= self._MAX_STRUCTURED_SKILL_DELEGATIONS:
                    logger.warning(
                        "[ImgActAgent] Structured skill delegation limit reached (%s), stop auto-delegating.",
                        self._MAX_STRUCTURED_SKILL_DELEGATIONS,
                    )
                    break

                next_skill = str(delegation["next_skill"])
                skill_args = delegation["skill_args"]
                source_skill = delegation.get("source_skill")
                delegated_message = self._agent_tools.build_structured_skill_delegation_message(
                    skill_name=next_skill,
                    skill_args=skill_args,  # type: ignore[arg-type]
                    source_skill=source_skill if isinstance(source_skill, str) else None,
                )
                if not delegated_message:
                    logger.warning("[ImgActAgent] Failed to build delegated skill message for %s", next_skill)
                    break

                delegated_skill_count += 1
                logger.info(
                    "[ImgActAgent] Auto-delegating structured skill step %s -> %s",
                    source_skill or "(unknown source)",
                    next_skill,
                )
                if on_status is not None:
                    await on_status(f"Delegating to skill: {next_skill}")
                if self._state != ActAgentState.READY:
                    self._transition_state(ActAgentState.READY)
                res = await self.chat(delegated_message, on_text_delta=on_text_delta)
                logger.info(f"[😼 ]{res.first_choice.message}")

            if on_status is not None and res.need_compress:
                await on_status("Compressing memory...")
            await self.compress_memory(res, on_text_delta=on_text_delta)
            # #region agent log
            try:
                with open("/home/agiuser/Project_Olympus_zyl/.cursor/debug.log", "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "id": f"log_{int(time.time()*1000)}",
                                "timestamp": int(time.time() * 1000),
                                "location": "img_act_agent.py:180",
                                "message": "run_once exit",
                                "data": {"tool_call_count": tool_call_count},
                                "runId": "debug",
                                "hypothesisId": "A",
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # #endregion
            return res.first_choice.message.content  # type: ignore
        except Exception as e:
            # 发生异常时，确保状态回到 READY，避免状态卡在中间状态
            logger.error(f"[ImgActAgent] run_once 发生异常，重置状态为 READY: {type(e).__name__}: {e}")
            if self._state != ActAgentState.READY:
                try:
                    # 尝试正常转换到 READY
                    self._transition_state(ActAgentState.READY)
                except RuntimeError:
                    # 如果无法正常转换，强制设置状态（仅在异常恢复时使用）
                    logger.warning("[ImgActAgent] 无法正常转换到 READY，强制设置状态")
                    self._state = ActAgentState.READY
            raise

    async def add_img_data_to_memory(self) -> bool:
        if self._agent_card_ref.robot_dataloader is None:
            logger.info("robot_dataloader is unavailable; skip image memory update")
            return False

        a2d_data: A2DData | None = await self._agent_card_ref.robot_dataloader.get_latest_concatenate_image_base64()

        if a2d_data is not None:
            a2d_data.img_info()

            await self.memory_manager.add_robot_img_message(
                img_frame_id=a2d_data.frame_id,
                img_type=a2d_data.image_type,
                base64_str=a2d_data.concatenated_image_base64,
            )

            if len(self.memory_manager.current_task_node.contexts) > 50:
                self.memory_manager.current_task_node.compress_policy_discard_oldest(drop_n=1)
                logger.warning("memory is too long, drop oldest 2 contexts")
            has_new_image = True
        else:
            logger.warning("could not get data from a2d_sdk")
            has_new_image = False

        return has_new_image

    # ========== 工具调用历史管理 ==========
    def get_tool_call_history(self) -> list[ToolCallRecord]:
        """获取当前任务的工具调用历史"""
        return self.tool_call_history.copy()

    def reset_tool_history(self):
        """重置工具调用历史（用于开始新任务）"""
        self.tool_call_history.clear()
        self._tool_call_step_counter = 0

    # ========== 优雅退出 ==========
    async def shutdown(self) -> None:
        await self._llm_client.shutdown()
        await self._memory_manager.shutdown()
        await self._service_manager.shutdown()
        await asyncio.sleep(1)  # 等待1秒后退出事件循环
        await self._agent_tools.shutdown()
        await asyncio.sleep(1)  # 等待1秒后退出事件循环

    async def terminate(self) -> None:
        await self._llm_client.terminate()
        await self._memory_manager.terminate()
        await self._service_manager.terminate()
        await asyncio.sleep(1)  # 等待1秒后退出事件循环
        await self._agent_tools.terminate()
        await asyncio.sleep(1)  # 等待1秒后退出事件循环
