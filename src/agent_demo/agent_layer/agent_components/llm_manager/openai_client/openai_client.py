from openai._streaming import AsyncStream
import asyncio
import logging
import os
import time
from agent_demo.types.interaction_types import InteractionPackage
from typing import Any, AsyncGenerator, Awaitable, Callable
from ..chat_api_native.base_chat_api import BaseChatAPI, llmState
from agent_demo.types.agent_types import (
    OpenAIChoice,
    OpenAISendMsg,
    OpenAIResponseMsg,
    AssistantMessageType,
    ToolCallParam,
    FunctionSubParam,
    BaseAgentCard,
    TextParam,
    RefusalParam,
)
from openai.types.chat import ChatCompletionChunk, ChatCompletion
from openai import AsyncOpenAI
from openai import APIStatusError
from agent_demo.common.root_logger import table_to_str
import json
import httpx

logger = logging.getLogger(__name__)


class OpenAIClient(BaseChatAPI):
    def __init__(self, agent_card: BaseAgentCard):
        super().__init__(agent_card=agent_card)
        # Some OpenAI-compatible gateways block the default OpenAI Python SDK
        # User-Agent for streaming requests while still allowing raw SSE.
        sdk_user_agent = os.getenv("OPENAI_SDK_USER_AGENT", "Olympus/1.0")
        self._client: AsyncOpenAI = AsyncOpenAI(
            api_key=self._agent_card_ref.config.api_key,
            project=(
                self._agent_card_ref.config.user_group_info.project.code
                if self._agent_card_ref.config.user_group_info and self._agent_card_ref.config.user_group_info.project
                else None
            ),
            organization=(
                self._agent_card_ref.config.user_group_info.organization.code
                if (
                    self._agent_card_ref.config.user_group_info
                    and self._agent_card_ref.config.user_group_info.organization
                )
                else None
            ),
            base_url=self._agent_card_ref.config.base_url,
            timeout=self._agent_card_ref.config.timeout,
            max_retries=self._agent_card_ref.config.max_retries,
            default_headers={"User-Agent": sdk_user_agent},
        )

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @staticmethod
    def _normalize_wire_api(wire_api: str) -> str:
        return wire_api.strip().lower().replace("_", ".").replace("-", ".")

    @property
    def _wire_api(self) -> str:
        return self._normalize_wire_api(getattr(self._agent_card_ref.config, "wire_api", "chat.completions"))

    @property
    def _uses_responses_api(self) -> bool:
        return self._wire_api in {"responses", "response"}

    # ========== 优雅退出 ==========
    async def shutdown(self) -> None:
        if not self._client.is_closed:
            await self._client.close()

    async def terminate(self) -> None:
        await self.shutdown()

    # ========= 重置方法 =========
    async def reset(self):
        self._state = llmState.READY

    async def init_client(self):
        # 如果已经是 READY 状态，不需要再次转换（避免重复初始化）
        if self._state != llmState.READY:
            self._transition_state(llmState.READY)
        logger.info("[Init][OpenAI_Client][Done]")

    async def sync_chat(
        self,
        send_package: OpenAISendMsg,
        on_text_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> OpenAIResponseMsg:
        if os.environ.get("ROBOCLAW_DUMP_PROMPTS"):
            self._dump_send_package(send_package)

        # 确保状态从 READY 开始
        if self._state != llmState.READY:
            logger.warning(f"[OpenAIClient] 状态不是 READY ({self._state})，强制重置为 READY")
            self._state = llmState.READY

        try:
            # 状态转换
            self._transition_state(llmState.STREAM_GENERATING if on_text_delta else llmState.SYNC_GENERATING)

            if on_text_delta is None:
                res_original = await self._chat(send_package)
            else:
                res_original = await self._chat_with_streaming(send_package, on_text_delta)

            # 状态转换
            self._transition_state(llmState.COMPLETED)

            res_analysis = await self._analyze_sync_response(res_original)

            if os.environ.get("ROBOCLAW_DUMP_PROMPTS"):
                self._dump_response(res_analysis)

            # 状态转换
            self._transition_state(llmState.READY)

            return res_analysis
        except Exception as e:
            # 发生异常时，确保状态回到 READY，避免状态卡在中间状态
            logger.error(f"[OpenAIClient] sync_chat 发生异常，重置状态为 READY: {type(e).__name__}: {e}")
            if isinstance(e, APIStatusError):
                logger.error("[OpenAIClient] API status detail: status=%s body=%s", e.status_code, e.body)
            if self._state != llmState.READY:
                try:
                    # 尝试正常转换到 READY（SYNC_GENERATING 允许转换到 READY）
                    self._transition_state(llmState.READY)
                except RuntimeError:
                    # 如果无法正常转换，强制设置状态（仅在异常恢复时使用）
                    logger.warning("[OpenAIClient] 无法正常转换到 READY，强制设置状态")
                    self._state = llmState.READY
            raise

    async def _chat_with_streaming(
        self,
        send_package: OpenAISendMsg,
        on_text_delta: Callable[[str], Awaitable[None]],
    ) -> ChatCompletion | dict[str, Any]:
        if self._uses_responses_api:
            request_args = self._build_responses_request(send_package)
            try:
                return await self._chat_with_responses_api_stream(request_args, on_text_delta)
            except Exception as e:
                logger.warning(
                    "[OpenAIClient] responses.stream failed, trying chat.completions stream fallback: %s: %s",
                    type(e).__name__,
                    e,
                )
                try:
                    return await self._chat_with_chat_completions_stream(send_package, on_text_delta)
                except Exception as chat_stream_e:
                    logger.warning(
                        "[OpenAIClient] chat.completions stream fallback failed, falling back to non-streaming response: %s: %s",
                        type(chat_stream_e).__name__,
                        chat_stream_e,
                    )
                    return await self._chat(send_package)

        try:
            return await self._chat_with_chat_completions_stream(send_package, on_text_delta)
        except Exception as e:
            logger.warning(
                "[OpenAIClient] chat.completions stream failed, falling back to non-streaming response: %s: %s",
                type(e).__name__,
                e,
            )
            return await self._chat(send_package)

    async def _chat(
        self,
        send_package: OpenAISendMsg,
    ) -> ChatCompletion | dict[str, Any]:
        if self._uses_responses_api:
            if self._agent_card_ref.config.choices_n != 1:
                logger.warning(
                    "[OpenAIClient] responses API does not support n=%s; using a single response",
                    self._agent_card_ref.config.choices_n,
                )
            return await self._chat_with_responses_api(self._build_responses_request(send_package))

        response = await self._client.chat.completions.create(
            model=self._agent_card_ref.config.model,
            n=self._agent_card_ref.config.choices_n,
            temperature=self._agent_card_ref.config.temperature,
            reasoning_effort=self._agent_card_ref.config.reasoning_effort,
            max_completion_tokens=self._agent_card_ref.config.max_completion_tokens,
            messages=send_package.contexts,
            tools=send_package.tools_list,
            stream=False,  # sync_chat 用非流式
        )
        return response

    def _build_responses_request(self, send_package: OpenAISendMsg) -> dict[str, Any]:
        instructions_parts: list[str] = []
        response_input: list[dict] = []
        for context in send_package.contexts:
            role = str(context.get("role", "user"))
            if role == "system":
                instruction_text = self._extract_text_content(context.get("content"))
                if instruction_text:
                    instructions_parts.append(instruction_text)
                continue
            response_input.extend(self._convert_chat_message_to_response_items(context))

        request_args: dict[str, Any] = {
            "model": self._agent_card_ref.config.model,
            "input": response_input,
            "stream": False,
            "instructions": "\n\n".join(instructions_parts).strip() or "You are a helpful coding assistant.",
            "reasoning": {"effort": self._agent_card_ref.config.reasoning_effort},
        }
        if send_package.tools_list:
            request_args["tools"] = self._build_responses_tools(send_package.tools_list)
            request_args["parallel_tool_calls"] = True
        return request_args

    async def _chat_with_responses_api(self, request_args: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._agent_card_ref.config.base_url.rstrip('/')}/responses"
        max_attempts = max(1, int(self._agent_card_ref.config.max_retries) + 1)
        response: httpx.Response | None = None

        async with httpx.AsyncClient(
            timeout=self._agent_card_ref.config.timeout,
            headers={
                "Authorization": f"Bearer {self._agent_card_ref.config.api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        ) as client:
            for attempt in range(1, max_attempts + 1):
                try:
                    response = await client.post(url, json=request_args)
                    if response.is_error:
                        logger.error(
                            "[OpenAIClient] responses API status detail: status=%s body=%s",
                            response.status_code,
                            response.text,
                        )
                        if self._should_retry_responses_status(response.status_code) and attempt < max_attempts:
                            backoff_seconds = self._responses_retry_backoff(attempt)
                            logger.warning(
                                "[OpenAIClient] responses API transient error %s on attempt %s/%s; retrying in %.1fs",
                                response.status_code,
                                attempt,
                                max_attempts,
                                backoff_seconds,
                            )
                            await asyncio.sleep(backoff_seconds)
                            continue
                        response.raise_for_status()
                    break
                except httpx.RequestError as exc:
                    if attempt >= max_attempts:
                        raise
                    backoff_seconds = self._responses_retry_backoff(attempt)
                    logger.warning(
                        "[OpenAIClient] responses API request error on attempt %s/%s: %s; retrying in %.1fs",
                        attempt,
                        max_attempts,
                        exc,
                        backoff_seconds,
                    )
                    await asyncio.sleep(backoff_seconds)

        if response is None:
            raise RuntimeError("responses API request did not produce a response")

        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            return self._parse_responses_sse(response.text)

        if not response.text.strip():
            raise ValueError("responses API returned an empty body")

        return response.json()

    async def _chat_with_responses_api_stream(
        self,
        request_args: dict[str, Any],
        on_text_delta: Callable[[str], Awaitable[None]],
    ) -> dict[str, Any]:
        stream_args = dict(request_args)
        stream_args.pop("stream", None)
        final_response: dict[str, Any] | None = None

        async with self._client.responses.stream(**stream_args) as response_stream:
            async for event in response_stream:
                if event.type == "response.output_text.delta" and event.delta:
                    await on_text_delta(event.delta)
                elif event.type == "response.completed":
                    final_response = event.response.to_dict()

            if final_response is None:
                final_response = (await response_stream.get_final_response()).to_dict()

        return final_response

    async def _chat_with_chat_completions_stream(
        self,
        send_package: OpenAISendMsg,
        on_text_delta: Callable[[str], Awaitable[None]],
    ) -> ChatCompletion:
        response_stream: AsyncStream[ChatCompletionChunk] = await self._client.chat.completions.create(
            model=self._agent_card_ref.config.model,
            n=self._agent_card_ref.config.choices_n,
            temperature=self._agent_card_ref.config.temperature,
            reasoning_effort=self._agent_card_ref.config.reasoning_effort,
            max_completion_tokens=self._agent_card_ref.config.max_completion_tokens,
            messages=send_package.contexts,
            tools=send_package.tools_list,
            stream=True,
            stream_options={"include_usage": True},
        )

        response_id = ""
        response_model = self._agent_card_ref.config.model
        response_created = int(time.time())
        response_object = "chat.completion"
        system_fingerprint = ""
        usage_payload: dict[str, Any] | None = None
        choices_acc: dict[int, dict[str, Any]] = {}

        async for chunk in response_stream:
            response_id = chunk.id or response_id
            response_model = chunk.model or response_model
            response_created = chunk.created or response_created
            response_object = chunk.object or response_object
            system_fingerprint = chunk.system_fingerprint or system_fingerprint

            if chunk.usage is not None:
                usage_payload = chunk.usage.to_dict()

            for choice in chunk.choices:
                choice_acc = choices_acc.setdefault(
                    choice.index,
                    {
                        "content_parts": [],
                        "tool_calls": {},
                        "finish_reason": "stop",
                    },
                )

                if choice.finish_reason:
                    choice_acc["finish_reason"] = choice.finish_reason

                delta = choice.delta
                if delta.content:
                    choice_acc["content_parts"].append(delta.content)
                    await on_text_delta(delta.content)

                for tool_call in delta.tool_calls or []:
                    tool_index = int(getattr(tool_call, "index", 0) or 0)
                    tool_acc = choice_acc["tool_calls"].setdefault(
                        tool_index,
                        {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    if tool_call.id:
                        tool_acc["id"] = tool_call.id

                    function_delta = getattr(tool_call, "function", None)
                    if function_delta is not None:
                        if function_delta.name:
                            tool_acc["function"]["name"] += function_delta.name
                        if function_delta.arguments:
                            tool_acc["function"]["arguments"] += function_delta.arguments

        response_payload: dict[str, Any] = {
            "id": response_id,
            "choices": [],
            "created": response_created,
            "model": response_model,
            "object": response_object,
            "system_fingerprint": system_fingerprint or None,
        }
        if usage_payload is not None:
            response_payload["usage"] = usage_payload

        for choice_index in sorted(choices_acc):
            choice_acc = choices_acc[choice_index]
            message_payload: dict[str, Any] = {
                "role": "assistant",
                "content": "".join(choice_acc["content_parts"]) or None,
            }
            tool_calls = [
                choice_acc["tool_calls"][tool_index]
                for tool_index in sorted(choice_acc["tool_calls"])
                if choice_acc["tool_calls"][tool_index]["function"]["name"]
            ]
            if tool_calls:
                message_payload["tool_calls"] = tool_calls

            response_payload["choices"].append(
                {
                    "finish_reason": choice_acc["finish_reason"],
                    "index": choice_index,
                    "message": message_payload,
                }
            )

        return ChatCompletion.model_validate(response_payload)

    @staticmethod
    def _should_retry_responses_status(status_code: int) -> bool:
        return status_code in {408, 409, 429, 500, 502, 503, 504}

    @staticmethod
    def _responses_retry_backoff(attempt: int) -> float:
        return min(2.0 * attempt, 8.0)

    def _parse_responses_sse(self, raw_text: str) -> dict[str, Any]:
        latest_response: dict[str, Any] | None = None
        current_data_lines: list[str] = []

        def flush_event() -> None:
            nonlocal latest_response, current_data_lines
            if not current_data_lines:
                return

            data_str = "\n".join(current_data_lines).strip()
            current_data_lines = []
            if not data_str or data_str == "[DONE]":
                return

            payload = json.loads(data_str)
            response_data = payload.get("response")
            if isinstance(response_data, dict):
                latest_response = response_data

        for raw_line in raw_text.splitlines():
            line = raw_line.rstrip("\r")
            if not line:
                flush_event()
                continue
            if line.startswith("data:"):
                current_data_lines.append(line[5:].lstrip())

        flush_event()
        if latest_response is None:
            raise ValueError(f"responses API returned SSE without a final response payload: {raw_text[:500]}")
        return latest_response

    def _build_responses_tools(self, tools_list: list[dict]) -> list[dict]:
        response_tools: list[dict] = []
        for tool in tools_list:
            if not isinstance(tool, dict):
                continue
            if tool.get("type") != "function":
                logger.warning("[OpenAIClient] Skip unsupported responses tool type: %s", tool.get("type"))
                continue

            function_def = tool.get("function", tool)
            name = function_def.get("name")
            parameters = function_def.get("parameters", {})
            if not name:
                logger.warning("[OpenAIClient] Skip malformed function tool without name: %s", tool)
                continue

            response_tools.append(
                {
                    "type": "function",
                    "name": name,
                    "description": function_def.get("description"),
                    "parameters": parameters,
                    "strict": bool(function_def.get("strict", False)),
                }
            )
        return response_tools

    def _build_responses_input(self, contexts: list[dict]) -> list[dict]:
        response_input: list[dict] = []
        for context in contexts:
            response_input.extend(self._convert_chat_message_to_response_items(context))
        return response_input

    def _extract_text_content(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            content_type = content.get("type")
            if content_type == "text":
                return str(content.get("text", ""))
            if content_type == "refusal":
                return str(content.get("refusal", ""))
            return json.dumps(content, ensure_ascii=False)
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type")
                    if item_type == "text":
                        parts.append(str(item.get("text", "")))
                    elif item_type == "image_url":
                        parts.append("[image]")
                    else:
                        parts.append(json.dumps(item, ensure_ascii=False))
                else:
                    parts.append(str(item))
            return "\n".join(part for part in parts if part)
        return str(content)

    def _convert_chat_message_to_response_items(self, context: dict) -> list[dict]:
        role = str(context.get("role", "user"))
        if role == "tool":
            tool_item = self._convert_tool_message_to_response_item(context)
            return [tool_item] if tool_item else []

        if role not in {"user", "assistant", "system", "developer"}:
            logger.warning("[OpenAIClient] Unknown message role %s; fallback to user", role)
            role = "user"

        response_items: list[dict] = []
        message_item = self._convert_standard_message_to_response_item(role=role, context=context)
        if message_item is not None:
            response_items.append(message_item)

        if role == "assistant":
            for tool_call in context.get("tool_calls") or []:
                tool_call_item = self._convert_tool_call_to_response_item(tool_call)
                if tool_call_item is not None:
                    response_items.append(tool_call_item)

        return response_items

    def _convert_standard_message_to_response_item(self, role: str, context: dict) -> dict | None:
        content = context.get("content")
        content_items = self._convert_message_content_to_response_content(role=role, content=content)
        if not content_items:
            return None
        return {
            "type": "message",
            "role": role,
            "content": content_items,
        }

    def _convert_message_content_to_response_content(self, role: str, content: Any) -> list[dict]:
        text_item_type = "output_text" if role == "assistant" else "input_text"
        refusal_item_type = "refusal" if role == "assistant" else "input_text"

        if content is None:
            return []

        if isinstance(content, str):
            return [{"type": text_item_type, "text": content}]

        if isinstance(content, dict):
            content_type = content.get("type")
            if content_type == "text":
                return [{"type": text_item_type, "text": str(content.get("text", ""))}]
            if content_type == "refusal":
                if refusal_item_type == "refusal":
                    return [{"type": "refusal", "refusal": str(content.get("refusal", ""))}]
                return [{"type": "input_text", "text": str(content.get("refusal", ""))}]
            if content_type == "image_url":
                image_url = content.get("image_url", {})
                return [
                    {
                        "type": "input_image",
                        "image_url": image_url.get("url"),
                        "detail": image_url.get("detail", "auto"),
                    }
                ]
            return [{"type": text_item_type, "text": json.dumps(content, ensure_ascii=False)}]

        if isinstance(content, list):
            content_items: list[dict] = []
            for item in content:
                if not isinstance(item, dict):
                    content_items.append({"type": text_item_type, "text": str(item)})
                    continue

                item_type = item.get("type")
                if item_type == "text":
                    content_items.append({"type": text_item_type, "text": str(item.get("text", ""))})
                elif item_type == "refusal":
                    if refusal_item_type == "refusal":
                        content_items.append({"type": "refusal", "refusal": str(item.get("refusal", ""))})
                    else:
                        content_items.append({"type": "input_text", "text": str(item.get("refusal", ""))})
                elif item_type == "image_url":
                    image_url = item.get("image_url", {})
                    content_items.append(
                        {
                            "type": "input_image",
                            "image_url": image_url.get("url"),
                            "detail": image_url.get("detail", "auto"),
                        }
                    )
                else:
                    content_items.append({"type": text_item_type, "text": json.dumps(item, ensure_ascii=False)})
            return content_items

        return [{"type": text_item_type, "text": str(content)}]

    def _convert_tool_call_to_response_item(self, tool_call: dict) -> dict | None:
        if not isinstance(tool_call, dict):
            return None

        function_def = tool_call.get("function", {})
        call_id = str(tool_call.get("id", "")).strip()
        name = str(function_def.get("name", "")).strip()
        if not call_id or not name:
            logger.warning("[OpenAIClient] Skip malformed tool call in context: %s", tool_call)
            return None

        return {
            "type": "function_call",
            "call_id": call_id,
            "name": name,
            "arguments": str(function_def.get("arguments", "{}")),
        }

    def _convert_tool_message_to_response_item(self, context: dict) -> dict | None:
        call_id = str(context.get("tool_call_id", "")).strip()
        if not call_id:
            logger.warning("[OpenAIClient] Skip malformed tool response without tool_call_id: %s", context)
            return None

        return {
            "type": "function_call_output",
            "call_id": call_id,
            "output": self._encode_tool_output(context.get("content")),
        }

    def _encode_tool_output(self, content: Any) -> str:
        if isinstance(content, dict) and content.get("type") == "text":
            content = content.get("text", "")

        if isinstance(content, str):
            try:
                return json.dumps(json.loads(content), ensure_ascii=False)
            except json.JSONDecodeError:
                return json.dumps(content, ensure_ascii=False)

        return json.dumps(content, ensure_ascii=False)

    def _extract_usage_stats(self, response: ChatCompletion | dict[str, Any]) -> tuple[int, int, int, float | None]:
        if isinstance(response, ChatCompletion):
            usage = response.usage
            if usage is None:
                return 0, 0, 0, None

            cache_hit_rate: float | None = None
            if usage.prompt_tokens_details and usage.prompt_tokens_details.cached_tokens and usage.prompt_tokens:
                cache_hit_rate = round((usage.prompt_tokens_details.cached_tokens / usage.prompt_tokens) * 100, 2)
            return usage.prompt_tokens, usage.completion_tokens, usage.total_tokens, cache_hit_rate

        usage = response.get("usage")
        if not isinstance(usage, dict):
            return 0, 0, 0, None

        cache_hit_rate = None
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", 0) or 0)
        input_details = usage.get("input_tokens_details", {})
        if isinstance(input_details, dict):
            cached_tokens = int(input_details.get("cached_tokens", 0) or 0)
            if input_tokens and cached_tokens:
                cache_hit_rate = round((cached_tokens / input_tokens) * 100, 2)
        return input_tokens, output_tokens, total_tokens, cache_hit_rate

    async def _analyze_sync_response(self, response: ChatCompletion | dict[str, Any]) -> OpenAIResponseMsg:
        if isinstance(response, ChatCompletion):
            return await self._analyze_chat_completion_response(response)
        return await self._analyze_responses_response(response)

    async def _analyze_chat_completion_response(self, response: ChatCompletion) -> OpenAIResponseMsg:
        # 1️⃣ 解析主信息
        response_msg: OpenAIResponseMsg = OpenAIResponseMsg(
            id=response.id,
            model=response.model,
            created=response.created,
            object=response.object,
            system_fingerprint=response.system_fingerprint or "",
        )

        # 2️⃣ 解析用量信息（如果有）
        prompt_tokens, completion_tokens, total_tokens, cache_hit_rate = self._extract_usage_stats(response)
        response_msg.prompt_tokens = prompt_tokens
        response_msg.completion_tokens = completion_tokens
        response_msg.total_tokens = total_tokens
        response_msg.cache_hit_rate = cache_hit_rate
        response_msg.need_compress = response_msg.total_tokens > self._agent_card_ref.config.compression_threshold
        self._agent_card_ref.all_token_usage = total_tokens
        self._agent_card_ref.prompt_token_usage = prompt_tokens

        # 3️⃣ 解析每个 choice 的信息
        for idx, choice in enumerate(response.choices):
            msg = AssistantMessageType()
            if choice.message.refusal:
                msg.refusal = RefusalParam(refusal=choice.message.refusal)
            elif choice.message.content:
                msg.content = TextParam(text=choice.message.content)

            # 4️⃣ 解析 Tool calls（如有）
            msg.tool_calls = [
                ToolCallParam(
                    id=tool_call.id,
                    function=FunctionSubParam(
                        name=tool_call.function.name,
                        arguments=tool_call.function.arguments,
                    ),
                )
                for tool_call in choice.message.tool_calls or []
            ]

            response_msg.choices.append(
                OpenAIChoice(
                    finish_reason=choice.finish_reason,
                    index=choice.index,
                    message=msg,
                    has_tool_call=bool(msg.tool_calls),  # 判断是否有工具调用
                )
            )

        if not self._agent_card_ref.silence:
            await self.show_response_msg_as_table(response_msg)

        return response_msg

    async def _analyze_responses_response(self, response: dict[str, Any]) -> OpenAIResponseMsg:
        response_msg = OpenAIResponseMsg(
            id=str(response.get("id", "")),
            model=str(response.get("model", "")),
            created=int(response.get("created_at", 0) or 0),
            object=str(response.get("object", "response")),
            system_fingerprint="",
        )

        prompt_tokens, completion_tokens, total_tokens, cache_hit_rate = self._extract_usage_stats(response)
        response_msg.prompt_tokens = prompt_tokens
        response_msg.completion_tokens = completion_tokens
        response_msg.total_tokens = total_tokens
        response_msg.cache_hit_rate = cache_hit_rate
        response_msg.need_compress = response_msg.total_tokens > self._agent_card_ref.config.compression_threshold
        self._agent_card_ref.all_token_usage = total_tokens
        self._agent_card_ref.prompt_token_usage = prompt_tokens

        message_text_parts: list[str] = []
        refusal_parts: list[str] = []
        tool_calls: list[ToolCallParam] = []

        for output_item in response.get("output", []) or []:
            if not isinstance(output_item, dict):
                continue
            output_type = output_item.get("type")
            if output_type == "message":
                for content_item in output_item.get("content", []) or []:
                    if not isinstance(content_item, dict):
                        continue
                    if content_item.get("type") == "output_text":
                        message_text_parts.append(str(content_item.get("text", "")))
                    elif content_item.get("type") == "refusal":
                        refusal_parts.append(str(content_item.get("refusal", "")))
            elif output_type == "function_call":
                tool_calls.append(
                    ToolCallParam(
                        id=str(output_item.get("call_id", "")),
                        function=FunctionSubParam(
                            name=str(output_item.get("name", "")),
                            arguments=str(output_item.get("arguments", "{}")),
                        ),
                    )
                )

        assistant_message = AssistantMessageType(tool_calls=tool_calls)
        message_text = "".join(message_text_parts).strip()
        refusal_text = "\n".join(refusal_parts).strip()
        if message_text:
            assistant_message.content = TextParam(text=message_text)
        elif refusal_text:
            assistant_message.refusal = RefusalParam(refusal=refusal_text)

        finish_reason = "tool_calls" if tool_calls else "stop"
        response_status = response.get("status")
        if response_status == "incomplete":
            incomplete_details = response.get("incomplete_details", {})
            if isinstance(incomplete_details, dict):
                finish_reason = str(incomplete_details.get("reason") or "incomplete")
            else:
                finish_reason = "incomplete"
        elif response_status == "failed":
            finish_reason = "error"

        response_msg.choices.append(
            OpenAIChoice(
                finish_reason=finish_reason,
                index=0,
                message=assistant_message,
                has_tool_call=bool(tool_calls),
            )
        )

        if not self._agent_card_ref.silence:
            await self.show_response_msg_as_table(response_msg)

        return response_msg

    async def show_response_msg_as_table(self, res: OpenAIResponseMsg) -> None:
        for table in res.rich_table():
            self._agent_card_ref.display_deque.append(
                InteractionPackage(
                    content_type="analyze_response",
                    agent_id=self._agent_card_ref.agent_id,
                    content=table,
                )
            )
            logger.info(table_to_str(table))

    async def _stream_chat(
        self,
        send_package: OpenAISendMsg,
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """
        异步流式对话：使用传入的 context 和 messages 调用 OpenAI。
        参数必须是 List[Dict[str, str]] 格式，如 {'role': 'user', 'content': 'Hello'}
        """
        if self._uses_responses_api:
            raise NotImplementedError("Streaming with responses API is not implemented in this project yet.")

        try:
            response_stream: AsyncStream[ChatCompletionChunk] = await self._client.chat.completions.create(
                model=self._agent_card_ref.config.model,
                n=self._agent_card_ref.config.choices_n,
                temperature=self._agent_card_ref.config.temperature,
                max_completion_tokens=self._agent_card_ref.config.max_completion_tokens,
                messages=send_package.contexts,
                tools=send_package.tools_list,
                stream=True,  # sync_chat 用非流式
            )
            async for chunk in response_stream:
                yield chunk

        except Exception as e:
            logger.error(f"Unexpected error in chat(): {e}", exc_info=True)
            raise

    _dump_counter: int = 0

    def _dump_send_package(self, send_package: OpenAISendMsg) -> None:
        OpenAIClient._dump_counter += 1
        dump_dir = os.path.join("applog", "prompt_dumps")
        os.makedirs(dump_dir, exist_ok=True)
        save_images = os.environ.get("ROBOCLAW_DUMP_IMAGES", "").strip() in ("1", "true", "yes")
        path = os.path.join(dump_dir, f"req_{OpenAIClient._dump_counter:04d}.json")
        try:
            sanitized = []
            img_count = 0
            for msg in send_package.contexts:
                entry = dict(msg)
                content = entry.get("content")
                if isinstance(content, list):
                    stripped = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "image_url":
                            img_count += 1
                            if save_images:
                                img_url = part.get("image_url", {})
                                url_str = img_url.get("url", "") if isinstance(img_url, dict) else str(img_url)
                                img_path = os.path.join(dump_dir, f"req_{OpenAIClient._dump_counter:04d}_img{img_count}.jpg")
                                self._save_base64_image(url_str, img_path)
                                stripped.append({"type": "image_url", "image_file": img_path})
                            else:
                                stripped.append({"type": "image_url", "image_url": "[base64 omitted]"})
                        else:
                            stripped.append(part)
                    entry["content"] = stripped
                sanitized.append(entry)
            payload = {
                "request_index": OpenAIClient._dump_counter,
                "model": self._agent_card_ref.config.model,
                "messages": sanitized,
                "tools_count": len(send_package.tools_list) if send_package.tools_list else 0,
                "images_count": img_count,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            logger.info("[PromptDump] Saved to %s (%d messages, %d images)", path, len(sanitized), img_count)
        except Exception as exc:
            logger.warning("[PromptDump] Failed: %s", exc)

    def _dump_response(self, response_msg: "OpenAIResponseMsg") -> None:
        dump_dir = os.path.join("applog", "prompt_dumps")
        os.makedirs(dump_dir, exist_ok=True)
        path = os.path.join(dump_dir, f"req_{OpenAIClient._dump_counter:04d}_response.json")
        try:
            resp_text = ""
            tool_calls_data = []
            if response_msg.choices:
                choice = response_msg.choices[0]
                msg = choice.message
                if msg.content is not None:
                    resp_text = msg.content.text if hasattr(msg.content, "text") else str(msg.content)
                elif msg.refusal is not None:
                    resp_text = msg.refusal.refusal if hasattr(msg.refusal, "refusal") else str(msg.refusal)
                for tc in msg.tool_calls:
                    tc_dict = tc.model_dump() if hasattr(tc, "model_dump") else tc.dict() if hasattr(tc, "dict") else str(tc)
                    tool_calls_data.append(tc_dict)

            payload = {
                "request_index": OpenAIClient._dump_counter,
                "model": response_msg.model,
                "response_text": resp_text,
                "tool_calls": tool_calls_data,
                "usage": {
                    "prompt_tokens": response_msg.prompt_tokens,
                    "completion_tokens": response_msg.completion_tokens,
                    "total_tokens": response_msg.total_tokens,
                },
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            logger.info("[PromptDump] Response saved to %s", path)
        except Exception as exc:
            logger.warning("[PromptDump] Response dump failed: %s", exc)

        self._dump_readable_md()

    def _dump_readable_md(self) -> None:
        """Generate a human-readable markdown log for the current request+response pair."""
        dump_dir = os.path.join("applog", "prompt_dumps")
        req_path = os.path.join(dump_dir, f"req_{OpenAIClient._dump_counter:04d}.json")
        resp_path = os.path.join(dump_dir, f"req_{OpenAIClient._dump_counter:04d}_response.json")
        md_path = os.path.join(dump_dir, f"req_{OpenAIClient._dump_counter:04d}_readable.md")
        try:
            req_data, resp_data = {}, {}
            if os.path.exists(req_path):
                with open(req_path, "r", encoding="utf-8") as f:
                    req_data = json.load(f)
            if os.path.exists(resp_path):
                with open(resp_path, "r", encoding="utf-8") as f:
                    resp_data = json.load(f)

            lines: list[str] = []
            lines.append(f"# Request #{req_data.get('request_index', '?')}")
            lines.append(f"**Model:** {req_data.get('model', '?')}  ")
            lines.append(f"**Tools:** {req_data.get('tools_count', 0)}  ")
            lines.append(f"**Images:** {req_data.get('images_count', 0)}")
            lines.append("")

            for msg in req_data.get("messages", []):
                role = msg.get("role", "?")
                name = msg.get("name", "")
                header = f"## [{role}]" + (f" ({name})" if name else "")
                lines.append(header)
                lines.append("")
                content = msg.get("content", "")
                if isinstance(content, str):
                    lines.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict):
                            if part.get("type") == "image_url":
                                img_file = part.get("image_file", "")
                                if img_file:
                                    lines.append(f"![image]({img_file})")
                                else:
                                    lines.append("*[image omitted]*")
                            elif part.get("type") == "text":
                                lines.append(str(part.get("text", "")))
                            else:
                                lines.append(str(part))
                        else:
                            lines.append(str(part))
                lines.append("")
                lines.append("---")
                lines.append("")

            lines.append("## VLM Response")
            lines.append("")
            resp_text = resp_data.get("response_text", "")
            if resp_text:
                try:
                    parsed = json.loads(resp_text)
                    lines.append("```json")
                    lines.append(json.dumps(parsed, ensure_ascii=False, indent=2))
                    lines.append("```")
                except (json.JSONDecodeError, TypeError):
                    lines.append(resp_text)
            tool_calls = resp_data.get("tool_calls", [])
            if tool_calls:
                lines.append("")
                lines.append("### Tool Calls")
                lines.append("")
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        fn = tc.get("function", tc)
                        name = fn.get("name", tc.get("name", "?"))
                        args = fn.get("arguments", tc.get("arguments", ""))
                        lines.append(f"- **{name}**: `{args}`")
                    else:
                        lines.append(f"- {tc}")

            usage = resp_data.get("usage", {})
            if usage:
                lines.append("")
                lines.append(f"**Tokens:** prompt={usage.get('prompt_tokens', '?')} "
                             f"completion={usage.get('completion_tokens', '?')} "
                             f"total={usage.get('total_tokens', '?')}")

            with open(md_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            logger.info("[PromptDump] Readable log saved to %s", md_path)
        except Exception as exc:
            logger.warning("[PromptDump] Readable dump failed: %s", exc)

    @staticmethod
    def _save_base64_image(data_url: str, path: str) -> None:
        try:
            import base64 as b64mod
            if "base64," in data_url:
                b64_data = data_url.split("base64,", 1)[1]
            else:
                b64_data = data_url
            raw = b64mod.b64decode(b64_data)
            with open(path, "wb") as f:
                f.write(raw)
        except Exception as exc:
            logger.warning("[PromptDump] Image save failed: %s", exc)
