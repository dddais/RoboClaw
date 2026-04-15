from asyncio.events import AbstractEventLoop
import logging
import inspect
import asyncio
import json
from pathlib import Path
from typing import get_type_hints, Optional, Any
from agent_demo.types.agent_types import (
    ServiceRegister,
    ORMCPTool,
    FunctionDef,
    TextParam,
    AgentToolsFunc,
    BaseAgentCard,
)
from ..ormcp_service_manager import ORMCPServiceManager
from ..memory_manager import MemoryManager
from agent_demo.machine_layer.base_dataloader import BaseDataloader
from .local_skill_registry import LocalSkillRegistry, LocalSkill
from agent_demo.common.project_env import PROJECT_ROOT

logger = logging.getLogger(__name__)


def extract_param_docs(func) -> dict:
    """从 docstring 中提取参数描述"""
    doc = inspect.getdoc(func)
    if not doc:
        raise ValueError(f"函数 `{func.__name__}` 缺少 docstring 注释，无法提取参数说明。")

    param_docs = {}
    lines = doc.splitlines()
    for line in lines:
        line = line.strip()
        if ":" in line:
            name, desc = line.split(":", 1)
            param_docs[name.strip()] = desc.strip()
    return param_docs


def build_tool_from_func(func: AgentToolsFunc, service_name: str) -> ORMCPTool:
    name = func.__name__
    doc = inspect.getdoc(func) or "No description."
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)
    param_docs = extract_param_docs(func)
    required = []
    properties = {}

    for param_name, param in sig.parameters.items():
        if param.default is inspect.Parameter.empty:
            required.append(param_name)
        hint = type_hints.get(param_name, str)
        param_type = "string"
        if hint is bool:
            param_type = "boolean"
        elif hint is int:
            param_type = "integer"
        elif hint is float:
            param_type = "number"
        properties[param_name] = {"type": param_type, "description": param_docs.get(param_name, f"参数 {param_name}")}

    func_def = FunctionDef(
        name=name, description=doc, parameters={"type": "object", "required": required, "properties": properties}
    )
    return ORMCPTool(service_name=service_name, func_definition=func_def)


class AgentTools:
    def __init__(
        self,
        memory_manager: MemoryManager,
        service_manager: ORMCPServiceManager,
        agent_card: BaseAgentCard,
        agent_instance: Optional[Any] = None,  # 用于访问 agent 实例和 LLM 客户端
    ):
        self._agent_card_ref: BaseAgentCard = agent_card
        self._memory_manager_ref = memory_manager
        self._service_manager_ref = service_manager
        self._agent_instance = agent_instance  # 存储 agent 实例引用
        self._service_name = "AgentTools"
        self._description = "本服务为常驻服务，本服务提供了一系列用于辅助你管理机器人的工具"
        self._agent_services_register: ServiceRegister = ServiceRegister(
            service_name=self._service_name,
            description=self._description,
            is_activation=True,
            is_agent_service=True,
        )
        self._skill_service_name = "SkillTools"
        self._skill_registry = LocalSkillRegistry(configured_paths=self._agent_card_ref.skill_paths)
        self._skill_services_register: ServiceRegister = ServiceRegister(
            service_name=self._skill_service_name,
            description=self._skill_registry.build_service_description(),
            is_activation=True,
            is_agent_service=True,
        )
        self._tool_registry: dict[str, dict[str, AgentToolsFunc]] = {}
        self._pending_context_injections: list[dict[str, Any]] = []

    async def init_agent_tools(self) -> None:
        await self.register_activate_service_tool()
        await self.register_skill_service_tool()
        await self.register_agent_service()
        await self.update_agent_tools_prompt()

    async def register_activate_service_tool(self):
        """注册管理机器人的工具"""
        tool_methods: list[AgentToolsFunc] = [
            self.activate_service,
            self.navigate_to_pose,
            self.fetch_env,
        ]
        for method in tool_methods:
            tool = build_tool_from_func(method, service_name=self._service_name)
            self._agent_services_register.tools_list.append(tool)
            await self.register_tool(service_name=tool.service_name, tool_name=tool.func_definition.name, func=method)

    async def register_agent_service(self) -> None:
        await self._service_manager_ref.registry_service(self._agent_services_register)
        await self._service_manager_ref.registry_service(self._skill_services_register)

    async def register_skill_service_tool(self):
        self._skill_services_register.description = self._skill_registry.build_service_description()
        tool_methods: list[AgentToolsFunc] = [
            self.list_skills,
            self.get_skill_details,
        ]
        for method in tool_methods:
            tool = build_tool_from_func(method, service_name=self._skill_service_name)
            self._skill_services_register.tools_list.append(tool)
            await self.register_tool(service_name=tool.service_name, tool_name=tool.func_definition.name, func=method)

    async def activate_service(self, service_name: str, is_activation: bool) -> str:
        """
        激活或关闭某个服务。
        :param service_name: 服务名称,必须从已知的服务从中选择，大小写敏感
        :param is_activation: True为激活,False为不激活
        """
        if service_name == self._service_name:
            return f"{service_name}为常驻服务，你不能关闭这个服务"

        for service in self._service_manager_ref._services_register_list:
            if service.service_name == service_name:
                service.is_activation = is_activation
                logger.info(f"[activate_service] {service.service_name} -> {service.is_activation}")
        await self.update_agent_tools_prompt()

        return f"设置成功，{service_name}的状态已经更改为{str(is_activation)}"

    async def navigate_to_pose(self, x: int, y: int, theta: int) -> str:
        """
        导航至当前地图里的指定坐标，只需要调用一次，后续会自动导航。（注意：如果机器人已经在指定坐标附近，会自动停止）
        :param x: 当前地图里的x坐标
        :param y: 当前地图里的y坐标
        :param theta: 停止时面朝的角度
        """
        if isinstance(self._agent_card_ref.robot_dataloader, BaseDataloader):
            self._agent_card_ref.robot_dataloader.slam.navigate_to_pose(x, y, theta)
            return "调用成功，开始导航。"
        else:
            return "当前环境下没有SLAM模块，无法调用此工具。"

    async def fetch_env(self) -> str:
        """
        获取当前环境的最新相机拼接图，并将该图像加入当前会话上下文，供后续推理直接参考。
        """
        if self._agent_card_ref.robot_dataloader is None:
            return json.dumps({"error": "机器人数据加载器未初始化"}, ensure_ascii=False, indent=2)

        a2d_data = await self._agent_card_ref.robot_dataloader.get_latest_concatenate_image_base64()
        if a2d_data is None or not a2d_data.concatenated_image_base64:
            return json.dumps({"error": "无法获取最新环境图像"}, ensure_ascii=False, indent=2)

        self._pending_context_injections.append(
            {
                "kind": "robot_image",
                "img_frame_id": a2d_data.frame_id,
                "img_type": a2d_data.image_type,
                "base64_str": a2d_data.concatenated_image_base64,
            }
        )

        return json.dumps(
            {
                "status": "ok",
                "frame_id": a2d_data.frame_id,
                "image_type": a2d_data.image_type,
                "image_ts": a2d_data.image_ts,
                "message": "Latest environment image has been attached to the current conversation context.",
            },
            ensure_ascii=False,
            indent=2,
        )

    async def flush_pending_context_injections(self) -> None:
        while self._pending_context_injections:
            injection = self._pending_context_injections.pop(0)
            if injection.get("kind") != "robot_image":
                logger.warning("[AgentTools] Skip unknown pending context injection: %s", injection)
                continue

            await self._memory_manager_ref.add_robot_img_message(
                img_frame_id=int(injection["img_frame_id"]),
                img_type=str(injection["img_type"]),
                base64_str=str(injection["base64_str"]),
            )

            if len(self._memory_manager_ref.current_task_node.contexts) > 50:
                self._memory_manager_ref.current_task_node.compress_policy_discard_oldest(drop_n=1)
                logger.warning("[fetch_env] memory is too long, drop oldest 1 context")

    def _resolve_run_artifact_path(self, path_text: str) -> Path:
        raw_path = Path(path_text).expanduser()
        if raw_path.is_absolute():
            return raw_path.resolve()
        return (PROJECT_ROOT / raw_path).resolve()

    async def ensure_run_artifacts(self, run_dir: str) -> str:
        """
        为 workflow skill 创建标准运行目录。
        :param run_dir: 运行输出目录，支持相对路径（相对仓库根目录）或绝对路径
        """
        base_dir = self._resolve_run_artifact_path(run_dir)
        logs_dir = base_dir / "logs"
        dataset_dir = base_dir / "dataset"
        logs_dir.mkdir(parents=True, exist_ok=True)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        return json.dumps(
            {
                "run_dir": str(base_dir),
                "logs_dir": str(logs_dir),
                "dataset_dir": str(dataset_dir),
                "created": True,
            },
            ensure_ascii=False,
            indent=2,
        )

    async def append_jsonl_record(self, file_path: str, record_json: str) -> str:
        """
        追加一条 JSONL 记录，供 workflow skill 写入轮次日志、状态日志或数据集条目。
        :param file_path: JSONL 文件路径，支持相对路径（相对仓库根目录）或绝对路径
        :param record_json: 要写入的一条 JSON 记录，必须是合法 JSON 字符串
        """
        resolved_path = self._resolve_run_artifact_path(file_path)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)

        record = json.loads(record_json)
        serialized = json.dumps(record, ensure_ascii=False)
        with resolved_path.open("a", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.write("\n")

        return json.dumps(
            {
                "file_path": str(resolved_path),
                "appended": True,
                "record_preview": serialized[:200],
            },
            ensure_ascii=False,
            indent=2,
        )

    async def list_skills(self) -> str:
        """
        列出当前可用的本地 skill。
        """
        skills = [skill.summary_dict() for skill in self._skill_registry.list_skills(refresh=True)]
        self._skill_services_register.description = self._skill_registry.build_service_description()
        await self.update_agent_tools_prompt()
        return json.dumps({"skills": skills, "count": len(skills)}, ensure_ascii=False, indent=2)

    async def get_skill_details(self, skill_name: str) -> str:
        """
        查看某个 skill 的完整说明与资源清单。
        :param skill_name: skill 的 frontmatter name，大小写敏感
        """
        skill = self._skill_registry.get_skill(skill_name, refresh=True)
        if skill is None:
            return json.dumps(
                {
                    "error": f"未找到 skill: {skill_name}",
                    "available_skills": [item.name for item in self._skill_registry.list_skills(refresh=False)],
                },
                ensure_ascii=False,
                indent=2,
            )
        return json.dumps(skill.detail_dict(), ensure_ascii=False, indent=2)

    async def run_skill(self, skill_name: str, user_request: str) -> str:
        """
        已弃用。请通过主对话入口使用 `$skill-name` 触发本地 skill，使其走与 TUI/Gradio/Feishu 一致的工具调用路径。
        :param skill_name: skill 的 frontmatter name，大小写敏感
        :param user_request: 需要 skill 处理的具体请求
        """
        skill = self._skill_registry.get_skill(skill_name, refresh=True)
        if skill is None:
            return json.dumps(
                {
                    "error": f"未找到 skill: {skill_name}",
                    "available_skills": [item.name for item in self._skill_registry.list_skills(refresh=False)],
                },
                ensure_ascii=False,
                indent=2,
            )

        return json.dumps(
            {
                "error": "run_skill 已弃用，请改为在主对话入口中使用 `$skill-name` 触发本地 skill。",
                "skill_name": skill.name,
                "user_request": user_request,
                "recommended_invocation": f"${skill.name} {user_request}".strip(),
                "skill_dir": str(skill.skill_dir),
            },
            ensure_ascii=False,
            indent=2,
        )

    def build_structured_skill_delegation_message(
        self,
        skill_name: str,
        skill_args: dict[str, object],
        source_skill: str | None = None,
    ) -> str | None:
        skill = self._skill_registry.get_skill(skill_name, refresh=True)
        if skill is None:
            logger.warning("[AgentTools] Cannot delegate to missing skill: %s", skill_name)
            return None

        request_lines = [
            f"${skill.name} Execute this delegated skill step.",
        ]
        if source_skill:
            request_lines.append(f"Delegated by workflow skill `{source_skill}`.")
        request_lines.extend(
            [
                "Use the following structured inputs unless safety or the live world state requires adaptation.",
                "Structured inputs (JSON):",
                json.dumps(skill_args, ensure_ascii=False, indent=2),
            ]
        )

        expanded = self._skill_registry.expand_inline_request(
            "\n".join(request_lines),
            refresh=False,
            execution_context=self._build_skill_execution_context(),
        )
        return expanded.message

    def _build_skill_execution_prompt(self, skill: LocalSkill) -> str:
        sections: list[str] = [
            "You are executing a local Olympus skill.",
            "Follow the skill instructions closely, but do not claim access to tools or files that are not explicitly present.",
            f"Skill name: {skill.name}",
            f"Skill description: {skill.description}",
        ]
        if skill.default_prompt:
            sections.append(f"Skill default prompt:\n{skill.default_prompt}")
        if skill.scripts or skill.references or skill.assets:
            sections.append(
                "Bundled resources:\n"
                f"- scripts: {skill.scripts}\n"
                f"- references: {skill.references}\n"
                f"- assets: {skill.assets}"
            )
        sections.append(f"Skill instructions:\n{skill.body}")
        return "\n\n".join(sections)

    def _build_skill_execution_context(self) -> str:
        sections = [
            "This skill executes inside the main Olympus agent instance and shares that agent's MCP tool access.",
            "Any tool from an active service below can be called directly during the skill.",
            "If a required service is inactive, first call `AgentTools___activate_service`, then call that service's tools.",
            "",
            "Service snapshot:",
        ]
        services = getattr(self._service_manager_ref, "_services_register_list", None)
        if not services:
            sections.append(
                "- No live service registry snapshot is available yet. Fall back to the Server Registry Block in system context."
            )
            return "\n".join(sections)

        for service in services:
            status = "active" if service.is_activation else "inactive"
            tools = (
                ", ".join(f"`{service.service_name}___{tool.tool_name}`" for tool in service.tools_list)
                or "(no tools discovered)"
            )
            sections.append(f"- `{service.service_name}` [{status}]: {tools}")
        return "\n".join(sections)

    async def register_tool(self, service_name: str, tool_name: str, func: AgentToolsFunc) -> None:
        if service_name not in self._tool_registry:
            self._tool_registry[service_name] = {}
        self._tool_registry[service_name][tool_name] = func

    def get_tool(self, service_name: str, tool_name: str) -> Optional[AgentToolsFunc]:
        """根据 service_name 和 tool_name 获取已注册的工具函数"""
        return self._tool_registry.get(service_name, {}).get(tool_name, None)

    async def tools_routing(self, service_name: str, tool_name: str, tool_args: dict) -> TextParam:
        tool_func: AgentToolsFunc | None = self.get_tool(service_name, tool_name)
        if tool_func is None:
            raise ValueError(f"未找到对应的工具函数: service={service_name}, tool={tool_name}")

        # AgentToolsFunc: TypeAlias = Callable[..., str|TextParam] | Callable[..., Awaitable[str|TextParam]]
        # 支持异步与同步函数的调用
        if inspect.iscoroutinefunction(tool_func):
            result = await tool_func(**tool_args)
        else:
            loop: AbstractEventLoop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, lambda: tool_func(**tool_args))

        # 确保返回 TextParam 类型
        if not isinstance(result, TextParam):
            result = TextParam(text=str(result))

        return result

    async def update_agent_tools_prompt(self):
        await self._memory_manager_ref.update_service_registry(self._service_manager_ref._services_register_list)

    # ========== 优雅退出 ==========
    async def shutdown(self) -> None:
        return

    async def terminate(self) -> None:
        await self.shutdown()
