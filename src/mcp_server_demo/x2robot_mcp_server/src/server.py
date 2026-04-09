"""
x2robot MCP Server
===================
MCP server that bridges the RoboClaw Agent to the x2robot Bridge HTTP API.
Communicates with the Agent via STDIO MCP protocol, and with the robot via
HTTP REST calls to x2robot_bridge_server.

Environment variables:
    X2ROBOT_BRIDGE_URL  - Base URL of the bridge server (default: http://localhost:8766)
"""

import os
import sys
import anyio
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
import httpx
from httpx._models import Response
import logging

logger = logging.getLogger(name=__name__)

app = Server("x2robot_mcp_server")

bridge_base_url = os.environ.get("X2ROBOT_BRIDGE_URL", "")
if not bridge_base_url:
    bridge_base_url = "http://localhost:8766"
    logger.warning(
        "X2ROBOT_BRIDGE_URL not set, using default %s (only valid for local mock testing). "
        "For real deployment, set X2ROBOT_BRIDGE_URL=http://<robot_ip>:8766",
        bridge_base_url,
    )

DEFAULT_POLICY_HOST = "192.168.120.73"
DEFAULT_POLICY_PORT = 57770

AUTO_START_DELAY_S = float(os.environ.get("X2ROBOT_AUTO_START_DELAY", "1.0"))


@app.call_tool()
async def fetch_tool(name: str, arguments: dict[str, str]) -> list[types.TextContent]:
    result: list[types.TextContent] = []
    if name == "start_task":
        await start_task(result)
    elif name == "stop_task":
        await stop_task(result)
    elif name == "reset_task":
        await reset_task(result)
    elif name == "set_prompt":
        await set_prompt(result, arguments)
    elif name == "set_evaluate_params":
        await set_evaluate_params(result, arguments)
    elif name == "get_status":
        await get_status(result)
    elif name == "get_prompt":
        await get_prompt(result)
    elif name == "emergency_stop":
        await emergency_stop(result)
    else:
        result.append(types.TextContent(type="text", text=f"非法的工具名请求: {name}"))
    return result


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="start_task",
            description="启动 x2robot 自主执行模式（设置 running_mode=1）。注意：set_evaluate_params 会自动启动任务，通常不需要单独调用此工具",
            inputSchema={
                "type": "object",
                "required": [],
                "properties": {},
            },
        ),
        types.Tool(
            name="stop_task",
            description="停止 x2robot 当前任务（设置 running_mode=0）",
            inputSchema={
                "type": "object",
                "required": [],
                "properties": {},
            },
        ),
        types.Tool(
            name="reset_task",
            description="停止当前任务并重置 x2robot 机械臂到初始位姿",
            inputSchema={
                "type": "object",
                "required": [],
                "properties": {},
            },
        ),
        types.Tool(
            name="set_prompt",
            description="设置 x2robot 的任务提示词",
            inputSchema={
                "type": "object",
                "required": ["prompt"],
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "任务提示词，描述要执行的任务",
                    },
                },
            },
        ),
        types.Tool(
            name="get_prompt",
            description="获取 x2robot 当前的任务提示词",
            inputSchema={
                "type": "object",
                "required": [],
                "properties": {},
            },
        ),
        types.Tool(
            name="set_evaluate_params",
            description="每当用户输入新的任务指令时都需要该工具，用于设置 x2robot 的任务参数（提示词、推理服务器配置等），设置成功后会自动启动任务",
            inputSchema={
                "type": "object",
                "required": ["evaluate_params"],
                "properties": {
                    "evaluate_params": {
                        "type": "object",
                        "description": "评估参数配置",
                        "required": ["prompt"],
                        "properties": {
                            "policy": {
                                "type": "object",
                                "description": "推理服务器配置（可选）",
                                "required": [],
                                "properties": {
                                    "host": {
                                        "type": "string",
                                        "description": f"推理服务器地址（默认: {DEFAULT_POLICY_HOST}）",
                                    },
                                    "port": {
                                        "type": "integer",
                                        "description": f"推理服务器端口（默认: {DEFAULT_POLICY_PORT}）",
                                    },
                                },
                            },
                            "prompt": {
                                "type": "string",
                                "description": "任务提示词，根据用户任务指令描述要执行的任务",
                            },
                            "step_interval": {
                                "type": "number",
                                "description": "执行步间隔（秒），可选",
                            },
                        },
                    },
                },
            },
        ),
        types.Tool(
            name="get_status",
            description="获取 x2robot 的详细状态信息（运行模式、关节位置、推理服务器连接等）",
            inputSchema={
                "type": "object",
                "required": [],
                "properties": {},
            },
        ),
        types.Tool(
            name="emergency_stop",
            description="紧急停止 x2robot（立即设置 running_mode=0）",
            inputSchema={
                "type": "object",
                "required": [],
                "properties": {},
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

async def start_task(result: list[types.TextContent]):
    url = f"{bridge_base_url}/task/start"
    await send_request(result, url, "POST")


async def stop_task(result: list[types.TextContent]):
    url = f"{bridge_base_url}/task/stop"
    await send_request(result, url, "POST")


async def reset_task(result: list[types.TextContent]):
    url = f"{bridge_base_url}/task/reset"
    await send_request(result, url, "POST")


async def set_prompt(result: list[types.TextContent], arguments: dict):
    prompt = arguments.get("prompt")
    if not prompt:
        result.append(types.TextContent(type="text", text="错误: 缺少必需的参数 'prompt'"))
        return
    url = f"{bridge_base_url}/task/set_prompt"
    await send_request(result, url, "POST", {"prompt": prompt})


async def get_prompt(result: list[types.TextContent]):
    url = f"{bridge_base_url}/task/prompt"
    await send_request(result, url, "GET")


async def set_evaluate_params(result: list[types.TextContent], arguments: dict):
    evaluate_params = arguments.get("evaluate_params")
    if not evaluate_params:
        result.append(types.TextContent(type="text", text="错误: 缺少必需的参数 'evaluate_params'"))
        logger.error("set_evaluate_params: missing 'evaluate_params'")
        return

    logger.info("set_evaluate_params: raw params = %s", evaluate_params)

    if "policy" not in evaluate_params:
        evaluate_params["policy"] = {}
    policy = evaluate_params["policy"]
    if "host" not in policy or not policy["host"]:
        policy["host"] = DEFAULT_POLICY_HOST
    if "port" not in policy or not policy["port"]:
        policy["port"] = DEFAULT_POLICY_PORT

    logger.info("set_evaluate_params: processed params = %s", evaluate_params)

    url = f"{bridge_base_url}/task/set_params"
    payload = {"evaluate_params": evaluate_params}
    success = await send_request(result, url, "POST", payload)

    if success:
        logger.info("set_evaluate_params: success, auto-starting after %.1fs", AUTO_START_DELAY_S)
        result.append(types.TextContent(
            type="text",
            text=f"\n[等待 {AUTO_START_DELAY_S} 秒后自动启动任务]",
        ))
        await anyio.sleep(AUTO_START_DELAY_S)
        await start_task(result)
    else:
        logger.error("set_evaluate_params: request failed")


async def get_status(result: list[types.TextContent]):
    url = f"{bridge_base_url}/status"
    await send_request(result, url, "GET")


async def emergency_stop(result: list[types.TextContent]):
    url = f"{bridge_base_url}/task/emergency_stop"
    await send_request(result, url, "POST")


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------

async def send_request(
    result: list[types.TextContent],
    url: str,
    method: str = "POST",
    json_data: dict | None = None,
) -> bool:
    headers = {"Content-Type": "application/json"}
    timeout = httpx.Timeout(timeout=30.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            if method == "GET":
                response_data: Response = await client.get(url, headers=headers)
            else:
                response_data: Response = await client.post(url, headers=headers, json=json_data)

            response_data.raise_for_status()

            try:
                response_json = response_data.json()
                if isinstance(response_json, dict):
                    if "success" in response_json and "data" in response_json:
                        if response_json["success"]:
                            result.append(types.TextContent(
                                type="text",
                                text=f"成功: {response_json.get('data', response_json)}",
                            ))
                            return True
                        else:
                            result.append(types.TextContent(
                                type="text",
                                text=f"失败: {response_json.get('message', response_json.get('data', response_json))}",
                            ))
                            return False
                    else:
                        result.append(types.TextContent(type="text", text=str(response_json)))
                        return True
                else:
                    result.append(types.TextContent(type="text", text=str(response_json)))
                    return True
            except Exception:
                result.append(types.TextContent(type="text", text=response_data.text))
                return True

        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP错误 {e.response.status_code}: {e.response.text}"
            logger.error(error_msg)
            result.append(types.TextContent(type="text", text=error_msg))
            return False
        except httpx.RequestError as e:
            error_msg = f"请求失败: {e}"
            logger.error(error_msg)
            result.append(types.TextContent(
                type="text",
                text=f"连接 x2robot bridge 失败，请确保 bridge 服务正在运行在 {bridge_base_url}",
            ))
            return False
        except Exception as e:
            error_msg = f"未知错误: {e}"
            logger.error(error_msg)
            result.append(types.TextContent(type="text", text=error_msg))
            return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    async def arun():
        async with stdio_server() as streams:
            await app.run(streams[0], streams[1], app.create_initialization_options())

    anyio.run(arun)
    return 0


if __name__ == "__main__":
    sys.exit(main())
