# x2robot MCP Server

MCP server for controlling x2robot through the bridge HTTP API.

## Architecture

```
Agent (STDIO MCP) → x2robot_mcp_server → (HTTP) → x2robot_bridge_server (robot side)
```

## Configuration

Set the bridge URL via environment variable:

```bash
export X2ROBOT_BRIDGE_URL=http://<robot_ip>:8766
```

## Tools

| Tool | Description |
|------|-------------|
| `start_task` | Start autonomous execution (running_mode=1) |
| `stop_task` | Stop execution (running_mode=0) |
| `reset_task` | Stop and reset arms to home pose |
| `set_evaluate_params` | Set task params and auto-start |
| `get_status` | Query robot status |
| `set_prompt` | Set task prompt |
| `get_prompt` | Get current prompt |
| `emergency_stop` | Immediate emergency stop |

## Integration

Add to `ormcp_services.json`:

```json
"x2robot_mcp_server": {
    "connection_type": "STDIO",
    "description": {
        "simple_cn": "x2robot 机器人控制服务，通过 bridge 控制机器人执行策略任务，支持启动/停止/重置任务、设置提示词、配置推理服务器等",
        "simple_en": "x2robot control service via bridge HTTP API, supporting start/stop/reset tasks, set prompts, configure inference server, etc."
    },
    "need_activation": true,
    "command": "uv",
    "args": ["--directory", "src/mcp_server_demo/x2robot_mcp_server/src", "run", "server.py"],
    "env": {"X2ROBOT_BRIDGE_URL": "http://<robot_ip>:8766"},
    "url": ""
}
```
