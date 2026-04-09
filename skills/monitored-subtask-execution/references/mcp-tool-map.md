# Robot MCP Tool Map

## Tool naming convention (important)

In this repository, OpenAI function-calling tool names are constructed as:

`{service_name}___{tool_name}`

The separator `___` is defined in:

- `src/agent_demo/types/agent_types/agent_components_types/ormcp_service_types/ormcp_service_types.py`

## Supported robot MCP services

Only one robot service should be active at a time. Check the Server Registry to determine which one is enabled.

### Service 1: `corobot_mcp_server` (Agibot G01 CoRobot)

Implementation: `src/mcp_server_demo/corobot_mcp_server/src/server.py`

This service wraps the local CoRobot PolicyTask HTTP API (default `http://localhost:8765`) and exposes these tools:

- `corobot_mcp_server___set_evaluate_params`
- `corobot_mcp_server___start_task`
- `corobot_mcp_server___stop_task`
- `corobot_mcp_server___reset_task`
- `corobot_mcp_server___get_status`
- `corobot_mcp_server___get_prompt`
- `corobot_mcp_server___set_prompt`

Default policy server: `127.0.0.1:8001`

### Service 2: `x2robot_mcp_server` (Turtle2 x2robot)

Implementation: `src/mcp_server_demo/x2robot_mcp_server/src/server.py`

This service wraps the x2robot bridge HTTP API (default `X2ROBOT_BRIDGE_URL` env var) and exposes these tools:

- `x2robot_mcp_server___set_evaluate_params`
- `x2robot_mcp_server___start_task`
- `x2robot_mcp_server___stop_task`
- `x2robot_mcp_server___reset_task`
- `x2robot_mcp_server___get_status`
- `x2robot_mcp_server___get_prompt`
- `x2robot_mcp_server___set_prompt`
- `x2robot_mcp_server___emergency_stop`

Default inference server: `192.168.0.20:57770`

## `set_evaluate_params` arguments template

The argument structure is the same for both services:

```json
{
  "evaluate_params": {
    "policy": {"host": "<inference_host>", "port": <inference_port>},
    "prompt": "Put the primer into the drawer labeled PRIMER and close it.",
    "step_interval": 1.5
  }
}
```

Notes:

- `policy` can be omitted and the server will fill in defaults.
- On success, `set_evaluate_params` waits briefly and auto-calls `start_task`, so a separate `start_task` call is usually unnecessary.

## MCP client side (service manager)

Service configuration:

- `src/agent_demo/config/ormcp_services.json`

Connection and routing:

- `src/agent_demo/agent_layer/agent_components/ormcp_service_manager/ormcp_service_manager.py`
- `src/agent_demo/agent_layer/agent_components/ormcp_service_manager/ormcp_service_connection.py`

Key behavior:

- Start the MCP server over STDIO with `uv --directory <server_dir> run ...`
- Register tools into the agent tool list through `list_tools()`
- Execute tools and return text results through `call_tool()`, with retry and timeout handling
